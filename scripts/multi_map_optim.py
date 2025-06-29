import os
import sys
import numpy as np
from hbnm.io import Data
from hbnm.model.utils import subdiag, fisher_z
from scipy.stats import pearsonr
from optimization import load_data

def load_biological_maps(data, map_names=None, linearize=False):
    """
    Load biological maps for multi-map optimization.
    
    Parameters
    ----------
    data : Data
        HBNM data object
    map_names : list of str, optional
        Names of biological maps to load. These should correspond to files in the
        data/maps/ directory (without .pscalar.nii extension).
        If None, only loads the default T1w/T2w map.
    linearize : bool, optional
        If True, applies the same linearization (erf transformation) as used for T1w/T2w.
        If False (default), uses maps as-is.
        
    Returns
    -------
    maps : ndarray
        Biological maps matrix of shape (n_maps, n_regions).
        
    Notes
    -----
    - T1w/T2w map is loaded from HDF5 and linearized using erf transformation
    - Biological maps are loaded from data/maps/ as CIFTI files
    - All maps are extracted for left hemisphere only (first 180 regions)
    """
    from hbnm.model.utils import linearize_map
    import nibabel as nib
    
    if map_names is None:
        # Default: only T1w/T2w map (backwards compatible)
        _, hmap, _ = load_data(data)
        return hmap[None, :]  # Shape: (1, n_regions)
    
    maps_list = []
    maps_dir = os.path.join(data.input_dir, 'maps')
    
    for map_name in map_names:
        # Check if this is the special T1w/T2w map
        if map_name.lower() in ['t1wt2w', 't1w_t2w', 't1w/t2w']:
            # Load T1w/T2w from HDF5 (already linearized by load_data)
            _, hmap, _ = load_data(data)
            maps_list.append(hmap)
            print(f"  - {map_name}: loaded from HDF5 (linearized)")
            
        else:
            # Load biological map from maps folder
            map_file = f"{map_name}.pscalar.nii"
            map_path = os.path.join(maps_dir, map_file)
            
            if os.path.exists(map_path):
                # Load CIFTI file
                cifti_img = nib.load(map_path)
                map_data = cifti_img.get_fdata()
                
                # Handle 2D data (take first row if needed)
                if map_data.ndim == 2:
                    map_data = map_data[0, :]
                
                # Extract left hemisphere (first 180 regions)
                map_left = map_data[:180]
                
                # Apply linearization if requested
                if linearize:
                    map_left = linearize_map(map_left)
                    print(f"  - {map_name}: loaded from maps/ (linearized)")
                else:
                    print(f"  - {map_name}: loaded from maps/ (as-is)")
                
                maps_list.append(map_left)
                
                # Show statistics
                mean_val = map_left.mean()
                std_val = map_left.std()
                print(f"    mean={mean_val:.4f}, std={std_val:.4f}")
                
            else:
                print(f"Warning: Map file '{map_file}' not found in {maps_dir}. Skipping.")
    
    if not maps_list:
        raise ValueError("No valid biological maps found!")
    
    # Stack maps
    maps = np.vstack(maps_list)
    print(f"Loaded {len(maps_list)} biological maps with {maps.shape[1]} regions each")
    
    return maps

if __name__ == '__main__':
    """
    Enhanced multi-map optimization script.
    
    Arguments:
    1- model type: 'homogeneous', 'heterogeneous', 'multimap', or number of maps (e.g., '6')
    2- minimum number of samples / sampler
    3- number of samplers
    4- sampler id
    5- optimization task: 'sampler' or 'wrapper'
    6- append to output directory
    7- (optional) biological map names separated by commas (e.g., 'NMDA_avg,GRIN1,GRIN2A')
    8- (optional) linearize flag: 'linearize' or 'no_linearize' (default: 'no_linearize')
    
    Examples:
    python multi_map_optim.py homogeneous 1000 10 0 sampler test_homo
    python multi_map_optim.py heterogeneous 1000 10 0 sampler test_hetero
    python multi_map_optim.py multimap 1000 10 0 sampler test_multi NMDA_avg,GRIN1,GRIN2A
    python multi_map_optim.py multimap 1000 10 0 sampler test_multi NMDA_avg,GRIN1,GRIN2A linearize
    python multi_map_optim.py 6 1000 10 0 sampler test_6maps NMDA_avg,GRIN1,GRIN2A,GRIN2B,GRIN2C,GRIN2D
    """

    # Parse arguments
    model_type = sys.argv[1]
    n_samples = int(sys.argv[2])
    n_samplers = int(sys.argv[3])
    sampler_id = int(sys.argv[4])
    task = sys.argv[5]
    append_directory = sys.argv[6]
    
    # Optional: biological map names
    biological_maps = None
    linearize_maps = False
    
    if len(sys.argv) > 7:
        map_names = sys.argv[7].split(',')
        biological_maps = map_names
    
    if len(sys.argv) > 8:
        linearize_flag = sys.argv[8].lower()
        if linearize_flag == 'linearize':
            linearize_maps = True
        elif linearize_flag == 'no_linearize':
            linearize_maps = False
        else:
            print(f"Warning: Unknown linearize flag '{sys.argv[8]}'. Using default (no_linearize).")

    # Set directories
    current_path = os.getcwd()
    parent_path = os.path.abspath(os.path.join(current_path, os.pardir))
    input_dir = parent_path + '/data/'
    output_dir = parent_path + '/outputs/'

    # Load structural connectivity and functional connectivity
    data = Data(input_dir, output_dir)
    sc, hmap, fc_obj = load_data(data)
    
    fc_obj = fisher_z(subdiag(fc_obj))
    rejection_threshold = 1.0 - pearsonr(fc_obj, subdiag(sc))[0]

    # Determine optimization class and parameters based on model type
    if model_type.lower() == 'homogeneous':
        # Homogeneous model (no maps)
        from optimization import Homogeneous
        
        pmc_opt = Homogeneous(input_dir, output_dir + append_directory + '/')
        pmc_opt.initialize(sc, fc=fc_obj, n_particles=n_samples,
                          rejection_threshold=rejection_threshold, norm_sc=True)
        
        print(f"Initialized homogeneous model with {len(pmc_opt.prior)} parameters")
        
    elif model_type.lower() == 'heterogeneous':
        # Single-map heterogeneous model (backwards compatible)
        from optimization import Heterogeneous
        
        pmc_opt = Heterogeneous(input_dir, output_dir + append_directory + '/')
        pmc_opt.initialize(sc, fc=fc_obj, gradient=hmap, n_particles=n_samples,
                          rejection_threshold=rejection_threshold, norm_sc=True)
        
        print(f"Initialized single-map heterogeneous model with {len(pmc_opt.prior)} parameters")
        
    elif model_type.lower() == 'multimap' or model_type.isdigit():
        # Multi-map model using new MultiMapHeterogeneous class
        from optimization import MultiMapHeterogeneous
        
        # Load biological maps
        if biological_maps is not None:
            print(f"Loading biological maps: {biological_maps}")
            print(f"Linearization: {'enabled' if linearize_maps else 'disabled'}")
            maps = load_biological_maps(data, biological_maps, linearize=linearize_maps)
        else:
            # Default to single T1w/T2w map if no maps specified
            print("No biological maps specified, using default T1w/T2w map")
            maps = hmap[None, :]
        
        # If model_type is a number, verify it matches the number of maps
        if model_type.isdigit():
            expected_maps = int(model_type)
            if maps.shape[0] != expected_maps:
                raise ValueError(f"Expected {expected_maps} maps but loaded {maps.shape[0]} maps")
        
        pmc_opt = MultiMapHeterogeneous(input_dir, output_dir + append_directory + '/')
        pmc_opt.initialize(sc, fc=fc_obj, maps=maps, n_particles=n_samples,
                          rejection_threshold=rejection_threshold, norm_sc=True)
        
        print(f"Initialized multi-map model with:")
        print(f"  - {pmc_opt.n_maps} biological maps")
        print(f"  - {len(pmc_opt.prior)} total parameters")
        print(f"  - Parameter breakdown:")
        print(f"    * w_EI: 1 bias + {pmc_opt.n_maps} coefficients")
        print(f"    * w_EE: 1 bias + {pmc_opt.n_maps} coefficients")
        print(f"    * G: 1 parameter")
        
    else:
        raise ValueError(f"Unknown model type: {model_type}. "
                        f"Use 'homogeneous', 'heterogeneous', 'multimap', or a number.")

    # Print optimization setup
    print(f"\nOptimization setup:")
    print(f"  - Model type: {model_type}")
    print(f"  - Samples per sampler: {n_samples}")
    print(f"  - Number of samplers: {n_samplers}")
    print(f"  - Current sampler ID: {sampler_id}")
    print(f"  - Task: {task}")
    print(f"  - Output directory: {output_dir + append_directory + '/'}")
    print(f"  - Rejection threshold: {rejection_threshold:.4f}")

    # Execute the requested task
    if task == 'sampler':
        print(f"\nRunning sampler {sampler_id}...")
        pmc_opt.run(sampler_id)
        print(f"Sampler {sampler_id} completed successfully!")
        
    elif task == 'wrapper':
        print(f"\nWrapping results from {n_samplers} samplers...")
        pmc_opt.wrap(n_samplers)
        print(f"Wrapper completed successfully!")
        
    else:
        raise NotImplementedError("The task should be 'sampler' or 'wrapper'")

    print(f"\nOptimization task '{task}' completed successfully!")
