import os
import sys
import numpy as np
from hbnm.io import Data
from hbnm.model.utils import subdiag, fisher_z
from scipy.stats import pearsonr
from optimization import load_data

def get_default_invert_flags(map_names):
    """
    Auto-determine inversion flags based on biological knowledge.
    
    Parameters
    ----------
    map_names : list of str
        Names of biological maps
        
    Returns
    -------
    list of bool
        Inversion flag for each map
        
    Notes
    -----
    Maps that should be INVERTED (negative biological relationship with synaptic strength):
    - T1w/T2w maps: Higher T1w/T2w (sensory) → Lower synaptic strength
    
    Maps that should be DIRECT (positive biological relationship with synaptic strength):
    - NMDA receptor maps: Higher NMDA density → Higher excitatory strength
    - AMPA receptor maps: Higher AMPA density → Higher excitatory strength
    - Glutamate receptor maps: Higher glutamate → Higher excitatory strength
    """
    invert_flags = []
    
    for name in map_names:
        name_lower = name.lower()
        
        # Maps that should be INVERTED (negative biological relationship)
        if any(keyword in name_lower for keyword in ['t1wt2w', 'myelin', 'gaba']):
            invert_flags.append(True)
            print(f"  - {name}: INVERTED (negative correlation with synaptic strength)")
            
        # Maps that should be DIRECT (positive biological relationship)  
        elif any(keyword in name_lower for keyword in ['nmda', 'dopamine', 'norepinephrine']):
            invert_flags.append(False)
            print(f"  - {name}: DIRECT (positive correlation with synaptic strength)")
            
        # Default: assume direct relationship for unknown maps
        else:
            print(f"  Warning: Unknown map type '{name}'. Assuming DIRECT relationship.")
            print(f"           If this is incorrect, use custom invert flags.")
            invert_flags.append(False)
            
    return invert_flags

def validate_maps_and_flags(maps, invert_flags, map_names):
    """
    Bulletproof validation with clear error messages.
    
    Parameters
    ----------
    maps : ndarray
        Biological maps matrix (n_maps, n_regions)
    invert_flags : list of bool
        Inversion flags for each map
    map_names : list of str
        Names of the maps
        
    Raises
    ------
    ValueError
        If validation fails
    """
    # Check counts match
    if len(invert_flags) != maps.shape[0]:
        raise ValueError(f"Mismatch: {len(invert_flags)} invert flags for {maps.shape[0]} maps")
        
    if len(map_names) != maps.shape[0]:
        raise ValueError(f"Mismatch: {len(map_names)} map names for {maps.shape[0]} maps")
        
    # Check for problematic maps
    for i, (name, flag, map_data) in enumerate(zip(map_names, invert_flags, maps)):
        if np.ptp(map_data) == 0:
            raise ValueError(f"Map '{name}' (index {i}) is constant - cannot normalize")
            
        if np.any(np.isnan(map_data)):
            raise ValueError(f"Map '{name}' (index {i}) contains NaN values")
            
        if np.any(np.isinf(map_data)):
            raise ValueError(f"Map '{name}' (index {i}) contains infinite values")
            
    print("✓ Map validation passed")
    print(f"  Maps: {map_names}")
    print(f"  Invert flags: {invert_flags}")

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
        maps = hmap[None, :]  # Shape: (1, n_regions)
        
        # T1w/T2w should be inverted (Demirtas methodology)
        invert_flags = [True]
        map_names = ['t1wt2w']
        
        print(f"Loaded default T1w/T2w map (backwards compatible)")
        print(f"Invert flags: {invert_flags}")
        
        return maps, invert_flags
    
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
    
    # Generate invert flags based on map names
    print(f"\nDetermining invert flags based on biological knowledge:")
    invert_flags = get_default_invert_flags(map_names)
    
    # Validate everything
    validate_maps_and_flags(maps, invert_flags, map_names)
    
    return maps, invert_flags

def parse_custom_priors(priors_string, model_type, maps_path):
    """
    Parse custom priors string into dictionary format.
    
    Parameters
    ----------
    priors_string : str
        Semicolon-separated parameter ranges
    model_type : str
        Type of model being used
    maps_path : str  
        Path to maps (used to determine if homogeneous)
    
    Returns
    -------
    dict
        Custom priors dictionary
    """
    # Determine if this is homogeneous or heterogeneous
    is_homogeneous = (model_type.lower() == 'homogeneous' or maps_path == "None")
    
    # Split by semicolons to get parameter ranges
    param_ranges = priors_string.split(';')
    
    if is_homogeneous:
        # Homogeneous: w_EI_bias, w_EE_bias, G
        if len(param_ranges) != 3:
            raise ValueError(f"Homogeneous model requires 3 parameter ranges, got {len(param_ranges)}")
        
        keys = ['w_EI_bias', 'w_EE_bias', 'G']
    else:
        # Heterogeneous: w_EI_bias, w_EI_coeff, w_EE_bias, w_EE_coeff, G  
        if len(param_ranges) != 5:
            raise ValueError(f"Heterogeneous model requires 5 parameter ranges, got {len(param_ranges)}")
        
        keys = ['w_EI_bias', 'w_EI_coeff', 'w_EE_bias', 'w_EE_coeff', 'G']
    
    # Parse each range
    custom_priors = {}
    for i, (key, range_str) in enumerate(zip(keys, param_ranges)):
        try:
            min_val, max_val = range_str.split(',')
            min_val, max_val = float(min_val), float(max_val)
            
            if min_val >= max_val:
                raise ValueError(f"Invalid range for {key}: min ({min_val}) >= max ({max_val})")
                
            custom_priors[key] = (min_val, max_val)
        except ValueError as e:
            raise ValueError(f"Error parsing range for parameter {key}: {e}")
    
    return custom_priors

if __name__ == '__main__':
    """
    Enhanced multi-map optimization script with custom prior support.
    
    Arguments:
    1- model type: 'homogeneous', 'heterogeneous', 'multimap', or number of maps (e.g., '6')
    2- minimum number of samples / sampler
    3- number of samplers
    4- sampler id
    5- optimization task: 'sampler' or 'wrapper'
    6- append to output directory
    7- functional connectivity object, must be a numpy array (.npy) # eventually an array of arrays for hPMC
    8- maps object, must be a numpy array (.npy), set to "None" for homogeneous model
    9- (optional) linearize flag: 'linearize' or 'no_linearize' (default: 'no_linearize')
    10- (optional) invert flags: bool value, comma separated
    11- (optional) custom priors: ranges for parameters in format:
        For homogeneous: "w_EI_bias_min,w_EI_bias_max;w_EE_bias_min,w_EE_bias_max;G_min,G_max"
        For heterogeneous: "w_EI_bias_min,w_EI_bias_max;w_EI_coeff_min,w_EI_coeff_max;w_EE_bias_min,w_EE_bias_max;w_EE_coeff_min,w_EE_coeff_max;G_min,G_max"
    
    Examples:
    python optimization_tms_fmri_data.py homogeneous 1000 10 0 sampler test_homo fc.npy None no_linearize invert "0.001,5.0;0.001,15.0;0.001,5.0"
    python optimization_tms_fmri_data.py heterogeneous 1000 10 0 sampler test_hetero fc.npy maps.npy no_linearize invert "0.001,2.0;0.0,2.5;0.001,5.0;0.0,15.0;0.001,5.0"
    python optimization_tms_fmri_data.py multimap 1000 10 0 sampler test_multi fc.npy maps.npy no_linearize invert "0.001,2.0;0.0,2.5;0.001,5.0;0.0,15.0;0.001,5.0"
    """

    # Parse arguments
    model_type = sys.argv[1]
    n_samples = int(sys.argv[2])
    n_samplers = int(sys.argv[3])
    sampler_id = int(sys.argv[4])
    task = sys.argv[5]
    append_directory = sys.argv[6]
    fc_vector_path = sys.argv[7]
    maps_path = sys.argv[8]

    # Optional: biological map names
    linearize_maps = False
    custom_invert_flags = None
        
    if len(sys.argv) > 9:
        linearize_flag = sys.argv[9].lower()
        if linearize_flag == 'linearize':
            linearize_maps = True
        elif linearize_flag == 'no_linearize':
            linearize_maps = False
        else:
            print(f"Warning: Unknown linearize flag '{sys.argv[9]}'. Using default (no_linearize).")
    
    # Optional: custom invert flags (10th argument)
    if len(sys.argv) > 10:
        try:
            custom_flags = sys.argv[10].split(',')
            invert_flags = []
            for flag in custom_flags:
                flag_lower = flag.lower().strip()
                if flag_lower in ['true', 't', '1', 'yes', 'invert']:
                    invert_flags.append(True)
                elif flag_lower in ['false', 'f', '0', 'no', 'direct']:
                    invert_flags.append(False)
                else:
                    raise ValueError(f"Invalid invert flag: '{flag}'. Use true/false.")
            print(f"Using custom invert flags: {invert_flags}")
        except Exception as e:
            print(f"Error parsing custom invert flags: {e}")
            print("Using smart defaults instead.")
            invert_flags = None
    
    # Optional: custom priors (11th argument) - MANDATORY if provided
    custom_priors = None
    if len(sys.argv) > 11:
        try:
            priors_string = sys.argv[11]
            custom_priors = parse_custom_priors(priors_string, model_type, maps_path)
            print(f"Using custom priors: {custom_priors}")
        except Exception as e:
            print(f"Error parsing custom priors: {e}")
            raise ValueError(f"Invalid custom priors specification. Please check format.")
    
    # Validate that custom priors are provided (mandatory requirement)
    if custom_priors is None:
        raise ValueError("Custom priors are mandatory. Please specify prior ranges as the 11th argument.")

    # Set directories
    current_path = os.getcwd()
    parent_path = os.path.abspath(os.path.join(current_path, os.pardir))
    input_dir = parent_path + '/data/'
    output_dir = parent_path + '/outputs/'

    # Load structural connectivity and functional connectivity
    data = Data(input_dir, output_dir)
    sc, hmap, fc_obj = load_data(data)
    # fc_obj = fisher_z(subdiag(fc_obj))

    # NOTE Temporary fix - This is technical debt
    # Override fc_obj with the FC from
    # Load FC from group means file
    fc = np.load(fc_vector_path)
    fc_obj = fc[:180,:180]
    fc_obj = fisher_z(subdiag(fc_obj))
    
    rejection_threshold = 1.0 - pearsonr(fc_obj, subdiag(sc))[0]

    # Determine optimization class and parameters based on model type
    if model_type.lower() == 'homogeneous':
        # Homogeneous model (no maps)
        from optimization import Homogeneous
        
        pmc_opt = Homogeneous(input_dir, output_dir + append_directory + '/')
        pmc_opt.initialize(sc, fc=fc_obj, n_particles=n_samples,
                          rejection_threshold=rejection_threshold, 
                          custom_priors=custom_priors, norm_sc=True)
        
        print(f"Initialized homogeneous model with {len(pmc_opt.prior)} parameters")
        
    elif model_type.lower() == 'heterogeneous':
        # Single-map heterogeneous model (backwards compatible)
        from optimization import Heterogeneous
        
        pmc_opt = Heterogeneous(input_dir, output_dir + append_directory + '/')
        pmc_opt.initialize(sc, fc=fc_obj, gradient=hmap, n_particles=n_samples,
                          rejection_threshold=rejection_threshold, 
                          custom_priors=custom_priors, norm_sc=True)
        
        print(f"Initialized single-map heterogeneous model with {len(pmc_opt.prior)} parameters")
        
    elif model_type.lower() == 'multimap' or model_type.isdigit():
        # Multi-map model using new MultiMapHeterogeneous class
        from optimization import MultiMapHeterogeneous
                
        # Provide an option to not provide maps and have it set the maps variable to None
        if maps_path == "None":
            print(f"!!! Maps path is None, using the Homogeneous model")
            maps = None
            invert_flags = None
        else:
            maps = np.load(maps_path)
            # Check the dimensions of the array, if it is not (n_maps, n_regions), raise an error
            if maps.shape[0] != len(invert_flags) or maps.shape[1] != sc.shape[0]:
                raise ValueError(f"Maps array has {maps.shape[0]} maps and {maps.shape[1]} regions, but expected {len(invert_flags)} maps and {sc.shape[0]} regions")

        pmc_opt = MultiMapHeterogeneous(input_dir, output_dir + append_directory + '/')
        pmc_opt.initialize(sc, fc=fc_obj, maps=maps, map_invert_flags=invert_flags, n_particles=n_samples,
                          rejection_threshold=rejection_threshold, 
                          custom_priors=custom_priors, norm_sc=True)
        
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
