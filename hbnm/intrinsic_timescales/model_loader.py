# hbnm/intrinsic_timescales/model_loader.py

import os
import glob
import numpy as np
import h5py
from hbnm.io import Data
from hbnm.model.utils import linearize_map
import nibabel as nib

class ModelLoader:
    """
    Handles loading and configuring brain network models with Bayesian averaging.
    Supports both homogeneous and heterogeneous models.
    """
    
    def __init__(self, config):
        """Initialize with configuration dictionary."""
        self.config = config
        self.data = Data(config['paths']['input_dir'], config['paths']['output_dir'])
        
        # Will be populated by load_model()
        self.model_params = None
        self.model_type = None
        self.maps = None
        self.map_invert_flags = None
        self.sc = None
        self.latest_iteration = None
        
    def _find_latest_iteration(self, folder_path, pattern):
        """Find the latest iteration file in the specified folder."""
        search_path = os.path.join(self.data.output_dir, folder_path, pattern)
        iteration_files = glob.glob(search_path)
        
        if not iteration_files:
            return None
        
        # Extract iteration numbers and find the maximum
        iteration_numbers = []
        for file_path in iteration_files:
            filename = os.path.basename(file_path)
            try:
                # Extract number from "iteration_XX.hdf5"
                iter_num = int(filename.split('_')[1].split('.')[0])
                iteration_numbers.append((iter_num, file_path))
            except (ValueError, IndexError):
                continue
        
        if iteration_numbers:
            latest_iter, latest_file = max(iteration_numbers, key=lambda x: x[0])
            relative_path = os.path.relpath(latest_file, self.data.output_dir)
            return latest_iter, relative_path
        
        return None
    
    def _compute_bayesian_average(self, theta_samples, weights):
        """Compute Bayesian Model Average from samples and weights."""
        normalized_weights = weights / np.sum(weights)
        theta_bma = np.sum(theta_samples * normalized_weights[np.newaxis, :], axis=1)
        return theta_bma
    
    def _detect_model_type(self, n_params):
        """Detect model type based on number of parameters."""
        if n_params == 3:
            return "homogeneous"
        elif n_params == 5:
            return "heterogeneous"
        else:
            raise ValueError(f"Unsupported parameter count: {n_params}. Expected 3 (homogeneous) or 5 (heterogeneous)")
    
    def _format_parameters(self, theta_bma, model_type):
        """Format parameters into the correct structure for the model type."""
        if model_type == "homogeneous":
            return {
                'w_EI': theta_bma[0],
                'w_EE': theta_bma[1], 
                'G': theta_bma[2]
            }
        elif model_type == "heterogeneous":
            return {
                'w_EI': (theta_bma[0], theta_bma[1]),  # (baseline, scaling)
                'w_EE': (theta_bma[2], theta_bma[3]),  # (baseline, scaling)
                'G': theta_bma[4]
            }
    
    def _load_custom_map(self, map_path):
        """Load and process custom map from various formats."""
        if map_path.endswith('.npy'):
            # Already linearized numpy array
            print('WARNING: This function assumes that your custom map has already been linearized!')
            return np.load(map_path)
        elif map_path.endswith('.pscalar.nii'):
            # CIFTI pscalar file
            img = nib.load(map_path)
            map_data = img.get_fdata().squeeze()
            return linearize_map(map_data)
        else:
            raise ValueError(f"Unsupported map format: {map_path}")
    
    def _load_structural_connectivity(self):
        """Load structural connectivity matrix."""
        sc_path = self.config['model'].get('sc_path')
        if sc_path:
            print('WARNING: This function assumes that your SC matrix has already been normalized!')
            return np.load(sc_path)
        else:
            # Use default from data loader
            sc, _, _ = self.data.load_demirtas_neuron_2019_data()
            return sc
    
    def load_model(self):
        """
        Complete model loading workflow using configuration.
        
        Returns:
            dict: Contains all loaded model components
        """
        print("=== LOADING MODEL WITH CONFIG ===")
        
        # 1. Load structural connectivity
        print("Step 1: Loading structural connectivity...")
        self.sc = self._load_structural_connectivity()
        nc = self.sc.shape[0]
        print(f"  Regions: {nc}")
        
        # 2. Determine which iteration file to load
        print("Step 2: Finding model parameters...")
        model_config = self.config['model']
        
        if model_config['use_latest']:
            result = self._find_latest_iteration(model_config['folder'], 'iteration_*.hdf5')
            if result:
                self.latest_iteration, file_path = result
                print(f"  ✓ Auto-detected latest iteration: {self.latest_iteration}")
            else:
                file_path = f"{model_config['folder']}/iteration_{model_config['fallback_iteration']}.hdf5"
                self.latest_iteration = model_config['fallback_iteration']
                print(f"  ⚠ Auto-detection failed, using fallback: iteration {self.latest_iteration}")
        else:
            file_path = f"{model_config['folder']}/iteration_{model_config['fallback_iteration']}.hdf5"
            self.latest_iteration = model_config['fallback_iteration']
            print(f"  ✓ Using specified iteration: {self.latest_iteration}")
        
        # 3. Load and process model parameters
        print(f"Step 3: Loading parameters from {file_path}...")
        full_path = os.path.join(self.data.output_dir, file_path)
        
        with h5py.File(full_path, 'r') as fin_model:
            theta_samples = fin_model['theta'][:]
            weights = fin_model['weights'][:]
        
        # 4. Compute Bayesian Model Average
        print("Step 4: Computing Bayesian Model Average...")
        theta_bma = self._compute_bayesian_average(theta_samples, weights)
        
        # 5. Detect model type and format parameters
        print("Step 5: Formatting parameters...")
        n_params = theta_samples.shape[0]
        self.model_type = self._detect_model_type(n_params)
        self.model_params = self._format_parameters(theta_bma, self.model_type)
        
        print(f"  Model type: {self.model_type}")
        print(f"  Samples: {theta_samples.shape[1]} | Parameters: {n_params}")
        
        # 6. Load maps (for heterogeneous models)
        if self.model_type == "heterogeneous":
            print("Step 6: Loading heterogeneity maps...")
            map_path = model_config['map_path']
            self.maps = self._load_custom_map(map_path)
            self.map_invert_flags = model_config['invert_flags']
            print(f"  Map loaded: {self.maps.shape} | Invert flags: {self.map_invert_flags}")
        else:
            self.maps = None
            self.map_invert_flags = None
            print("Step 6: No maps needed (homogeneous model)")
        
        # 7. Print summary
        self._print_summary()
        
        return {
            'sc': self.sc,
            'model_params': self.model_params,
            'model_type': self.model_type,
            'maps': self.maps,
            'map_invert_flags': self.map_invert_flags,
            'latest_iteration': self.latest_iteration,
            'n_regions': nc
        }
    
    def _print_summary(self):
        """Print summary of loaded model."""
        print(f"\n✓ {self.model_type.title()} model loaded successfully!")
        
        if self.model_type == "homogeneous":
            print(f"  Parameters: w_EI={self.model_params['w_EI']:.4f}, "
                  f"w_EE={self.model_params['w_EE']:.4f}, G={self.model_params['G']:.4f}")
        elif self.model_type == "heterogeneous":
            print(f"  Parameters:")
            print(f"    w_EI: ({self.model_params['w_EI'][0]:.4f}, {self.model_params['w_EI'][1]:.4f}) [baseline, scaling]")
            print(f"    w_EE: ({self.model_params['w_EE'][0]:.4f}, {self.model_params['w_EE'][1]:.4f}) [baseline, scaling]")
            print(f"    G: {self.model_params['G']:.4f}")