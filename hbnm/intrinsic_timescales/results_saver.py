# hbnm/intrinsic_timescales/results_saver.py

"""
Results Storage Module for Intrinsic Timescales Analysis

This module handles efficient storage of complete experiment results in HDF5 format.
Saves configuration metadata, raw simulation results, processed data, and fitted
timescale parameters in a structured, compressed format for long-term storage.
"""

import h5py
import numpy as np
import os
from datetime import datetime
import json


class ResultsSaver:
    """
    Saves complete intrinsic timescales experiment results to HDF5 format.
    
    Provides efficient, structured storage of all experiment outputs including
    configuration metadata, raw simulation data, processed time series, and
    fitted exponential decay parameters.
    """
    
    def __init__(self, config):
        """Initialize with configuration dictionary."""
        self.config = config
        
        # Storage options (can be extended to config later)
        self.compression = 'gzip'
        self.compression_opts = 6  # Compression level (0-9)
        self.shuffle = True        # Improves compression
        self.fletcher32 = True     # Error detection
        
    def save_experiment(self, raw_results, decay_data, timescale_results, output_path):
        """
        Save complete experiment results to HDF5 file.
        
        Parameters:
        -----------
        raw_results : np.ndarray
            Raw simulation results with shape (n_trials, n_regions, n_timepoints)
        decay_data : dict
            Processed decay data from DataProcessor:
            {
                'time_relative': np.ndarray,
                'firing_rates': np.ndarray
            }
        timescale_results : np.ndarray
            Structured array from CurveFitter with fitted parameters
        output_path : str
            Full path for output HDF5 file
            
        Returns:
        --------
        str
            Path to saved file
        """
        print("=== SAVING EXPERIMENT RESULTS ===")
        
        # 1. Validate inputs
        self._validate_inputs(raw_results, decay_data, timescale_results, output_path)
        
        # 2. Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 3. Save to HDF5 with comprehensive structure
        with h5py.File(output_path, 'w') as f:
            
            # Save metadata
            print("Step 1: Saving experiment metadata...")
            self._save_metadata(f)
            
            # Save raw simulation results
            print("Step 2: Saving raw simulation results...")
            self._save_raw_results(f, raw_results)
            
            # Save processed decay data
            print("Step 3: Saving processed decay data...")
            self._save_processed_data(f, decay_data)
            
            # Save timescale fitting results
            print("Step 4: Saving timescale fitting results...")
            self._save_timescale_results(f, timescale_results)
            
            # Add file-level metadata
            self._add_file_metadata(f, raw_results, decay_data, timescale_results)
        
        # 4. Report storage efficiency
        self._report_storage_info(output_path, raw_results)
        
        print(f"✓ Experiment results saved successfully!")
        print(f"  File: {output_path}")
        
        return output_path
    
    def load_experiment(self, filepath):
        """
        Load complete experiment results from HDF5 file.
        
        Parameters:
        -----------
        filepath : str
            Path to saved HDF5 file
            
        Returns:
        --------
        dict
            Complete experiment data:
            {
                'metadata': dict,
                'raw_results': np.ndarray,
                'decay_data': dict,
                'timescale_results': np.ndarray
            }
        """
        print(f"=== LOADING EXPERIMENT RESULTS ===")
        print(f"Loading from: {filepath}")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Results file not found: {filepath}")
        
        with h5py.File(filepath, 'r') as f:
            
            # Load metadata
            metadata = self._load_metadata(f)
            
            # Load raw results
            raw_results = f['raw_results'][:]
            
            # Load processed data
            decay_data = {
                'time_relative': f['processed_data/time_relative'][:],
                'firing_rates': f['processed_data/firing_rates'][:]
            }
            
            # Load timescale results
            timescale_results = f['timescale_results'][:]
            
            print(f"✓ Experiment results loaded successfully!")
            print(f"  Raw results: {raw_results.shape}")
            print(f"  Processed data: {decay_data['firing_rates'].shape}")
            print(f"  Timescale results: {timescale_results.shape}")
            
            return {
                'metadata': metadata,
                'raw_results': raw_results,
                'decay_data': decay_data, 
                'timescale_results': timescale_results
            }
    
    # PRIVATE METHODS
    
    def _validate_inputs(self, raw_results, decay_data, timescale_results, output_path):
        """Validate all inputs before saving."""
        # Validate raw results
        if not isinstance(raw_results, np.ndarray) or raw_results.ndim != 3:
            raise ValueError("raw_results must be 3D numpy array (trials, regions, timepoints)")
        
        # Validate decay data
        required_keys = {'time_relative', 'firing_rates'}
        if not isinstance(decay_data, dict) or not required_keys.issubset(decay_data.keys()):
            raise ValueError(f"decay_data must be dict with keys: {required_keys}")
        
        if decay_data['firing_rates'].shape[0] != raw_results.shape[1]:
            raise ValueError("Number of regions must match between raw_results and decay_data")
        
        # Validate timescale results
        if not isinstance(timescale_results, np.ndarray):
            raise ValueError("timescale_results must be structured numpy array")
        
        if len(timescale_results) != raw_results.shape[1]:
            raise ValueError("Number of regions must match between raw_results and timescale_results")
        
        # Validate output path
        if not output_path.endswith('.hdf5'):
            raise ValueError("output_path must end with .hdf5")
    
    def _save_metadata(self, f):
        """Save experiment configuration metadata."""
        metadata_group = f.create_group('metadata')
        
        # Experiment info
        exp_info = metadata_group.create_group('experiment_info')
        exp_info.attrs['timestamp'] = datetime.now().isoformat()
        exp_info.attrs['experiment_name'] = self.config.get('experiment', {}).get('name', 'unknown')
        
        # Model configuration
        model_group = metadata_group.create_group('model_config')
        model_config = self.config.get('model', {})
        for key, value in model_config.items():
            if isinstance(value, (str, int, float, bool)):
                model_group.attrs[key] = value
            elif isinstance(value, list):
                model_group.attrs[key] = json.dumps(value)
        
        # Simulation configuration
        sim_group = metadata_group.create_group('simulation_config')
        sim_config = self.config.get('stimulation', {})
        for key, value in sim_config.items():
            if isinstance(value, (str, int, float, bool)):
                sim_group.attrs[key] = value
        
        # Parallel configuration
        parallel_group = metadata_group.create_group('parallel_config')
        parallel_config = self.config.get('parallel', {})
        for key, value in parallel_config.items():
            parallel_group.attrs[key] = value
        
        # Fitting configuration
        fitting_group = metadata_group.create_group('fitting_config')
        fitting_config = self.config.get('curve_fitting', {})
        
        # Handle nested parameter bounds
        if 'parameter_bounds' in fitting_config:
            bounds = fitting_config['parameter_bounds']
            for param, bound_list in bounds.items():
                fitting_group.attrs[f'{param}_bounds'] = bound_list
        
        # Other fitting parameters
        for key, value in fitting_config.items():
            if key != 'parameter_bounds' and isinstance(value, (str, int, float, bool)):
                fitting_group.attrs[key] = value
    
    def _save_raw_results(self, f, raw_results):
        """Save raw simulation results with compression."""
        print(f"  Raw results shape: {raw_results.shape}")
        print(f"  Data size: {raw_results.nbytes / 1024**2:.1f} MB")
        
        f.create_dataset(
            'raw_results',
            data=raw_results,
            compression=self.compression,
            compression_opts=self.compression_opts,
            shuffle=self.shuffle,
            fletcher32=self.fletcher32,
            chunks=True
        )
    
    def _save_processed_data(self, f, decay_data):
        """Save processed decay period data."""
        processed_group = f.create_group('processed_data')
        
        # Time relative to stimulation end
        processed_group.create_dataset(
            'time_relative',
            data=decay_data['time_relative'],
            compression=self.compression,
            compression_opts=self.compression_opts
        )
        
        # Firing rates during decay period
        print(f"  Processed data shape: {decay_data['firing_rates'].shape}")
        processed_group.create_dataset(
            'firing_rates',
            data=decay_data['firing_rates'],
            compression=self.compression,
            compression_opts=self.compression_opts,
            shuffle=self.shuffle,
            fletcher32=self.fletcher32,
            chunks=True
        )
    
    def _save_timescale_results(self, f, timescale_results):
        """Save fitted timescale parameters and quality metrics."""
        print(f"  Timescale results: {len(timescale_results)} regions")
        
        # Count successful fits
        n_successful = np.sum(timescale_results['success'])
        print(f"  Successful fits: {n_successful}/{len(timescale_results)} ({100*n_successful/len(timescale_results):.1f}%)")
        
        f.create_dataset(
            'timescale_results',
            data=timescale_results,
            compression=self.compression,
            compression_opts=self.compression_opts,
            fletcher32=self.fletcher32
        )
    
    def _add_file_metadata(self, f, raw_results, decay_data, timescale_results):
        """Add summary metadata to file root."""
        # Data dimensions
        f.attrs['n_trials'] = raw_results.shape[0]
        f.attrs['n_regions'] = raw_results.shape[1] 
        f.attrs['n_timepoints_full'] = raw_results.shape[2]
        f.attrs['n_timepoints_decay'] = decay_data['firing_rates'].shape[1]
        
        # Fitting summary
        n_successful = np.sum(timescale_results['success'])
        f.attrs['n_successful_fits'] = n_successful
        f.attrs['fitting_success_rate'] = n_successful / len(timescale_results)
        
        # Timescale statistics for successful fits
        if n_successful > 0:
            successful_mask = timescale_results['success']
            timescales = timescale_results[successful_mask]['timescale']
            f.attrs['mean_timescale'] = np.mean(timescales)
            f.attrs['std_timescale'] = np.std(timescales)
            f.attrs['min_timescale'] = np.min(timescales)
            f.attrs['max_timescale'] = np.max(timescales)
        
        # File creation info
        f.attrs['creation_timestamp'] = datetime.now().isoformat()
        f.attrs['hbnm_version'] = "0.1.0"  # From __init__.py
    
    def _load_metadata(self, f):
        """Load experiment metadata from HDF5 file."""
        metadata = {}
        
        if 'metadata' in f:
            metadata_group = f['metadata']
            
            # Load all metadata groups
            for group_name in metadata_group.keys():
                group = metadata_group[group_name]
                group_data = {}
                
                # Load attributes
                for attr_name in group.attrs.keys():
                    value = group.attrs[attr_name]
                    # Try to parse JSON lists
                    if isinstance(value, str) and (value.startswith('[') or value.startswith('{')):
                        try:
                            value = json.loads(value)
                        except json.JSONDecodeError:
                            pass  # Keep as string
                    group_data[attr_name] = value
                
                metadata[group_name] = group_data
        
        return metadata
    
    def _report_storage_info(self, output_path, raw_results):
        """Report storage efficiency information."""
        file_size_mb = os.path.getsize(output_path) / 1024**2
        raw_size_mb = raw_results.nbytes / 1024**2
        compression_ratio = raw_size_mb / file_size_mb if file_size_mb > 0 else 1.0
        
        print(f"  Storage efficiency:")
        print(f"    File size: {file_size_mb:.1f} MB")
        print(f"    Raw data size: {raw_size_mb:.1f} MB")
        print(f"    Compression ratio: {compression_ratio:.1f}x")
        print(f"    Space saved: {100*(1 - 1/compression_ratio):.1f}%")
