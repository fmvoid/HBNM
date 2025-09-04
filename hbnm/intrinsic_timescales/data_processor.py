# hbnm/intrinsic_timescales/data_processor.py

"""
Data Processing Module for Intrinsic Timescales Analysis

This module handles post-simulation data processing to prepare simulation results
for curve fitting. It extracts decay periods, validates data, and provides
minimal, clean datasets for exponential curve fitting.
"""

import numpy as np
import warnings
from . import patterns


class DataProcessor:
    """
    Processes simulation results for intrinsic timescales analysis.
    
    Handles extraction of post-stimulation decay periods from raw simulation
    results, with validation and preparation for exponential curve fitting.
    """
    
    def __init__(self, config):
        """Initialize with configuration dictionary."""
        self.config = config
        self.stim_config = config['stimulation']
        
        # Validate configuration
        self._validate_config()
    
    def extract_decay_period(self, simulation_results):
        """
        Extract post-stimulation decay period from simulation results.
        
        Parameters:
        -----------
        simulation_results : np.ndarray
            Raw simulation results with shape (n_trials, n_regions, n_timepoints)
            
        Returns:
        --------
        dict
            Clean, minimal dataset ready for curve fitting:
            {
                'time_relative': np.ndarray,     # Time points (t=0 at stim end)
                'firing_rates': np.ndarray       # Mean firing rates (n_regions, n_timepoints)
            }
        """
        print("=== PROCESSING SIMULATION DATA ===")
        
        # 1. Validate input data
        self._validate_simulation_results(simulation_results)
        n_trials, n_regions, n_timepoints = simulation_results.shape
        
        # 2. Create time vectors
        time_points = patterns.create_time_points(
            total_time=self.stim_config['total_time'],
            dt=self.stim_config['dt'], 
            n_save=self.stim_config['n_save']
        )
        
        if len(time_points) != n_timepoints:
            raise ValueError(f"Time points mismatch: expected {len(time_points)}, got {n_timepoints}")
        
        # 3. Extract decay period indices
        post_stim_start = self.stim_config['stim_start_time'] + self.stim_config['stim_duration']
        post_stim_end = self.stim_config['total_time']
        
        # Find indices for decay period
        decay_start_idx = np.searchsorted(time_points, post_stim_start)
        decay_end_idx = len(time_points)  # Go to the end
        
        print(f"Step 1: Extracting decay period...")
        print(f"  Post-stimulation period: {post_stim_start:.1f}s to {post_stim_end:.1f}s")
        print(f"  Time indices: {decay_start_idx} to {decay_end_idx-1}")
        
        # 4. Extract decay data
        decay_data = simulation_results[:, :, decay_start_idx:decay_end_idx]
        decay_time_points = time_points[decay_start_idx:decay_end_idx]
        
        # Create relative time (t=0 at stimulation end)
        time_relative = decay_time_points - post_stim_start
        
        print(f"  Decay data shape: {decay_data.shape}")
        print(f"  Time range: {time_relative[0]:.3f}s to {time_relative[-1]:.3f}s")
        
        # 5. Compute mean across trials
        print("Step 2: Computing trial averages...")
        mean_firing_rates = np.mean(decay_data, axis=0)  # Shape: (n_regions, n_decay_timepoints)
        
        print(f"  Mean firing rates shape: {mean_firing_rates.shape}")
        print(f"  Averaged across {n_trials} trials")
        
        # 6. Validate processed data
        print("Step 3: Validating processed data...")
        self._validate_firing_rates(mean_firing_rates)
        
        # 7. Create clean output structure (no metadata bloat)
        processed_data = {
            'time_relative': time_relative,
            'firing_rates': mean_firing_rates
        }
        
        print(f"✓ Data processing completed!")
        print(f"  Ready for curve fitting: {n_regions} regions × {len(time_relative)} time points")
        print(f"  Data covers {time_relative[-1]:.1f}s decay period")
        
        return processed_data
    
    def compute_summary_statistics(self, processed_data):
        """
        Compute basic statistics for validation and reporting.
        
        Parameters:
        -----------
        processed_data : dict
            Output from extract_decay_period()
            
        Returns:
        --------
        dict
            Summary statistics
        """
        firing_rates = processed_data['firing_rates']
        n_regions, n_timepoints = firing_rates.shape
        
        stats = {
            'n_regions': n_regions,
            'n_timepoints': n_timepoints,
            'mean_rate_overall': np.mean(firing_rates),
            'std_rate_overall': np.std(firing_rates),
            'min_rate_per_region': np.min(firing_rates, axis=1),
            'max_rate_per_region': np.max(firing_rates, axis=1),
            'mean_rate_per_region': np.mean(firing_rates, axis=1)
        }
        
        return stats
    
    def validate_for_fitting(self, processed_data):
        """
        Validate that processed data is ready for exponential curve fitting.
        
        Parameters:
        -----------
        processed_data : dict
            Output from extract_decay_period()
            
        Returns:
        --------
        bool
            True if data is ready for fitting
            
        Raises:
        -------
        ValueError
            If data fails validation
        """
        time_relative = processed_data['time_relative']
        firing_rates = processed_data['firing_rates']
        
        # Check time vector
        if not np.all(np.diff(time_relative) > 0):
            raise ValueError("Time vector is not monotonically increasing")
        
        if time_relative[0] != 0.0:
            raise ValueError(f"Time should start at 0, got {time_relative[0]}")
        
        # Check firing rates
        if np.any(np.isnan(firing_rates)) or np.any(np.isinf(firing_rates)):
            raise ValueError("Firing rates contain NaN or Inf values")
        
        # Check for reasonable decay behavior
        n_regions = firing_rates.shape[0]
        problematic_regions = []
        
        for region in range(n_regions):
            initial_rate = firing_rates[region, 0]
            final_rate = firing_rates[region, -1]
            
            # Very basic check: firing rates should be positive and reasonable
            if initial_rate <= 0 or final_rate <= 0:
                problematic_regions.append(region)
            elif initial_rate > 1000 or final_rate > 1000:  # Unreasonably high
                problematic_regions.append(region)
        
        if problematic_regions:
            warnings.warn(f"Problematic firing rates in regions: {problematic_regions[:5]}...")
        
        print("✓ Data validation passed - ready for exponential curve fitting")
        return True
    
    # PRIVATE METHODS
    
    def _validate_config(self):
        """Validate that configuration contains required parameters."""
        required_keys = ['total_time', 'dt', 'n_save', 'stim_start_time', 
                        'stim_duration', 'stim_region_idx']
        
        for key in required_keys:
            if key not in self.stim_config:
                raise ValueError(f"Missing required stimulation config: '{key}'")
    
    def _validate_simulation_results(self, results):
        """Validate simulation results format and content."""
        if not isinstance(results, np.ndarray):
            raise ValueError("Simulation results must be numpy array")
        
        if results.ndim != 3:
            raise ValueError(f"Expected 3D array (trials, regions, timepoints), got {results.ndim}D")
        
        if np.any(np.isnan(results)) or np.any(np.isinf(results)):
            raise ValueError("Simulation results contain NaN or Inf values")
        
        n_trials, n_regions, n_timepoints = results.shape
        print(f"  Input data: {n_trials} trials × {n_regions} regions × {n_timepoints} timepoints")
    
    def _validate_firing_rates(self, firing_rates):
        """Validate processed firing rates."""
        mean_rate = np.mean(firing_rates)
        std_rate = np.std(firing_rates)
        
        print(f"  Firing rate validation:")
        print(f"    Mean: {mean_rate:.3f} Hz")
        print(f"    Std:  {std_rate:.3f} Hz")
        print(f"    Range: {np.min(firing_rates):.3f} - {np.max(firing_rates):.3f} Hz")
        
        # Check for reasonable firing rates
        if mean_rate < 0.1:
            warnings.warn(f"Very low mean firing rate: {mean_rate:.3f} Hz")
        elif mean_rate > 100:
            warnings.warn(f"Very high mean firing rate: {mean_rate:.3f} Hz")
        
        # Check for negative rates
        negative_count = np.sum(firing_rates < 0)
        if negative_count > 0:
            warnings.warn(f"Found {negative_count} negative firing rate values")
        
        print(f"    ✓ Firing rates validated")
