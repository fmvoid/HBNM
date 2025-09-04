# hbnm/intrinsic_timescales/simulation.py

import numpy as np
import multiprocessing as mp
from multiprocessing import Pool
import time
from copy import deepcopy
import warnings

from . import patterns


class StimulationSimulator:
    """
    Handles parallel simulation of brain network models with various stimulation protocols.
    Designed for extensibility to support complex TMS patterns in the future.
    """
    
    def __init__(self, config):
        """Initialize with configuration dictionary."""
        self.config = config
        
        # Extract configuration sections for convenience
        self.stim_config = config['stimulation']
        self.parallel_config = config['parallel']
        self.exp_config = config['experiment']
        
        # Will be populated during simulation
        self.results = None
        self.simulation_metadata = {}
        
        # Validate configuration
        self._validate_config()
    
    def run_stimulation_experiment(self, model_data):
        """
        Main orchestrator for running parallel stimulation experiments.
        
        Parameters:
        -----------
        model_data : dict
            Output from ModelLoader.load_model() containing:
            - 'sc': structural connectivity
            - 'model_params': formatted parameters
            - 'model_type': 'homogeneous' or 'heterogeneous'
            - 'maps': heterogeneity maps (if applicable)
            - 'map_invert_flags': map inversion flags
            - 'n_regions': number of brain regions
            
        Returns:
        --------
        np.ndarray
            Simulation results with shape (n_trials, n_regions, n_timepoints).
            All metadata is available in the original config.
        """
        print("=== STARTING STIMULATION EXPERIMENT ===")
        
        # 1. Validate setup
        self._validate_setup(model_data)
        
        # 2. Create stimulation array
        print("Step 1: Creating stimulation protocol...")
        stim_array = self.create_stimulation_array(model_data['n_regions'])
        
        # 3. Setup simulation parameters
        print("Step 2: Preparing parallel simulation...")
        sim_params = self._build_simulation_params(model_data, stim_array)
        n_trials = self.exp_config['n_trials']
        
        # 4. Setup multiprocessing
        n_cores = self._setup_multiprocessing()
        
        # 5. Run parallel simulations
        print(f"Step 3: Running {n_trials} trials on {n_cores} cores...")
        results = self._run_parallel_simulations(sim_params, n_trials, n_cores)
        
        # 6. Validate and process results
        print("Step 4: Validating results...")
        processed_results = self._process_results(results, model_data)
        
        # 7. Store results for potential later access
        self.results = processed_results
        
        print("✓ Stimulation experiment completed successfully!")
        print(f"  Returning results array: {processed_results.shape}")
        
        # Return just the results array - config already contains all metadata
        return processed_results
    
    def create_stimulation_array(self, n_regions):
        """
        Create stimulation array based on configuration.
        Uses the patterns module for actual pattern generation.
        Future: Extend this for complex TMS patterns (1Hz, 5Hz, Theta Burst).
        
        Parameters:
        -----------
        n_regions : int
            Number of brain regions
            
        Returns:
        --------
        np.ndarray
            Stimulation array with shape (n_time_steps, n_regions)
        """
        # Extract parameters from config
        total_time = self.stim_config['total_time']
        dt = self.stim_config['dt']
        start_time = self.stim_config['stim_start_time']
        duration = self.stim_config['stim_duration'] 
        amplitude = self.stim_config['stim_amplitude']
        target_region = self.stim_config['stim_region_idx']
        
        # Use patterns module to create the stimulation array
        stim_array = patterns.create_simple_pulse(
            total_time=total_time,
            dt=dt,
            start_time=start_time,
            duration=duration,
            amplitude=amplitude,
            target_region=target_region,
            n_regions=n_regions
        )
        
        # Log stimulation details (experiment-specific logic stays in simulator)
        sampling_rate = 1 / (dt * self.stim_config['n_save'])
        
        print(f"  Protocol: Single pulse")
        print(f"  Target: Region {target_region} | Amplitude: {amplitude}nA")
        print(f"  Duration: {duration}s | Total time: {total_time}s")
        print(f"  Sampling: {sampling_rate:.0f}Hz | Array shape: {stim_array.shape}")
        
        return stim_array
    
    def validate_setup(self, model_data):
        """
        Validate that everything is ready for simulation.
        Public method for users to check setup before expensive computation.
        """
        try:
            self._validate_setup(model_data)
            print("✓ Setup validation passed - ready for simulation!")
            return True
        except Exception as e:
            print(f"❌ Setup validation failed: {e}")
            return False
    
    # PRIVATE METHODS (internal helpers)
    
    def _validate_config(self):
        """Validate configuration completeness and consistency."""
        required_sections = ['stimulation', 'parallel', 'experiment']
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required config section: '{section}'")
        
        # Validate stimulation config
        required_stim_keys = ['total_time', 'dt', 'n_save', 'stim_start_time', 
                             'stim_duration', 'stim_region_idx', 'stim_amplitude']
        for key in required_stim_keys:
            if key not in self.stim_config:
                raise ValueError(f"Missing required stimulation parameter: '{key}'")
        
        # Validate parallel config
        required_parallel_keys = ['max_cores']
        for key in required_parallel_keys:
            if key not in self.parallel_config:
                raise ValueError(f"Missing required parallel parameter: '{key}'")
        
        # Validate experiment config
        if 'n_trials' not in self.exp_config:
            raise ValueError("Missing required experiment parameter: 'n_trials'")
    
    def _validate_setup(self, model_data):
        """Validate model_data and configuration compatibility."""
        required_keys = ['sc', 'model_params', 'model_type', 'n_regions']
        for key in required_keys:
            if key not in model_data:
                raise ValueError(f"Missing required model_data key: '{key}'")
        
        # Validate stimulation target
        target_region = self.stim_config['stim_region_idx']
        n_regions = model_data['n_regions']
        if target_region >= n_regions:
            raise ValueError(f"Invalid stimulation target {target_region} for {n_regions} regions")
        
        # Validate model type specific requirements
        if model_data['model_type'] == 'heterogeneous':
            if 'maps' not in model_data or model_data['maps'] is None:
                raise ValueError("Heterogeneous model requires maps in model_data")
            if 'map_invert_flags' not in model_data:
                raise ValueError("Heterogeneous model requires map_invert_flags in model_data")
    
    def _build_simulation_params(self, model_data, stim_array):
        """Build parameter tuple for parallel simulation workers."""
        return (
            model_data['sc'],
            model_data['model_params'], 
            model_data['model_type'],
            self.stim_config['total_time'],
            self.stim_config['dt'],
            self.stim_config['n_save'],
            stim_array,
            model_data.get('maps'),
            model_data.get('map_invert_flags')
        )
    
    def _setup_multiprocessing(self):
        """Configure multiprocessing based on system and configuration."""
        max_cores = self.parallel_config['max_cores']
        available_cores = mp.cpu_count()
        
        # Use all but 1 core, limited by max_cores setting
        n_cores = min(max_cores, available_cores - 1)
        n_cores = max(1, n_cores)  # Ensure at least 1 core
        
        print(f"  Available cores: {available_cores}")
        print(f"  Using cores: {n_cores} (limited by config: {max_cores})")
        
        return n_cores
    
    def _run_parallel_simulations(self, sim_params, n_trials, n_cores):
        """Execute parallel simulations with progress reporting."""
        # Prepare trial arguments
        trial_args = [(trial, sim_params) for trial in range(n_trials)]
        
        # Get time points for result array sizing
        time_points = self._compute_time_points()
        n_regions = sim_params[0].shape[0]  # from SC matrix
        
        # Initialize results storage
        all_results = np.zeros((n_trials, n_regions, len(time_points)))
        
        # Run simulations with progress reporting
        start_time = time.time()
        progress_freq = self.parallel_config.get('progress_freq', 50)
        
        with Pool(n_cores) as pool:
            for i, result in enumerate(pool.imap(self._run_single_trial, trial_args)):
                all_results[i] = result
                
                # Progress updates
                if (i + 1) % progress_freq == 0:
                    self._report_progress(i + 1, n_trials, start_time)
        
        # Final timing report
        total_time = time.time() - start_time
        print(f"\n✓ Parallel simulation completed!")
        print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        print(f"  Average per trial: {total_time/n_trials:.3f}s")
        print(f"  Effective rate: {n_trials/total_time:.1f} trials/second")
        
        return all_results
    
    def _run_single_trial(self, args):
        """
        Run a single simulation trial.
        This function must be picklable for multiprocessing.
        """
        trial_num, sim_params = args
        
        # Import inside function for multiprocessing compatibility
        from hbnm.bnm import Bnm
        
        # Unpack simulation parameters
        (sc, model_params, model_type, total_time, dt, n_save, 
         stim_array, model_maps, map_invert_flags) = sim_params
        
        # Create model based on type
        if model_type == "homogeneous":
            trial_model = Bnm(sc)
            trial_model.set('w_EI', model_params['w_EI'])
            trial_model.set('w_EE', model_params['w_EE'])
            trial_model.set('G', model_params['G'])
            
        elif model_type == "heterogeneous":
            trial_model = Bnm(sc, maps=model_maps, map_invert_flags=map_invert_flags)
            trial_model.set('w_EI', model_params['w_EI'])  # (baseline, scaling) tuple
            trial_model.set('w_EE', model_params['w_EE'])  # (baseline, scaling) tuple
            trial_model.set('G', model_params['G'])
            
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Run simulation with stimulation config parameters
        # Get these from the config that was passed in sim_params
        # For now, we'll use the hardcoded values but future versions should pass these
        trial_model.dmf.integrate(
            t=total_time,
            dt=dt,
            n_save=n_save,
            stimulation=stim_array,
            delays=False,      # Future: get from config
            include_BOLD=True, # Future: get from config  
            from_fixed=True,   # Future: get from config
            sim_seed=trial_num
        )
        
        # Extract and return firing rate time series
        r_E_trial = trial_model.dmf.sim.time_series('r_E')
        return r_E_trial
    
    def _compute_time_points(self):
        """Compute time points array for simulation."""
        return np.arange(0, self.stim_config['total_time'] + self.stim_config['dt'], 
                        self.stim_config['dt'] * self.stim_config['n_save'])
    
    def _report_progress(self, completed, total, start_time):
        """Report simulation progress."""
        elapsed = time.time() - start_time
        rate = completed / elapsed
        eta = (total - completed) / rate
        print(f"  Completed {completed}/{total} trials - "
              f"Rate: {rate:.1f} trials/s - ETA: {eta:.1f}s")
    
    def _process_results(self, results, model_data):
        """Process and validate simulation results."""
        # Basic validation
        expected_shape = (self.exp_config['n_trials'], 
                         model_data['n_regions'], 
                         len(self._compute_time_points()))
        
        if results.shape != expected_shape:
            raise ValueError(f"Results shape {results.shape} doesn't match expected {expected_shape}")
        
        # Sanity checks
        mean_rate = results.mean()
        std_rate = results.std()
        
        print(f"  Final shape: {results.shape}")
        print(f"  Sanity check - Mean rate: {mean_rate:.3f}Hz, Std: {std_rate:.3f}Hz")
        
        # Check for reasonable firing rates
        if mean_rate < 0.1 or mean_rate > 100:
            warnings.warn(f"Unusual mean firing rate: {mean_rate:.3f}Hz")
        
        # Check for NaN/Inf values
        if np.any(np.isnan(results)) or np.any(np.isinf(results)):
            raise ValueError("Results contain NaN or Inf values")
        
        return results
    
    def get_supported_patterns(self):
        """Return list of currently supported stimulation patterns."""
        return ['single_pulse']  # Future: add TMS patterns from patterns module
