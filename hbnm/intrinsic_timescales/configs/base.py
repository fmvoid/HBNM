# Default configuration that all experiments inherit
BASE_CONFIG = {
    'paths': {
        'input_dir': "/home/frank/HBNM/data/",
        'output_dir': "/home/frank/HBNM/outputs/"
    },
    'stimulation': {
        'total_time': 7.0,
        'dt': 0.0001,
        'n_save': 10,
        'stim_start_time': 3.0,
        'stim_duration': 1.0,
        'stim_region_idx': 0,
        'delays': False,
        'include_BOLD': True,
        'from_fixed': True
    },
    'parallel': {
        'max_cores': 10,
        'progress_freq': 50
    },
    'curve_fitting': {
        'parameter_bounds': {
            'A': [-10.0, 10.0],      # Amplitude bounds (Hz)
            'D': [0.001, 20.0],      # Decay rate bounds (1/s)  
            'B': [-5.0, 15.0]        # Baseline bounds (Hz)
        },
        'max_iterations': 2000,      # Maximum function evaluations
        'method': 'trf'              # Trust Region Reflective algorithm
    }
}