from .base import BASE_CONFIG
import copy

# Start with base config and modify for NMDA experiment
NMDA_CONFIG = copy.deepcopy(BASE_CONFIG)

# Add NMDA-specific settings
NMDA_CONFIG.update({
    'model': {
        'folder': 'NMDA_AVG',
        'use_latest': True,
        'fallback_iteration': 20,
        'map_path': '/home/frank/HBNM/data/heterogeneity_vectors/linearized/NMDA_linearized.npy',
        'invert_flags': [False],
        'sc_path': '/home/frank/HBNM/data/SC_Left_Hemi.npy'
    },
    'stimulation': {
        **NMDA_CONFIG['stimulation'],  # Keep base settings
        'stim_amplitude': 0.1          # NMDA-specific amplitude
    },
    'experiment': {
        'name': 'NMDA_AVG_intrinsic_timescales',
        'n_trials': 100
    }
})