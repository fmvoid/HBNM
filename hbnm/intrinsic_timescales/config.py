# This script exists only for config loading / management

def load_config(experiment_name):
    """Load configuration for a specific experiment."""
    
    if experiment_name == "nmda_avg":
        from .configs.nmda_avg import NMDA_CONFIG
        return NMDA_CONFIG
    # elif experiment_name == "dopamine_d1":
    #     from .configs.dopamine_d1 import DOPAMINE_D1_CONFIG
    #     return DOPAMINE_D1_CONFIG
    else:
        raise ValueError(f"Unknown experiment: {experiment_name}")