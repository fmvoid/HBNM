# hbnm/intrinsic_timescales/patterns.py

"""
Stimulation Pattern Generation Module

This module provides functions for creating various brain stimulation patterns
used in intrinsic timescales experiments and other neuroscience simulations.

Current patterns:
- Simple rectangular pulse
- (Future) Repetitive TMS patterns (1Hz, 5Hz, etc.)
- (Future) Theta burst stimulation
- (Future) Continuous burst stimulation

The module also provides utilities for validating stimulation arrays and 
computing stimulation statistics.
"""

import numpy as np
import warnings


def create_time_points(total_time, dt, n_save):
    """
    Create time points array for simulation.
    
    Parameters:
    -----------
    total_time : float
        Total simulation time in seconds
    dt : float
        Integration time step in seconds
    n_save : int
        Save every nth time step
        
    Returns:
    --------
    np.ndarray
        Array of time points in seconds
    """
    return np.arange(0, total_time + dt, dt * n_save)


def create_simple_pulse(total_time, dt, start_time, duration, amplitude, target_region, n_regions):
    """
    Create a simple rectangular pulse stimulation pattern.
    
    Parameters:
    -----------
    total_time : float
        Total simulation time in seconds
    dt : float
        Integration time step in seconds
    start_time : float
        Pulse start time in seconds
    duration : float
        Pulse duration in seconds
    amplitude : float
        Pulse amplitude in nA
    target_region : int
        Index of target brain region
    n_regions : int
        Total number of brain regions
        
    Returns:
    --------
    np.ndarray
        Stimulation array with shape (n_time_steps, n_regions)
    """
    # Compute number of time steps
    n_steps = int(total_time / dt) + 1
    
    # Initialize stimulation array
    stim_array = np.zeros((n_steps, n_regions))
    
    # Validate inputs
    if target_region >= n_regions:
        raise ValueError(f"target_region {target_region} >= n_regions {n_regions}")
    
    if start_time + duration > total_time:
        warnings.warn(f"Stimulation extends beyond total_time: {start_time + duration} > {total_time}")
    
    # Apply stimulation
    i_start = int(start_time / dt)
    i_end = i_start + int(duration / dt)
    
    # Ensure indices are within bounds
    i_start = max(0, i_start)
    i_end = min(n_steps, i_end)
    
    if i_end > i_start:
        stim_array[i_start:i_end, target_region] = amplitude
    else:
        warnings.warn("Stimulation duration too short for given time step")
    
    return stim_array


def validate_stimulation_array(stim_array, n_regions, total_time, dt):
    """
    Validate stimulation array dimensions and content.
    
    Parameters:
    -----------
    stim_array : np.ndarray
        Stimulation array to validate
    n_regions : int
        Expected number of brain regions
    total_time : float
        Expected total simulation time
    dt : float
        Integration time step
        
    Returns:
    --------
    bool
        True if valid
        
    Raises:
    -------
    ValueError
        If validation fails
    """
    expected_time_steps = int(total_time / dt) + 1
    expected_shape = (expected_time_steps, n_regions)
    
    if stim_array.shape != expected_shape:
        raise ValueError(f"Stimulation array shape {stim_array.shape} != expected {expected_shape}")
    
    if np.any(np.isnan(stim_array)) or np.any(np.isinf(stim_array)):
        raise ValueError("Stimulation array contains NaN or Inf values")
    
    # Check for reasonable amplitude values (warn if outside typical range)
    max_amplitude = np.max(np.abs(stim_array))
    if max_amplitude > 10.0:
        warnings.warn(f"Large stimulation amplitude detected: {max_amplitude} nA")
    
    return True


def compute_stimulation_statistics(stim_array, dt):
    """
    Compute basic statistics about stimulation pattern.
    
    Parameters:
    -----------
    stim_array : np.ndarray
        Stimulation array with shape (n_time_steps, n_regions)
    dt : float
        Time step in seconds
        
    Returns:
    --------
    dict
        Dictionary containing stimulation statistics
    """
    # Find stimulated regions
    stimulated_regions = np.any(stim_array != 0, axis=0)
    n_stimulated = np.sum(stimulated_regions)
    
    # Compute total stimulation duration per region
    stim_durations = []
    for region in range(stim_array.shape[1]):
        if stimulated_regions[region]:
            non_zero_steps = np.sum(stim_array[:, region] != 0)
            duration = non_zero_steps * dt
            stim_durations.append(duration)
    
    # Find amplitude statistics
    non_zero_values = stim_array[stim_array != 0]
    
    stats = {
        'n_stimulated_regions': n_stimulated,
        'stimulated_region_indices': np.where(stimulated_regions)[0].tolist(),
        'total_stimulation_time': np.sum(stim_durations) if stim_durations else 0.0,
        'mean_amplitude': np.mean(np.abs(non_zero_values)) if len(non_zero_values) > 0 else 0.0,
        'max_amplitude': np.max(np.abs(non_zero_values)) if len(non_zero_values) > 0 else 0.0,
        'stimulation_durations': stim_durations
    }
    
    return stats


def print_stimulation_summary(stim_array, dt, config=None):
    """
    Print a human-readable summary of stimulation pattern.
    
    Parameters:
    -----------
    stim_array : np.ndarray
        Stimulation array
    dt : float
        Time step in seconds
    config : dict, optional
        Stimulation configuration for additional context
    """
    stats = compute_stimulation_statistics(stim_array, dt)
    
    print("=== STIMULATION SUMMARY ===")
    print(f"Array shape: {stim_array.shape}")
    print(f"Stimulated regions: {stats['n_stimulated_regions']}")
    
    if stats['n_stimulated_regions'] > 0:
        print(f"Region indices: {stats['stimulated_region_indices']}")
        print(f"Max amplitude: {stats['max_amplitude']:.3f} nA")
        print(f"Mean amplitude: {stats['mean_amplitude']:.3f} nA")
        
        for i, (region_idx, duration) in enumerate(zip(stats['stimulated_region_indices'], 
                                                      stats['stimulation_durations'])):
            print(f"Region {region_idx}: {duration:.3f}s stimulation")
    
    if config:
        total_time = config.get('total_time', 'unknown')
        sampling_rate = 1 / (dt * config.get('n_save', 1))
        print(f"Total time: {total_time}s | Sampling: {sampling_rate:.0f}Hz")


# FUTURE: TMS Pattern Generation Functions
# These are placeholders for future implementation

def create_repetitive_tms_pattern(frequency, n_pulses, pulse_duration, amplitude, 
                                 target_region, n_regions, total_time, dt):
    """
    Future function for creating repetitive TMS patterns (1Hz, 5Hz, etc.).
    
    This is a placeholder for future implementation of complex TMS protocols.
    """
    raise NotImplementedError("TMS patterns not yet implemented")


def create_theta_burst_pattern(burst_frequency, intra_burst_frequency, n_bursts, 
                             pulse_duration, amplitude, target_region, n_regions, 
                             total_time, dt):
    """
    Future function for creating theta burst stimulation patterns.
    
    Theta burst typically consists of bursts of 3 pulses at 50Hz,
    with bursts repeated at 5Hz (theta frequency).
    """
    raise NotImplementedError("Theta burst patterns not yet implemented")


def create_continuous_burst_pattern(frequency, burst_duration, amplitude, 
                                   target_region, n_regions, total_time, dt):
    """
    Future function for creating continuous burst stimulation.
    
    High-frequency continuous stimulation for specified duration.
    """
    raise NotImplementedError("Continuous burst patterns not yet implemented")
