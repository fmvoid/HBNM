"""
Spatial autocorrelation analysis for brain maps.

This module provides functions to compute and visualize spatial autocorrelation
for brain connectivity data using distance matrices and surrogate null models.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


def compute_variogram(distance_matrix, surrogate_maps, empirical_map, bin_width=5, 
                           figsize=(10, 6), map_name=None, return_data=False):
    """
    Compute and plot spatial autocorrelation analysis comparing empirical data to surrogate maps.
    Plots correlation vs distance, truncated at the point where empirical correlation crosses zero.
    
    Parameters
    ----------
    distance_matrix : numpy.ndarray
        Square matrix of distances between ROIs (shape: n_rois x n_rois)
    surrogate_maps : numpy.ndarray
        Array of surrogate maps (shape: n_surrogates x n_rois)
    empirical_map : numpy.ndarray
        Empirical map values (shape: n_rois)
    bin_width : float, optional
        Width of distance bins in same units as distance_matrix (default: 5)
    figsize : tuple, optional
        Figure size for the plot (default: (10, 6))
    map_name : str, optional
        Name of the map to be used in the plot title (default: None)
    return_data : bool, optional
        If True, return correlation data along with plot (default: False)
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated plot figure
    data : dict, optional
        If return_data=True, returns dictionary with:
        - 'bin_centers': distance bin centers (truncated)
        - 'empirical_correlations': empirical correlation values (truncated)
        - 'surrogate_correlations': list of surrogate correlation arrays (truncated)
        - 'zero_crossing_distance': distance where empirical correlation crosses zero
    """
    
    # Get number of ROIs
    n_rois = len(empirical_map)
    
    # Extract distances and values for all unique ROI pairs
    distances = []
    empirical_values_i = []
    empirical_values_j = []
    
    for i in range(n_rois):
        for j in range(i + 1, n_rois):  # Only unique pairs
            distances.append(distance_matrix[i, j])
            empirical_values_i.append(empirical_map[i])
            empirical_values_j.append(empirical_map[j])
    
    # Convert to arrays
    distances = np.array(distances)
    empirical_values_i = np.array(empirical_values_i)
    empirical_values_j = np.array(empirical_values_j)
    
    # Create distance bins
    max_distance = distances.max()
    bin_edges = np.arange(0, max_distance + bin_width, bin_width)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Compute empirical autocorrelation
    empirical_correlations = []
    valid_bin_centers = []
    
    for i in range(len(bin_edges) - 1):
        # Find pairs in this distance bin
        in_bin = (distances >= bin_edges[i]) & (distances < bin_edges[i + 1])
        
        if np.sum(in_bin) < 2:  # Need at least 2 pairs
            continue
            
        # Get values for this bin
        xi_bin = empirical_values_i[in_bin]
        xj_bin = empirical_values_j[in_bin]
        
        # Compute correlation if possible
        if len(xi_bin) > 1 and np.std(xi_bin) > 0 and np.std(xj_bin) > 0:
            corr, _ = pearsonr(xi_bin, xj_bin)
            empirical_correlations.append(corr)
            valid_bin_centers.append(bin_centers[i])
    
    empirical_correlations = np.array(empirical_correlations)
    valid_bin_centers = np.array(valid_bin_centers)
    
    # Find where empirical correlation crosses zero
    zero_crossing_idx = None
    zero_crossing_distance = None
    
    for i, corr in enumerate(empirical_correlations):
        if corr <= 0:
            zero_crossing_idx = i
            zero_crossing_distance = valid_bin_centers[i]
            print(f"Empirical correlation crosses zero at distance: {zero_crossing_distance:.2f}")
            break
    
    # If no zero crossing found, use all data
    if zero_crossing_idx is None:
        zero_crossing_idx = len(empirical_correlations)
        zero_crossing_distance = valid_bin_centers[-1] if len(valid_bin_centers) > 0 else None
        print("Empirical correlation does not cross zero within distance range")
    
    # Truncate empirical data at zero crossing
    empirical_correlations_truncated = empirical_correlations[:zero_crossing_idx]
    valid_bin_centers_truncated = valid_bin_centers[:zero_crossing_idx]
    
    # Compute surrogate autocorrelations
    all_surrogate_correlations = []
    
    for surrogate_idx in range(surrogate_maps.shape[0]):
        surrogate_map = surrogate_maps[surrogate_idx]
        
        # Extract values for all unique pairs for this surrogate
        surrogate_values_i = []
        surrogate_values_j = []
        
        for i in range(n_rois):
            for j in range(i + 1, n_rois):
                surrogate_values_i.append(surrogate_map[i])
                surrogate_values_j.append(surrogate_map[j])
        
        surrogate_values_i = np.array(surrogate_values_i)
        surrogate_values_j = np.array(surrogate_values_j)
        
        # Compute correlations for each bin
        surrogate_correlations = []
        
        for i in range(len(bin_edges) - 1):
            in_bin = (distances >= bin_edges[i]) & (distances < bin_edges[i + 1])
            
            if np.sum(in_bin) < 2:
                continue
                
            xi_bin = surrogate_values_i[in_bin]
            xj_bin = surrogate_values_j[in_bin]
            
            if len(xi_bin) > 1 and np.std(xi_bin) > 0 and np.std(xj_bin) > 0:
                corr, _ = pearsonr(xi_bin, xj_bin)
                surrogate_correlations.append(corr)
        
        # Truncate surrogate data at the same point as empirical
        surrogate_correlations = np.array(surrogate_correlations)
        surrogate_correlations_truncated = surrogate_correlations[:zero_crossing_idx]
        all_surrogate_correlations.append(surrogate_correlations_truncated)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot all surrogates in gray (truncated)
    for surrogate_corr in all_surrogate_correlations:
        # Handle potential length mismatch
        n_points = min(len(valid_bin_centers_truncated), len(surrogate_corr))
        ax.plot(valid_bin_centers_truncated[:n_points], surrogate_corr[:n_points], 
                color='gray', alpha=0.3, linewidth=0.8)
    
    # Plot empirical in black (truncated)
    ax.plot(valid_bin_centers_truncated, empirical_correlations_truncated, 'k-', linewidth=2, 
            label='Empirical', zorder=10)
    
    # Add horizontal line at y=0
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
    
    # Formatting
    ax.set_xlabel('Distance')
    ax.set_ylabel('Pearson Correlation')
    title = f'Spatial Autocorrelation'
    if map_name is not None:
        title += f' - {map_name}'
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    
    if return_data:
        data = {
            'bin_centers': valid_bin_centers_truncated,
            'empirical_correlations': empirical_correlations_truncated,
            'surrogate_correlations': all_surrogate_correlations,
            'zero_crossing_distance': zero_crossing_distance
        }
        return fig, data
    else:
        return fig
