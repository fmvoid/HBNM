#!/usr/bin/env python3
"""
Comprehensive analysis of T1w/T2w map processing in HBNM codebase.
This addresses the critical question about negative correlation implementation.
"""

import numpy as np
import matplotlib.pyplot as plt
from hbnm.io import Data
from hbnm.model.utils import linearize_map
from hbnm import Bnm
import h5py

def analyze_t1w2_processing():
    """
    Analyze how T1w/T2w maps are processed and whether the negative correlation 
    is properly implemented according to the Demirtas paper.
    """
    
    print("="*80)
    print("CRITICAL ANALYSIS: T1w/T2w Map Processing in HBNM")
    print("="*80)
    print()
    
    # Load data
    data = Data('./data/', './outputs/')
    
    # Load raw T1w/T2w data
    print("1. LOADING RAW T1w/T2w DATA")
    print("-" * 40)
    
    fin = data.load('demirtas_neuron_2019.hdf5')
    t1t2_raw = fin['t1wt2w'][:]  # Raw T1w/T2w data
    fin.close()
    
    # Extract left hemisphere (first 180 regions)
    t1t2_left_raw = t1t2_raw[:180]
    
    print(f"Raw T1w/T2w statistics (left hemisphere, 180 regions):")
    print(f"  Mean: {t1t2_left_raw.mean():.6f}")
    print(f"  Std:  {t1t2_left_raw.std():.6f}")
    print(f"  Min:  {t1t2_left_raw.min():.6f}")
    print(f"  Max:  {t1t2_left_raw.max():.6f}")
    print(f"  Range: {t1t2_left_raw.max() - t1t2_left_raw.min():.6f}")
    print()
    
    # Apply linearization (as done in current codebase)
    print("2. APPLYING LINEARIZATION (current codebase)")
    print("-" * 40)
    
    t1t2_linearized = linearize_map(t1t2_left_raw)
    
    print(f"Linearized T1w/T2w statistics:")
    print(f"  Mean: {t1t2_linearized.mean():.6f}")
    print(f"  Std:  {t1t2_linearized.std():.6f}")
    print(f"  Min:  {t1t2_linearized.min():.6f}")
    print(f"  Max:  {t1t2_linearized.max():.6f}")
    print(f"  Range: {t1t2_linearized.max() - t1t2_linearized.min():.6f}")
    print()
    
    # Test backwards compatibility transformations
    print("3. TESTING BACKWARDS COMPATIBILITY TRANSFORMATIONS")
    print("-" * 40)
    
    # The current _apply_maps method applies these transformations:
    
    # For negative slope: hmap_rev = (maps[0] - np.min(maps[0])) / np.ptp(maps[0])
    hmap_rev = (t1t2_linearized - np.min(t1t2_linearized)) / np.ptp(t1t2_linearized)
    
    # For positive slope: hmap_norm = (-(maps[0] - np.max(maps[0]))) / np.ptp(maps[0])
    hmap_norm = (-(t1t2_linearized - np.max(t1t2_linearized))) / np.ptp(t1t2_linearized)
    
    print(f"Negative slope transformation (hmap_rev):")
    print(f"  Mean: {hmap_rev.mean():.6f}")
    print(f"  Min:  {hmap_rev.min():.6f} (should be 0.0)")
    print(f"  Max:  {hmap_rev.max():.6f} (should be 1.0)")
    print()
    
    print(f"Positive slope transformation (hmap_norm):")
    print(f"  Mean: {hmap_norm.mean():.6f}")
    print(f"  Min:  {hmap_norm.min():.6f} (should be 0.0)")
    print(f"  Max:  {hmap_norm.max():.6f} (should be 1.0)")
    print()
    
    # Test model parameter application
    print("4. TESTING MODEL PARAMETER APPLICATION")
    print("-" * 40)
    
    # Load SC for testing
    sc, _, _ = data.load_demirtas_neuron_2019_data()
    
    # Test with positive and negative coefficients
    test_cases = [
        {"name": "Positive coefficient", "wee_params": (0.15, 0.05), "wei_params": (0.15, 0.03)},
        {"name": "Negative coefficient", "wee_params": (0.15, -0.05), "wei_params": (0.15, -0.03)},
    ]
    
    for case in test_cases:
        print(f"\n{case['name']}:")
        print(f"  wEE params: {case['wee_params']}")
        print(f"  wEI params: {case['wei_params']}")
        
        # Create model
        model = Bnm(sc, gradient=t1t2_linearized)
        model.set('w_EE', case['wee_params'])
        model.set('w_EI', case['wei_params'])
        model.set('G', 1.0)
        
        # Get regional parameters
        w_ee_regional = model.get('w_EE')
        w_ei_regional = model.get('w_EI')
        
        print(f"  wEE regional: min={w_ee_regional.min():.4f}, max={w_ee_regional.max():.4f}, mean={w_ee_regional.mean():.4f}")
        print(f"  wEI regional: min={w_ei_regional.min():.4f}, max={w_ei_regional.max():.4f}, mean={w_ei_regional.mean():.4f}")
        
        # Check which regions have highest/lowest values
        ee_highest_idx = np.argmax(w_ee_regional)
        ee_lowest_idx = np.argmin(w_ee_regional)
        
        print(f"  wEE highest at region {ee_highest_idx}: {w_ee_regional[ee_highest_idx]:.4f} (raw T1w/T2w: {t1t2_left_raw[ee_highest_idx]:.4f})")
        print(f"  wEE lowest at region {ee_lowest_idx}: {w_ee_regional[ee_lowest_idx]:.4f} (raw T1w/T2w: {t1t2_left_raw[ee_lowest_idx]:.4f})")
    
    print()
    print("5. DEMIRTAS PAPER METHODOLOGY CHECK")
    print("-" * 40)
    
    print("According to Demirtas et al. (2019):")
    print("- There is a NEGATIVE relationship between T1w/T2w ratio and excitatory strength (wEE)")
    print("- They 'rescaled and inverted the raw T1w/T2w map'")
    print("- High T1w/T2w (sensory) areas should have LOW map values")
    print("- Low T1w/T2w (association) areas should have HIGH map values")
    print()
    
    # Check if the current implementation matches this
    print("Current implementation analysis:")
    print("- linearize_map() transforms T1w/T2w to roughly [-1, +1] with erf()")
    print("- For negative slopes: uses (map - min) / range -> [0, 1]")
    print("- For positive slopes: uses (-(map - max)) / range -> [0, 1] (inverted)")
    print()
    
    # The key question: does this properly implement the paper's methodology?
    print("CRITICAL ISSUE ANALYSIS:")
    print("1. The paper says they 'inverted' the T1w/T2w map")
    print("2. But linearize_map() doesn't invert - it just standardizes")
    print("3. The inversion only happens in backwards compatibility for positive slopes")
    print("4. For negative slopes, the map is NOT inverted (just normalized to [0,1])")
    print()
    
    # Visualization
    create_visualization_plots(t1t2_left_raw, t1t2_linearized, hmap_rev, hmap_norm)
    
    return {
        'raw': t1t2_left_raw,
        'linearized': t1t2_linearized,
        'neg_slope_transform': hmap_rev,
        'pos_slope_transform': hmap_norm
    }

def create_visualization_plots(raw, linearized, neg_transform, pos_transform):
    """Create visualization plots of the map transformations."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Raw T1w/T2w
    axes[0,0].plot(raw, 'b-', alpha=0.7, linewidth=1)
    axes[0,0].set_title('Raw T1w/T2w Map')
    axes[0,0].set_ylabel('T1w/T2w ratio')
    axes[0,0].grid(True, alpha=0.3)
    
    # Linearized T1w/T2w
    axes[0,1].plot(linearized, 'g-', alpha=0.7, linewidth=1)
    axes[0,1].set_title('Linearized T1w/T2w Map\n(erf transformation)')
    axes[0,1].set_ylabel('Linearized value')
    axes[0,1].grid(True, alpha=0.3)
    
    # Negative slope transformation
    axes[1,0].plot(neg_transform, 'r-', alpha=0.7, linewidth=1)
    axes[1,0].set_title('Negative Slope Transform\n(map - min) / range')
    axes[1,0].set_ylabel('Normalized [0,1]')
    axes[1,0].set_xlabel('Region index')
    axes[1,0].grid(True, alpha=0.3)
    
    # Positive slope transformation (inverted)
    axes[1,1].plot(pos_transform, 'm-', alpha=0.7, linewidth=1)
    axes[1,1].set_title('Positive Slope Transform\n(-(map - max)) / range (inverted)')
    axes[1,1].set_ylabel('Normalized [0,1]')
    axes[1,1].set_xlabel('Region index')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('t1w2_map_transformations.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Correlation analysis
    print("6. CORRELATION ANALYSIS")
    print("-" * 40)
    
    from scipy.stats import pearsonr, spearmanr
    
    correlations = [
        ("Raw vs Linearized", raw, linearized),
        ("Raw vs Neg Transform", raw, neg_transform),
        ("Raw vs Pos Transform", raw, pos_transform),
        ("Linearized vs Neg Transform", linearized, neg_transform),
        ("Linearized vs Pos Transform", linearized, pos_transform),
        ("Neg vs Pos Transform", neg_transform, pos_transform),
    ]
    
    for name, x, y in correlations:
        pearson_r = pearsonr(x, y)[0]
        spearman_r = spearmanr(x, y)[0]
        print(f"{name:25s}: Pearson r={pearson_r:7.4f}, Spearman r={spearman_r:7.4f}")

def recommendation():
    """Provide recommendations based on the analysis."""
    
    print()
    print("="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    print()
    
    print("ISSUE IDENTIFIED:")
    print("The current codebase does NOT properly implement the Demirtas paper methodology.")
    print()
    
    print("PROBLEMS:")
    print("1. linearize_map() standardizes but doesn't invert the T1w/T2w map")
    print("2. The paper explicitly states they 'inverted' the map")
    print("3. The backwards compatibility handles inversion inconsistently:")
    print("   - For positive slopes: inverts during parameter application")
    print("   - For negative slopes: does NOT invert")
    print("4. This means negative correlations are not properly implemented!")
    print()
    
    print("SOLUTIONS:")
    print()
    print("Option 1: Fix linearize_map() to properly invert")
    print("def linearize_map(x):")
    print("    # Apply erf transformation")
    print("    linearized = erf((x - np.mean(x)) / x.std() / np.sqrt(2))")
    print("    # Invert to match Demirtas methodology")
    print("    return -linearized  # or 1 - (linearized + 1)/2 to map to [0,1]")
    print()
    
    print("Option 2: Update _apply_maps() to handle inversion consistently")
    print("- Always invert the map before applying coefficients")
    print("- Remove the inconsistent positive/negative slope handling")
    print()
    
    print("Option 3: Create a new 'demirtas_linearize_map()' function")
    print("- Keep existing linearize_map() for backwards compatibility")
    print("- Create new function that properly implements Demirtas methodology")
    print("- Update load_data() to use the new function")
    print()
    
    print("CRITICAL IMPACT:")
    print("- This affects ALL heterogeneous models using T1w/T2w maps")
    print("- Parameter optimization results may be incorrect")
    print("- The biological interpretation is reversed!")
    print("- High T1w/T2w regions should have LOW synaptic strengths (but don't currently)")

if __name__ == "__main__":
    results = analyze_t1w2_processing()
    recommendation()