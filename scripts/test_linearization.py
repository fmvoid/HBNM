#!/usr/bin/env python3
"""
Simple test script to compare linearized vs non-linearized biological maps
and see their effect on model parameters.
"""

import os
import sys
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from hbnm.io import Data
from hbnm import Bnm
from optimization import load_data

def test_linearization_effect():
    """Test the effect of linearization on biological maps and model parameters."""
    
    # Setup
    input_dir = '../data/'
    output_dir = '../outputs/'
    data = Data(input_dir, output_dir)
    
    # Load structural connectivity
    sc, hmap_t1w2, fc_obj = load_data(data)
    
    print("=== Testing Linearization Effects ===\n")
    
    # Test with a few GRIN receptor maps
    test_maps = ['NMDA_avg', 'GRIN1', 'GRIN2A']
    
    for use_linearization in [False, True]:
        print(f"{'='*50}")
        print(f"Testing with linearization: {use_linearization}")
        print(f"{'='*50}")
        
        # Load biological maps
        from scripts.multi_map_optim import load_biological_maps
        try:
            maps = load_biological_maps(data, test_maps, linearize=use_linearization)
            
            # Create model
            model = Bnm(sc, maps=maps)
            
            # Test parameter application
            # Use small coefficients to see the effect
            wee_params = (0.15, 0.02, -0.01, 0.015)  # bias + 3 coefficients
            wei_params = (0.15, -0.01, 0.02, -0.005)  # bias + 3 coefficients
            
            model.set('w_EE', wee_params)
            model.set('w_EI', wei_params)
            model.set('G', 1.5)
            
            # Get resulting regional parameters
            w_ee_regional = model.get('w_EE')
            w_ei_regional = model.get('w_EI')
            
            print(f"\nRegional parameter statistics:")
            print(f"w_EE: min={w_ee_regional.min():.4f}, max={w_ee_regional.max():.4f}, "
                  f"mean={w_ee_regional.mean():.4f}, std={w_ee_regional.std():.4f}")
            print(f"w_EI: min={w_ei_regional.min():.4f}, max={w_ei_regional.max():.4f}, "
                  f"mean={w_ei_regional.mean():.4f}, std={w_ei_regional.std():.4f}")
            
            # Test stability
            try:
                is_unstable = model.check_stability()
                print(f"Model stability: {'UNSTABLE' if is_unstable else 'STABLE'}")
                
                if not is_unstable:
                    # Try to run moments method
                    model.moments_method(BOLD=True)
                    corr_bold = model.get('corr_bold')
                    if corr_bold is not None:
                        print(f"BOLD correlation range: {corr_bold.min():.3f} to {corr_bold.max():.3f}")
                    else:
                        print("BOLD correlation: None (moments method failed)")
                else:
                    print("Skipping moments method due to instability")
                    
            except Exception as e:
                print(f"Stability check failed: {e}")
            
        except Exception as e:
            print(f"Error loading maps: {e}")
        
        print()

def recommendation():
    """Provide recommendation based on the tests."""
    print("=== Recommendation ===")
    print()
    print("Based on the tests above, consider the following:")
    print()
    print("1. **Use linearization if:**")
    print("   - You want consistency with the T1w/T2w map processing")
    print("   - The linearized version produces more stable models")
    print("   - The parameter ranges look more reasonable")
    print()
    print("2. **Don't use linearization if:**")
    print("   - Your biological maps are already optimally scaled")
    print("   - The non-linearized version works better")
    print("   - You want to preserve the original HCP preprocessing")
    print()
    print("3. **The linearize_map() function:**")
    print("   - Applies: erf((x - mean(x)) / std(x) / sqrt(2))")
    print("   - Transforms any distribution to roughly [-1, 1] range")
    print("   - Makes the distribution more Gaussian-like")
    print()
    print("4. **For your GRIN receptor data:**")
    print("   - Try both approaches in your optimization")
    print("   - Compare model fit quality and stability")
    print("   - Use whichever gives better results")

if __name__ == "__main__":
    try:
        test_linearization_effect()
        recommendation()
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you're running this from the scripts directory")
        print("and that the biological map files exist in data/maps/") 