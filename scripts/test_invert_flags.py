#!/usr/bin/env python3
"""
Test script to verify the new invert flags functionality.
"""

import os
import sys
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from hbnm.io import Data
from hbnm import Bnm
from optimization import load_data
from multi_map_optim import get_default_invert_flags, validate_maps_and_flags, load_biological_maps

def test_invert_flags():
    """Test the new invert flags functionality."""
    
    print("="*80)
    print("TESTING INVERT FLAGS FUNCTIONALITY")
    print("="*80)
    print()
    
    # Setup
    input_dir = './data/'
    output_dir = './outputs/'
    data = Data(input_dir, output_dir)
    
    # Load structural connectivity  
    sc, hmap_t1w2, fc_obj = load_data(data)
    
    print("1. TESTING SMART DEFAULTS")
    print("-" * 40)
    
    # Test smart defaults with different map types
    test_map_names = [
        ['t1wt2w'],
        ['NMDA_avg'],
        ['GRIN1', 'GRIN2A'],
        ['t1wt2w', 'NMDA_avg', 'GRIN1'],
        ['unknown_map']
    ]
    
    for names in test_map_names:
        print(f"\nMap names: {names}")
        flags = get_default_invert_flags(names)
        print(f"Generated flags: {flags}")
    
    print("\n2. TESTING MODEL WITH INVERT FLAGS")
    print("-" * 40)
    
    # Create synthetic maps for testing
    n_regions = 180
    
    # Create T1w/T2w-like map (high values in sensory areas)
    t1w2_map = hmap_t1w2  # Use real T1w/T2w map
    
    # Create NMDA-like map (should correlate positively with synaptic strength)
    nmda_map = np.random.normal(0, 1, n_regions)
    nmda_map = (nmda_map - nmda_map.min()) / np.ptp(nmda_map)  # Normalize to [0,1]
    
    # Stack maps
    test_maps = np.vstack([t1w2_map, nmda_map])
    test_map_names = ['t1wt2w', 'nmda_synthetic']
    test_invert_flags = [True, False]  # T1w/T2w inverted, NMDA direct
    
    print(f"Test maps shape: {test_maps.shape}")
    print(f"Map names: {test_map_names}")
    print(f"Invert flags: {test_invert_flags}")
    
    # Validate maps and flags
    validate_maps_and_flags(test_maps, test_invert_flags, test_map_names)
    
    # Test model creation
    print("\n3. TESTING MODEL CREATION")
    print("-" * 40)
    
    try:
        model = Bnm(sc, maps=test_maps, map_invert_flags=test_invert_flags)
        print("✓ Model created successfully")
        
        # Test parameter application
        test_params = (0.15, 0.02, -0.01)  # bias + 2 coefficients
        model.set('w_EE', test_params)
        model.set('w_EI', test_params)
        model.set('G', 1.0)
        
        # Get regional parameters
        w_ee_regional = model.get('w_EE')
        w_ei_regional = model.get('w_EI')
        
        print(f"✓ Parameters applied successfully")
        print(f"  wEE range: [{w_ee_regional.min():.4f}, {w_ee_regional.max():.4f}]")
        print(f"  wEI range: [{w_ei_regional.min():.4f}, {w_ei_regional.max():.4f}]")
        
        # Test stability
        is_unstable = model.check_stability()
        print(f"✓ Stability check completed (unstable: {is_unstable})")
        
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False
    
    print("\n4. TESTING BACKWARDS COMPATIBILITY")
    print("-" * 40)
    
    try:
        # Test single map backwards compatibility
        model_compat = Bnm(sc, gradient=hmap_t1w2)  # Old style
        model_compat.set('w_EE', (0.15, 0.05))
        model_compat.set('w_EI', (0.15, 0.03))
        model_compat.set('G', 1.0)
        
        w_ee_compat = model_compat.get('w_EE')
        print(f"✓ Backwards compatibility works")
        print(f"  wEE range: [{w_ee_compat.min():.4f}, {w_ee_compat.max():.4f}]")
        
    except Exception as e:
        print(f"✗ Backwards compatibility failed: {e}")
        return False
    
    print("\n5. TESTING ERROR HANDLING")
    print("-" * 40)
    
    # Test mismatched flag count
    try:
        wrong_flags = [True]  # Only 1 flag for 2 maps
        model_error = Bnm(sc, maps=test_maps, map_invert_flags=wrong_flags)
        print("✗ Should have raised ValueError for mismatched flag count")
        return False
    except ValueError as e:
        print(f"✓ Correctly caught flag count mismatch: {e}")
    
    # Test constant map
    try:
        constant_map = np.ones((1, n_regions))  # Constant map
        constant_flags = [True]
        model_const = Bnm(sc, maps=constant_map, map_invert_flags=constant_flags)
        model_const.set('w_EE', (0.15, 0.05))  # This should fail
        print("✗ Should have raised ValueError for constant map")
        return False
    except ValueError as e:
        print(f"✓ Correctly caught constant map: {e}")
    
    print("\n" + "="*80)
    print("ALL TESTS PASSED!")
    print("="*80)
    return True

if __name__ == "__main__":
    success = test_invert_flags()
    if not success:
        sys.exit(1) 