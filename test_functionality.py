#!/usr/bin/env python3

import numpy as np
import sys

try:
    print(f"Python version: {sys.version}")
    
    # Import the main class
    from hbnm.bnm import Bnm
    print("✓ Successfully imported Bnm class")
    
    # Try to create a simple brain network model
    # Create a simple 2x2 structural connectivity matrix
    sc = np.array([[0, 1], [1, 0]])
    print("✓ Created test structural connectivity matrix")
    
    # Try to instantiate the model
    model = Bnm(sc)
    print("✓ Successfully created Bnm model instance")
    
    # Try a basic method
    print("Available methods:", [method for method in dir(model) if not method.startswith('_')])
    
    # Try calling a simple method
    print("\nTesting basic methods...")
    
    # Test getting structural connectivity back
    sc_retrieved = model.get('SC')
    print("✓ Retrieved structural connectivity:", sc_retrieved.shape)
    
    # Try the core computational method (BOLD)
    model.moments_method(BOLD=True)
    print("✓ Successfully ran moments_method")

    # Try getting results
    correlation = model.get('corr_bold')
    print("✓ Retrieved correlation matrix:", correlation.shape)


except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()