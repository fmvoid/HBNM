#!/usr/bin/env python3

import sys
import traceback

print(f"Python version: {sys.version}")
print(f"Current working directory: {sys.path[0]}")

try:
    print("Attempting to import hbnm...")
    import hbnm
    print("✓ Successfully imported hbnm")
    
    print("Attempting to import hbnm.bnm...")
    import hbnm.bnm as bnm
    print("✓ Successfully imported hbnm.bnm")
    
except Exception as e:
    print(f"✗ Import failed: {e}")
    print("\nFull traceback:")
    traceback.print_exc()