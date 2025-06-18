#!/usr/bin/env python3

import os
import sys
import numpy as np
import h5py

print(f"Python version: {sys.version}")

try:
    # Test 1: Import the Data class
    from hbnm.io import Data
    print("✓ Successfully imported Data class")
    
    # Test 2: Set up paths
    current_path = os.getcwd()
    input_dir = current_path + '/data/'
    output_dir = current_path + '/outputs/'
    print(f"Current path: {current_path}")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    
    # Test 3: Instantiate the class
    data_handler = Data(input_dir, output_dir)
    print("✓ Successfully created Data instance")
    print(f"✓ Output directory created: {os.path.exists(output_dir)}")
    
    # Test 4: List available files
    if os.path.exists(input_dir):
        files = os.listdir(input_dir)
        hdf5_files = [f for f in files if f.endswith('.hdf5')]
        print(f"✓ Found HDF5 files: {hdf5_files}")
    else:
        print(f"✗ Input directory {input_dir} doesn't exist")

    # Test 5: Load the HDF5 research data
    print("\n=== Loading Research Data ===")
    hdf5_file = data_handler.load('demirtas_neuron_2019.hdf5')
    print("✓ Successfully opened HDF5 file")
    
    # Explore what's inside
    print("\nDatasets in the file:")

    # Check each dataset's properties
    for key in hdf5_file.keys():
        if isinstance(hdf5_file[key], h5py.Dataset):
            dset = hdf5_file[key]
            print(f"{key}: shape={dset.shape}, dtype={dset.dtype}")
            
            # Data range for numerical data
            if np.issubdtype(dset.dtype, np.number):
                data = dset[...]  # Load the data
                print(f"  Range: {data.min():.4f} to {data.max():.4f}")
        
    # Don't forget to close the file
    hdf5_file.close()
    print("\n✓ Closed HDF5 file")
        
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()