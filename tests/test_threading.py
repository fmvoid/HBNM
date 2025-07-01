# save as test_cpu_usage.py
import numpy as np
import time
import subprocess
import os

def get_cpu_usage():
    """Get current CPU usage percentage"""
    try:
        # macOS command to get CPU usage
        result = subprocess.run(['top', '-l', '1', '-n', '0'], 
                              capture_output=True, text=True, timeout=2)
        for line in result.stdout.split('\n'):
            if 'CPU usage:' in line:
                return line.strip()
    except:
        return "CPU usage: unknown"

def heavy_matrix_operations():
    """Simulate the exact operations your optimization does"""
    print("Starting heavy matrix operations...")
    print("Initial CPU:", get_cpu_usage())
    
    # This mimics what happens in moments_method() and check_stability()
    n = 2000
    np.random.seed(42)
    
    # Create matrices similar to your model
    A = np.random.rand(n, n).astype(np.float64)
    A = (A + A.T) / 2  # Make symmetric like your SC matrices
    
    print("Running eigenvalue decomposition (like check_stability)...")
    start = time.time()
    
    # This is the expensive operation in your model
    eigenvals, eigenvects = np.linalg.eig(A)
    
    mid = time.time()
    print(f"Eigenvalues computed in {mid-start:.2f}s")
    print("CPU during eigenvalue computation:", get_cpu_usage())
    
    # Matrix inversion (like in FIC calculation)  
    print("Running matrix inversion (like FIC calculation)...")
    inv_A = np.linalg.pinv(A[:500, :500])  # Smaller for faster test
    
    end = time.time()
    print(f"Total time: {end-start:.2f}s")
    print("Final CPU:", get_cpu_usage())

if __name__ == "__main__":
    heavy_matrix_operations()