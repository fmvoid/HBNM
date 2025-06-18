#!/usr/bin/env python3

import numpy as np
from hbnm.bnm import Bnm

print("Testing real-world brain network model...")

# Create a more realistic brain network (68 regions like in the papers)
n_regions = 68
np.random.seed(42)  # Reproducible results

# Create a random structural connectivity matrix (normally you'd load real data)
sc = np.random.rand(n_regions, n_regions)
sc = (sc + sc.T) / 2  # Make symmetric
np.fill_diagonal(sc, 0)  # No self-connections

print(f"✓ Created {n_regions}x{n_regions} structural connectivity matrix")

# Create the model
model = Bnm(sc)
print("✓ Created brain network model")

# Set some parameters (from the example scripts)
model.set('w_EI', 1.0)
model.set('w_EE', 1.0) 
model.set('G', 0.5)
print("✓ Set model parameters")

# Run the computation
model.moments_method(BOLD=True)
print("✓ Computed BOLD functional connectivity")

# Get results
fc = model.get('corr_bold')
print(f"✓ Retrieved FC matrix: {fc.shape}")
print(f"✓ FC correlation range: {fc.min():.3f} to {fc.max():.3f}")


print("\n🧠 SUCCESS: Full brain network simulation working!")