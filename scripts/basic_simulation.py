import numpy as np
from hbnm.bnm import Bnm

# Create a simple brain network
n_regions = 68
np.random.seed(42)
sc = np.random.rand(n_regions, n_regions)
sc = (sc + sc.T) / 2
np.fill_diagonal(sc, 0)

# Create model
model = Bnm(sc)

# Set parameters
model.set('w_EI', 1.0)
model.set('w_EE', 1.0) 
model.set('G', 0.5)

# Run simulation for 1 minute
model.dmf.integrate(
    t=60.0 # seconds
)

