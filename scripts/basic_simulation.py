import numpy as np
from hbnm.bnm import Bnm
from hbnm.io import Data
import matplotlib.pyplot as plt
import os 

# Setup data
current_path = os.getcwd()
input_dir = current_path + '/data/'
output_dir = current_path + '/outputs/'

data = Data(input_dir, output_dir)
sc, hmap, fc_obj = data.load_demirtas_data()

# Create a simple brain network
# Create model
model = Bnm(sc, hmap)

# Set parameters
model.set('w_EI', [1.0, 2.0])
model.set('w_EE', [1.0, 2.0]) 
model.set('G', 0.5)

# Run simulation for 1 minute
model.dmf.integrate(
    t=60.0 # seconds
)

# ===== GET SIMULATION RESULTS =====

# Set region number
n_regions = 180

# 1. Get time series data

S_E = model.dmf.sim.S_E  # Excitatory synaptic gating variables (regions x time)
S_I = model.dmf.sim.S_I  # Inhibitory synaptic gating variables


time_points = model.dmf.sim.t_points  # Time points array
r_E = model.dmf.sim.time_series('r_E')  # Excitatory firing rates (Hz)
r_I = model.dmf.sim.time_series('r_I')  # Inhibitory firing rates (Hz)
BOLD = model.dmf.sim.time_series('y') # BOLD signals (% change)

print(f"Simulation duration: {time_points[-1]:.1f} seconds")
print(f"Sampling rate: {1/model.dmf.sim.dt:.1f} Hz")
print(f"Mean excitatory firing rate: {r_E.mean():.2f} Hz")
print(f"Mean inhibitory firing rate: {r_I.mean():.2f} Hz")

# 2. Compute functional connectivity from BOLD
BOLD_fc = model.dmf.sim.BOLD_corr(t_cutoff=10.0)  # Skip first 10 seconds
print(f"BOLD FC matrix shape: {BOLD_fc.shape}")
print(f"Mean FC: {BOLD_fc[np.triu_indices(n_regions, k=1)].mean():.3f}")

# 3. Compute functional connectivity from synaptic activity  
S_fc = model.dmf.sim.S_corr(t_cutoff=10.0)  # E-E correlations in upper-left quadrant
S_E_fc = S_fc[:n_regions, :n_regions]  # Extract E-E correlations
print(f"Synaptic FC matrix shape: {S_E_fc.shape}")
print(f"Mean synaptic FC: {S_E_fc[np.triu_indices(n_regions, k=1)].mean():.3f}")


# ===== VISUALIZATION =====

# Create plots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Plot 1: Time series of first 5 regions
axes[0, 0].plot(time_points[:30000], S_E[:3, :30000].T)  # First 5 seconds
axes[0, 0].set_xlabel('Time (s)')
axes[0, 0].set_ylabel('Synaptic Activity S_E')
axes[0, 0].set_title('Excitatory Synaptic Activity (first 15 regions)')
axes[0, 0].legend([f'Region {i+1}' for i in range(5)], fontsize=8)

# Plot 2: BOLD time series
axes[0, 1].plot(time_points[:50000], BOLD[:15, :50000].T)
axes[0, 1].set_xlabel('Time (s)')
axes[0, 1].set_ylabel('BOLD Signal (% change)')
axes[0, 1].set_title('BOLD Signals (first 15 regions)')

# Plot 3: Firing rate distribution
axes[0, 2].hist(r_E.flatten(), bins=50, alpha=0.7, label='Excitatory')
axes[0, 2].hist(r_I.flatten(), bins=50, alpha=0.7, label='Inhibitory')
axes[0, 2].set_xlabel('Firing Rate (Hz)')
axes[0, 2].set_ylabel('Count')
axes[0, 2].set_title('Firing Rate Distribution')
axes[0, 2].legend()

# Plot 4: BOLD FC matrix
im1 = axes[1, 0].imshow(BOLD_fc, cmap='RdBu_r', vmin=-1, vmax=1)
axes[1, 0].set_title('BOLD Functional Connectivity')
axes[1, 0].set_xlabel('Region')
axes[1, 0].set_ylabel('Region')
plt.colorbar(im1, ax=axes[1, 0], shrink=0.8)

# Plot 5: Synaptic FC matrix  
im2 = axes[1, 1].imshow(S_E_fc, cmap='RdBu_r', vmin=-1, vmax=1)
axes[1, 1].set_title('Synaptic Functional Connectivity')
axes[1, 1].set_xlabel('Region')
axes[1, 1].set_ylabel('Region')
plt.colorbar(im2, ax=axes[1, 1], shrink=0.8)

# Plot 6: FC comparison
fc_bold_vals = BOLD_fc[np.triu_indices(n_regions, k=1)]
fc_syn_vals = S_E_fc[np.triu_indices(n_regions, k=1)]
axes[1, 2].scatter(fc_syn_vals, fc_bold_vals, alpha=0.5, s=1)
axes[1, 2].set_xlabel('Synaptic FC')
axes[1, 2].set_ylabel('BOLD FC')
axes[1, 2].set_title('BOLD vs Synaptic FC Correlation')
corr_coef = np.corrcoef(fc_syn_vals, fc_bold_vals)[0, 1]
axes[1, 2].text(0.05, 0.95, f'r = {corr_coef:.3f}', transform=axes[1, 2].transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('simulation_results_real.png', dpi=300, bbox_inches='tight')
plt.show()
