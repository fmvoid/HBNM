import numpy as np
import matplotlib.pyplot as plt
from hbnm.io import Data
from hbnm.bnm import Bnm
from hbnm.model.utils import subdiag, fisher_z
from scipy.stats import pearsonr
import os

def analyze_optimization_results(data_path='heterogeneous/', 
                                iteration_file='iteration_2.hdf5'):
    """
    Analyze optimization results to find best parameter sets.
    
    Parameters
    ----------
    data_path : str
        Path to optimization results
    iteration_file : str  
        Name of iteration file to analyze
    """
    
    # Load data
    current_path = os.getcwd()
    input_dir = current_path + '/data/'
    output_dir = current_path + '/outputs/'
    
    data = Data(input_dir, output_dir)
    
    # Load iteration results
    fin = data.load(f'{data_path.replace(output_dir, "")}/{iteration_file}', from_output=True)
    theta = fin['theta'][:]
    distances = fin['distance'][:]
    weights = fin['weights'][:]
    epsilon = fin['epsilon'][()]
    fin.close()
    
    n_particles = theta.shape[1]
    n_params = theta.shape[0]
    
    print(f"=== Optimization Results Analysis ===")
    print(f"Number of particles: {n_particles}")
    print(f"Number of parameters: {n_params}")
    print(f"Rejection threshold (epsilon): {epsilon:.6f}")
    
    # Find best parameter sets
    # Sort by distance (ascending - smaller is better)
    sorted_indices = np.argsort(distances)
    best_indices = sorted_indices[:5]  # Top 5 best fits
    
    print(f"\n=== Top 5 Best Parameter Sets ===")
    for i, idx in enumerate(best_indices):
        distance = distances[idx]
        fit_quality = 1.0 - distance  # Convert back to correlation-like measure
        weight = weights[idx]
        
        print(f"\nRank {i+1}:")
        print(f"  Particle index: {idx}")
        print(f"  Distance: {distance:.6f}")
        print(f"  Fit quality (1-distance): {fit_quality:.6f}")
        print(f"  Weight: {weight:.6f}")
        print(f"  Parameters:")
        
        param_names = ['w_EI_min', 'w_EI_scale', 'w_EE_min', 'w_EE_scale', 'G']
        for j, param_name in enumerate(param_names):
            print(f"    {param_name}: {theta[j, idx]:.6f}")
    
    # Calculate statistics across all particles
    print(f"\n=== Parameter Statistics Across All Particles ===")
    param_names = ['w_EI_min', 'w_EI_scale', 'w_EE_min', 'w_EE_scale', 'G']
    for i, param_name in enumerate(param_names):
        mean_val = np.mean(theta[i, :])
        std_val = np.std(theta[i, :])
        weighted_mean = np.average(theta[i, :], weights=weights)
        
        print(f"{param_name}:")
        print(f"  Mean: {mean_val:.6f} ± {std_val:.6f}")
        print(f"  Weighted mean: {weighted_mean:.6f}")
    
    # Plot results
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Distance distribution
    axes[0, 0].hist(distances, bins=20, alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(distances[best_indices[0]], color='red', linestyle='--', 
                      label=f'Best fit: {distances[best_indices[0]]:.4f}')
    axes[0, 0].set_xlabel('Distance')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distance Distribution')
    axes[0, 0].legend()
    
    # Parameter distributions
    param_names = ['w_EI_min', 'w_EI_scale', 'w_EE_min', 'w_EE_scale', 'G']
    for i, param_name in enumerate(param_names):
        row = (i + 1) // 3
        col = (i + 1) % 3
        
        axes[row, col].hist(theta[i, :], bins=20, alpha=0.7, edgecolor='black')
        axes[row, col].axvline(theta[i, best_indices[0]], color='red', linestyle='--',
                              label=f'Best: {theta[i, best_indices[0]]:.4f}')
        axes[row, col].set_xlabel(param_name)
        axes[row, col].set_ylabel('Frequency')
        axes[row, col].set_title(f'{param_name} Distribution')
        axes[row, col].legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/optimization_analysis.png', dpi=300, bbox_inches='tight')
    # plt.show()
    
    return theta, distances, weights, best_indices

def evaluate_best_fit(theta, best_idx, save_plots=True):
    """
    Evaluate and plot the best fitting model.
    
    Parameters
    ----------
    theta : ndarray
        Parameter matrix
    best_idx : int
        Index of best parameter set
    save_plots : bool
        Whether to save plots
    """
    
    # Load empirical data
    current_path = os.getcwd()
    input_dir = current_path + '/data/'
    output_dir = current_path + '/outputs/'
    
    data = Data(input_dir, output_dir)
    sc, hmap, fc_obj = data.load_demirtas_neuron_2019_data()
    
    # Set up model with best parameters
    model = Bnm(sc, gradient=hmap)
    model.set('w_EI', (theta[0, best_idx], theta[1, best_idx]))
    model.set('w_EE', (theta[2, best_idx], theta[3, best_idx]))
    model.set('G', theta[4, best_idx])
    model.moments_method()
    
    # Calculate fit quality
    model_fc = model.get('corr_bold')
    correlation = pearsonr(subdiag(fc_obj), subdiag(model_fc))[0]
    
    print(f"\n=== Best Fit Model Evaluation ===")
    print(f"Correlation with empirical FC: {correlation:.4f}")
    print(f"Best parameters:")
    param_names = ['w_EI_min', 'w_EI_scale', 'w_EE_min', 'w_EE_scale', 'G']
    for i, param_name in enumerate(param_names):
        print(f"  {param_name}: {theta[i, best_idx]:.6f}")
    
    if save_plots:
        # Create comparison plot
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Empirical FC
        im1 = axes[0].imshow(fc_obj, cmap='RdBu_r', vmin=-1, vmax=1)
        axes[0].set_title('Empirical FC')
        axes[0].set_aspect('equal')
        plt.colorbar(im1, ax=axes[0])
        
        # Model FC
        im2 = axes[1].imshow(model_fc, cmap='RdBu_r', vmin=-1, vmax=1)
        axes[1].set_title(f'Best Model FC (r={correlation:.3f})')
        axes[1].set_aspect('equal')
        plt.colorbar(im2, ax=axes[1])
        
        # Scatter plot
        emp_fc_vec = fisher_z(subdiag(fc_obj))
        model_fc_vec = fisher_z(subdiag(model_fc))
        
        axes[2].scatter(model_fc_vec, emp_fc_vec, alpha=0.6)
        axes[2].set_xlabel('Model FC (Fisher z-transformed)')
        axes[2].set_ylabel('Empirical FC (Fisher z-transformed)')
        axes[2].set_title(f'Model vs Empirical FC\n(r = {correlation:.3f})')
        
        # Add diagonal line
        min_val = min(emp_fc_vec.min(), model_fc_vec.min())
        max_val = max(emp_fc_vec.max(), model_fc_vec.max())
        axes[2].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/best_fit_analysis.png', dpi=300, bbox_inches='tight')
        # plt.show()
    
    return correlation, model_fc

if __name__ == "__main__":
    # Analyze optimization results
    theta, distances, weights, best_indices = analyze_optimization_results(data_path='heterogeneous', iteration_file='iteration_1.hdf5')
    
    # Evaluate best fit
    best_idx = best_indices[0]  # Best fit index
    correlation, model_fc = evaluate_best_fit(theta, best_idx)