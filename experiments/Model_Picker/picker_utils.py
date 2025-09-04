"""
Brain Network Model Analysis Utilities

This module provides modular functions for analyzing fitted brain network models:

CORE COMPUTATION FUNCTIONS (minimal output):
- find_fixed_point(model): Find fixed point x* including long-range G·K·S_E effects
- compute_jacobian(model, include_bold): Compute Jacobian A = ∂f/∂x |_{x*}
- eigen_analysis(jacobian): Compute eigenvalues and eigenvectors

ANALYSIS FUNCTIONS (verbose output for debugging):
- analyze_fixed_point(fixed_point, verbose): Analyze fixed point properties
- analyze_jacobian_structure(jacobian, n_regions, include_bold, verbose): Analyze Jacobian structure
- analyze_eigenspectrum(eigen_results, verbose): Analyze stability and eigenspectrum

PIPELINE FUNCTIONS (structured output for batch processing):
- analyze_stability_and_margin(eigen_results): Step 2 - Stability & margin analysis
- run_homotopy_check(theta, sc, hmap, knob, perturbation_pct): Step 3 - Homotopy (knife-edge check)

- setup_particle_model(theta, sc, hmap): Setup model for single particle
- analyze_single_particle(particle_idx, theta, sc, hmap, homotopy_knob, homotopy_perturbation_pct): Complete analysis for one particle
- analyze_all_particles(theta_heterogeneous, sc, hmap, save_path, homotopy_knob, homotopy_perturbation_pct): Batch analysis with CSV output

VISUALIZATION FUNCTIONS:
- visualize_jacobian(jacobian, n_regions, include_bold): Plot Jacobian matrix structure

TYPICAL WORKFLOWS:

Single particle (interactive):
    # Core computations
    fixed_point = find_fixed_point(model)
    jacobian = compute_jacobian(model, include_bold=False)
    eigen_results = eigen_analysis(jacobian)
    
    # Optional analysis
    analyze_fixed_point(fixed_point)
    analyze_eigenspectrum(eigen_results)
    visualize_jacobian(jacobian, model.dmf._nc)

Batch processing:
    # Analyze all particles with Steps 2 and 3
    results_df = analyze_all_particles(
        theta_heterogeneous, sc, hmap, 'results.csv',
        homotopy_knob='G', homotopy_perturbation_pct=10.0
    )
    
    # Filter for particles passing both stability and homotopy checks
    good_particles = results_df[
        results_df['passes_stability'] & results_df['passes_homotopy']
    ]
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from hbnm.bnm import Bnm

def find_fixed_point(model):
    """
    Find the fixed point of the full network model.
    
    This function extracts the steady-state values that include long-range input 
    G·K·S_E and ensures FIC (Feedback Inhibitory Control) is applied at the 
    operating point for consistency.
    
    Parameters:
    -----------
    model : Bnm
        Brain network model with fitted parameters
        
    Returns:
    --------
    dict : Fixed point state variables including synaptic gating variables,
           firing rates, and input currents
    """
    # Ensure model parameters are set and FIC is computed at the fixed point
    # This handles both the Jacobian computation and FIC consistency
    model.dmf.set_jacobian(compute_fic=True)
    
    # Extract fixed point values (already includes long-range G·K·S_E effects)
    fixed_point = {
        'S_E': model.dmf._S_E_ss.copy(),  # Excitatory synaptic gating variables
        'S_I': model.dmf._S_I_ss.copy(),  # Inhibitory synaptic gating variables  
        'r_E': model.dmf._r_E_ss.copy(),  # Excitatory firing rates
        'r_I': model.dmf._r_I_ss.copy(),  # Inhibitory firing rates
        'I_E': model.dmf._I_E_ss.copy(),  # Excitatory input currents
        'I_I': model.dmf._I_I_ss.copy(),  # Inhibitory input currents
    }
    
    return fixed_point


def analyze_fixed_point(fixed_point, verbose=True):
    """
    Analyze and verify the fixed point solution.
    
    Parameters:
    -----------
    fixed_point : dict
        Fixed point data from find_fixed_point()
    verbose : bool
        Whether to print detailed analysis
        
    Returns:
    --------
    dict : Verification metrics including derivative magnitudes
    """
    S_E_fixed = fixed_point['S_E'] 
    S_I_fixed = fixed_point['S_I']
    r_E_fixed = fixed_point['r_E']
    r_I_fixed = fixed_point['r_I']
    
    if verbose:
        print("=== FIXED POINT ANALYSIS ===")
        print(f"Number of regions: {len(S_E_fixed)}")
        print(f"S_E fixed point range: [{S_E_fixed.min():.4f}, {S_E_fixed.max():.4f}]")
        print(f"S_I fixed point range: [{S_I_fixed.min():.4f}, {S_I_fixed.max():.4f}]")
        print(f"Excitatory firing rate range: [{r_E_fixed.min():.2f}, {r_E_fixed.max():.2f}] Hz")
        print(f"Inhibitory firing rate range: [{r_I_fixed.min():.2f}, {r_I_fixed.max():.2f}] Hz")
    
    return {
        'n_regions': len(S_E_fixed),
        'S_E_range': (S_E_fixed.min(), S_E_fixed.max()),
        'S_I_range': (S_I_fixed.min(), S_I_fixed.max()), 
        'r_E_range': (r_E_fixed.min(), r_E_fixed.max()),
        'r_I_range': (r_I_fixed.min(), r_I_fixed.max())
    }

def compute_jacobian(model, include_bold=False):
    """
    Compute the Jacobian matrix A = ∂f/∂x at the fixed point x*.
    
    The Jacobian is computed using the model's built-in method (likely finite 
    differences with appropriately scaled steps). This gives the linearization
    of the system around the steady state.
    
    Parameters:
    -----------
    model : Bnm
        Brain network model with computed fixed point
    include_bold : bool
        If True, include hemodynamic variables in Jacobian
        
    Returns:
    --------
    numpy.ndarray : Jacobian matrix A = ∂f/∂x |_{x*}
    """
    # Ensure Jacobian is computed (this was already called in find_fixed_point)
    model.dmf.set_jacobian(compute_fic=True)
    
    if include_bold:
        # Extended Jacobian including hemodynamics (S_E, S_I, x, f, v, q)
        model.dmf.moments_method(bold=True)
        J = model.dmf._jacobian_bold.copy()
    else:
        # Synaptic-only Jacobian (S_E, S_I)
        J = model.dmf._jacobian.copy()
    
    return J


def analyze_jacobian_structure(jacobian, n_regions, include_bold=False, verbose=True):
    """
    Analyze the structure of the Jacobian matrix.
    
    Parameters:
    -----------
    jacobian : numpy.ndarray
        Jacobian matrix from compute_jacobian()
    n_regions : int
        Number of brain regions
    include_bold : bool
        Whether Jacobian includes hemodynamic variables
    verbose : bool
        Whether to print detailed analysis
        
    Returns:
    --------
    dict : Structural analysis metrics
    """
    if include_bold:
        system_name = "Synaptic + Hemodynamic"
        n_vars_per_region = 6  # S_E, S_I, x, f, v, q
    else:
        system_name = "Synaptic Only"  
        n_vars_per_region = 2  # S_E, S_I
    
    analysis = {
        'system_name': system_name,
        'shape': jacobian.shape,
        'n_regions': n_regions,
        'n_vars_per_region': n_vars_per_region
    }
    
    if verbose:
        print(f"\n=== JACOBIAN ANALYSIS ({system_name}) ===")
        print(f"Jacobian shape: {jacobian.shape}")
        print(f"Number of regions: {n_regions}")
        print(f"Variables per region: {n_vars_per_region}")
    
    # For synaptic-only system, analyze block structure
    if not include_bold and verbose:
        A_EE = jacobian[:n_regions, :n_regions]
        A_EI = jacobian[:n_regions, n_regions:]
        A_IE = jacobian[n_regions:, :n_regions] 
        A_II = jacobian[n_regions:, n_regions:]
        
        print(f"\nJacobian block structure:")
        print(f"A_EE (E→E): {A_EE.shape}, diagonal range: [{np.diag(A_EE).min():.3f}, {np.diag(A_EE).max():.3f}]")
        print(f"A_EI (I→E): {A_EI.shape}, diagonal range: [{np.diag(A_EI).min():.3f}, {np.diag(A_EI).max():.3f}]")  
        print(f"A_IE (E→I): {A_IE.shape}, diagonal range: [{np.diag(A_IE).min():.3f}, {np.diag(A_IE).max():.3f}]")
        print(f"A_II (I→I): {A_II.shape}, diagonal range: [{np.diag(A_II).min():.3f}, {np.diag(A_II).max():.3f}]")
        
        analysis.update({
            'A_EE_diag_range': (np.diag(A_EE).min(), np.diag(A_EE).max()),
            'A_EI_diag_range': (np.diag(A_EI).min(), np.diag(A_EI).max()),
            'A_IE_diag_range': (np.diag(A_IE).min(), np.diag(A_IE).max()),
            'A_II_diag_range': (np.diag(A_II).min(), np.diag(A_II).max())
        })
    
    return analysis


def visualize_jacobian(jacobian, n_regions, include_bold=False):
    """
    Visualize the structure of the Jacobian matrix.
    
    Parameters:
    -----------
    jacobian : numpy.ndarray
        Jacobian matrix to visualize
    n_regions : int 
        Number of brain regions (for block structure annotation)
    include_bold : bool
        Whether Jacobian includes hemodynamic variables
    """
    system_name = "Synaptic + Hemodynamic" if include_bold else "Synaptic Only"
    
    plt.figure(figsize=(10, 8))
    plt.imshow(jacobian, cmap='RdBu_r', aspect='auto')
    plt.colorbar(label='Jacobian elements')
    plt.title(f'Jacobian Matrix Structure ({system_name})')
    plt.xlabel('Variable index')
    plt.ylabel('Variable index')
    
    if not include_bold:
        # Add lines to show block structure for synaptic-only case
        plt.axhline(n_regions-0.5, color='black', linewidth=2, alpha=0.7)
        plt.axvline(n_regions-0.5, color='black', linewidth=2, alpha=0.7)
        plt.text(n_regions/4, n_regions/4, 'A_EE', ha='center', va='center', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        plt.text(3*n_regions/2, n_regions/4, 'A_EI', ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        plt.text(n_regions/4, 3*n_regions/2, 'A_IE', ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        plt.text(3*n_regions/2, 3*n_regions/2, 'A_II', ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()


def eigen_analysis(jacobian):
    """
    Perform eigenvalue and eigenvector analysis of the Jacobian matrix.
    
    This computes eigenvalues λ and eigenvectors v such that A·v = λ·v,
    which characterizes the linear stability and dynamics around the fixed point.
    
    Parameters:
    -----------
    jacobian : numpy.ndarray
        Jacobian matrix A = ∂f/∂x |_{x*}
        
    Returns:
    --------
    dict : Contains eigenvalues, eigenvectors, and stability metrics
        - 'eigenvalues': complex eigenvalues λ
        - 'eigenvectors': corresponding eigenvectors (columns of returned matrix)
        - 'real_parts': real parts of eigenvalues  
        - 'imag_parts': imaginary parts of eigenvalues
        - 'dominant_eigenvalue': eigenvalue with largest real part
        - 'is_stable': whether fixed point is linearly stable (all λ have Re(λ) < 0)
    """
    # Compute eigenvalues and eigenvectors
    eigenvals, eigenvecs = np.linalg.eig(jacobian)
    
    # Extract real and imaginary parts
    real_parts = eigenvals.real
    imag_parts = eigenvals.imag
    
    # Find dominant eigenvalue (largest real part)
    dominant_idx = np.argmax(real_parts)
    dominant_eigenval = eigenvals[dominant_idx]
    
    # Check stability: stable if all eigenvalues have negative real parts
    is_stable = np.all(real_parts < 0)
    
    return {
        'eigenvalues': eigenvals,
        'eigenvectors': eigenvecs,
        'real_parts': real_parts,
        'imag_parts': imag_parts,
        'dominant_eigenvalue': dominant_eigenval,
        'dominant_index': dominant_idx,
        'is_stable': is_stable,
        'n_unstable': np.sum(real_parts >= 0),
        'spectral_radius': np.max(np.abs(eigenvals))
    }


def analyze_eigenspectrum(eigen_results, verbose=True):
    """
    Analyze the eigenspectrum for stability and dynamical insights.
    
    Parameters:
    -----------
    eigen_results : dict
        Results from eigen_analysis()
    verbose : bool
        Whether to print detailed analysis
        
    Returns:
    --------
    dict : Analysis summary including stability classification
    """
    eigenvals = eigen_results['eigenvalues']
    real_parts = eigen_results['real_parts']
    imag_parts = eigen_results['imag_parts']
    dominant_eigenval = eigen_results['dominant_eigenvalue']
    is_stable = eigen_results['is_stable']
    n_unstable = eigen_results['n_unstable']
    
    # Classify eigenvalues
    n_real = np.sum(np.abs(imag_parts) < 1e-10)
    n_complex = len(eigenvals) - n_real
    n_positive_real = np.sum(real_parts > 1e-10)
    n_negative_real = np.sum(real_parts < -1e-10)
    n_zero_real = np.sum(np.abs(real_parts) < 1e-10)
    
    if verbose:
        print("\n=== EIGENSPECTRUM ANALYSIS ===")
        print(f"Total eigenvalues: {len(eigenvals)}")
        print(f"Real eigenvalues: {n_real}")
        print(f"Complex eigenvalues: {n_complex} ({n_complex//2} conjugate pairs)" if n_complex > 0 else f"Complex eigenvalues: 0")
        print(f"\nStability analysis:")
        print(f"  Positive real parts: {n_positive_real}")
        print(f"  Negative real parts: {n_negative_real}")  
        print(f"  Near-zero real parts: {n_zero_real}")
        print(f"  System is: {'STABLE' if is_stable else 'UNSTABLE'}")
        
        print(f"\nDominant eigenvalue: {dominant_eigenval:.6f}")
        if np.abs(dominant_eigenval.imag) > 1e-10:
            freq = np.abs(dominant_eigenval.imag) / (2 * np.pi)
            print(f"  Dominant frequency: {freq:.3f} Hz")
        
        print(f"Spectral radius: {eigen_results['spectral_radius']:.6f}")
        
        # Show a few most unstable eigenvalues if system is unstable
        if not is_stable:
            unstable_idx = real_parts >= 0
            unstable_eigs = eigenvals[unstable_idx]
            print(f"\nUnstable eigenvalues ({n_unstable}):")
            for i, eig in enumerate(unstable_eigs[:5]):  # Show up to 5
                print(f"  {i+1}: {eig:.6f}")
    
    return {
        'n_total': len(eigenvals),
        'n_real': n_real,
        'n_complex': n_complex,
        'n_positive_real': n_positive_real,
        'n_negative_real': n_negative_real,
        'n_zero_real': n_zero_real,
        'stability': 'stable' if is_stable else 'unstable',
        'n_unstable_modes': n_unstable,
        'dominant_eigenvalue': dominant_eigenval,
        'spectral_radius': eigen_results['spectral_radius']
    }


# ============================================================================
# PIPELINE FUNCTIONS FOR BATCH PROCESSING
# ============================================================================

def analyze_stability_and_margin(eigen_results, epsilon=0.02):
    """
    Step 2: Stability & Margin Analysis
    
    Implements the stability criteria from the verification pipeline:
    - Leading stability: λ_max ≤ -ε  
    - Imaginary parts: reject if any Im(λ) ≠ 0 with Re(λ) ≥ 0
    - Classify stability regime based on λ_max
    
    Parameters:
    -----------
    eigen_results : dict
        Results from eigen_analysis()
    epsilon : float
        Stability margin threshold (default: 0.02)
        
    Returns:
    --------
    dict : Stability analysis results for DataFrame storage
    """
    real_parts = eigen_results['real_parts']
    imag_parts = eigen_results['imag_parts']
    eigenvalues = eigen_results['eigenvalues']
    
    # Core stability metrics
    lambda_max = np.max(real_parts)
    min_real = np.min(real_parts)
    max_real = np.max(real_parts)
    
    # Fraction of complex eigenvalues
    n_complex = np.sum(np.abs(imag_parts) > 1e-10)
    fraction_complex = n_complex / len(eigenvalues)
    
    # Slowest stable timescale: τ = -1/min|Re(λ)| for Re(λ) < 0
    stable_real_parts = real_parts[real_parts < 0]
    if len(stable_real_parts) > 0:
        slowest_timescale = -1 / np.min(np.abs(stable_real_parts))
    else:
        slowest_timescale = np.inf  # No stable modes
    
    # Leading stability check: λ_max ≤ -ε
    passes_leading_stability = lambda_max <= -epsilon
    
    # Oscillatory instability check: reject if any Im(λ) ≠ 0 with Re(λ) ≥ 0
    unstable_complex_mask = (np.abs(imag_parts) > 1e-10) & (real_parts >= 0)
    has_oscillatory_instability = np.any(unstable_complex_mask)
    passes_oscillatory_check = not has_oscillatory_instability
    
    # Overall stability pass
    passes_stability = passes_leading_stability and passes_oscillatory_check
    
    # Stability regime classification
    if lambda_max <= -0.5:
        stability_regime = 'very_damped'
        regime_timescale_range = '≤ 2s'
    elif lambda_max <= -0.1:
        stability_regime = 'robust_informative'
        regime_timescale_range = '2-10s'
    elif lambda_max <= -0.05:
        stability_regime = 'responsive'
        regime_timescale_range = '10-20s'
    else:
        stability_regime = 'knife_edge'
        regime_timescale_range = '> 20s'
    
    return {
        'lambda_max': lambda_max,
        'min_real': min_real,
        'max_real': max_real,
        'fraction_complex': fraction_complex,
        'slowest_timescale': slowest_timescale,
        'passes_leading_stability': passes_leading_stability,
        'passes_oscillatory_check': passes_oscillatory_check,
        'passes_stability': passes_stability,
        'has_oscillatory_instability': has_oscillatory_instability,
        'stability_regime': stability_regime,
        'regime_timescale_range': regime_timescale_range,
        'n_complex_eigenvalues': n_complex,
        'epsilon_used': epsilon
    }


def run_homotopy_check(theta, sc, hmap, knob='G', perturbation_pct=10.0, epsilon=0.0):
    """
    Step 3: Homotopy (Knife-Edge Check) - CORRECTED VERSION
    
    Test stability under small parameter perturbations using original theta parameters.
    This avoids the parameter extraction problem by working directly with theta.
    
    Parameters:
    -----------
    theta : array-like
        Original parameter vector [w_EI_bias, w_EI_coeff, w_EE_bias, w_EE_coeff, G]
    sc : array
        Structural connectivity matrix
    hmap : array
        Heterogeneity map
    knob : str
        Parameter to perturb ('G', 'w_EE', 'w_EI')
    perturbation_pct : float
        Percentage perturbation (default: 10.0 for ±10%)
    epsilon : float
        Stability threshold (default: 0.0 for λ_max ≤ 0)
        
    Returns:
    --------
    dict : Homotopy check results for DataFrame storage
    """
    # Generate perturbations (skip 1.0 since we already tested original in Step 2)
    perturbations = [
        1.0 - perturbation_pct/100.0,  # e.g., 0.9 for -10%
        1.0 + perturbation_pct/100.0   # e.g., 1.1 for +10%
    ]
    
    lambda_max_values = []
    failed_at_perturbation = None
    computation_errors = []
    
    # Get original parameter value for reporting
    if knob == 'G':
        original_value = theta[4]
    elif knob == 'w_EE':
        original_value = f"({theta[2]:.4f}, {theta[3]:.4f})"
    elif knob == 'w_EI':
        original_value = f"({theta[0]:.4f}, {theta[1]:.4f})"
    else:
        raise ValueError(f"Unknown knob parameter: {knob}. Must be one of: 'G', 'w_EE', 'w_EI'")
    
    for perturbation in perturbations:
        try:
            # Create perturbed theta vector
            theta_perturbed = theta.copy()
            
            if knob == 'G':
                theta_perturbed[4] = theta[4] * perturbation
            elif knob == 'w_EE':
                theta_perturbed[2] = theta[2] * perturbation  # bias
                theta_perturbed[3] = theta[3] * perturbation  # coeff
            elif knob == 'w_EI':
                theta_perturbed[0] = theta[0] * perturbation  # bias
                theta_perturbed[1] = theta[1] * perturbation  # coeff
            
            # Create fresh model with perturbed parameters
            model_perturbed = setup_particle_model(theta_perturbed, sc, hmap)
            
            # Run stability analysis
            fixed_point = find_fixed_point(model_perturbed)
            jacobian = compute_jacobian(model_perturbed, include_bold=False)
            eigen_results = eigen_analysis(jacobian)
            lambda_max = np.max(eigen_results['real_parts'])
            
            lambda_max_values.append(lambda_max)
            
            # Check if this perturbation fails stability
            if lambda_max > epsilon and failed_at_perturbation is None:
                failed_at_perturbation = perturbation
                
        except Exception as e:
            # Handle numerical failures (e.g., convergence issues)
            lambda_max_values.append(np.inf)
            computation_errors.append(f"Perturbation {perturbation:.3f}: {str(e)}")
            if failed_at_perturbation is None:
                failed_at_perturbation = perturbation
    
    # Determine pass/fail
    passes_homotopy = all(np.isfinite(lmax) and lmax <= epsilon for lmax in lambda_max_values)
    
    return {
        'passes_homotopy': passes_homotopy,
        'lambda_max_perturbed': lambda_max_values,
        'perturbations_tested': perturbations,
        'perturbation_pct': perturbation_pct,
        'knob_parameter': knob,
        'original_knob_value': original_value,
        'failed_at_perturbation': failed_at_perturbation,
        'n_perturbations_tested': len(perturbations),
        'n_perturbations_failed': sum(1 for lmax in lambda_max_values if not np.isfinite(lmax) or lmax > epsilon),
        'has_computation_errors': len(computation_errors) > 0,
        'computation_errors': computation_errors if computation_errors else None
    }


def setup_particle_model(theta, sc, hmap):
    """
    Setup model for a single particle with given parameters.
    
    Parameters:
    -----------
    theta : array-like
        Parameter vector for this particle [w_EI_bias, w_EI_coeff, w_EE_bias, w_EE_coeff, G]
    sc : array
        Structural connectivity matrix
    hmap : array
        Heterogeneity map
        
    Returns:
    --------
    Bnm : Configured brain network model
    """
    # Create model instance
    model = Bnm(sc, gradient=hmap)
    
    # Set parameters from theta vector
    model.set('w_EI', (theta[0], theta[1]))
    model.set('w_EE', (theta[2], theta[3]))
    model.set('G', theta[4])
    
    return model


def analyze_single_particle(particle_idx, theta, sc, hmap, homotopy_knob='G', homotopy_perturbation_pct=10.0):
    """
    Run complete analysis pipeline for a single particle.
    
    Currently implements:
    - Step 1: Model setup and core computations
    - Step 2: Stability & margin analysis
    - Step 3: Homotopy (knife-edge check)
    
    Parameters:
    -----------
    particle_idx : int
        Index of the particle being analyzed
    theta : array-like
        Parameter vector for this particle
    sc : array
        Structural connectivity matrix
    hmap : array
        Heterogeneity map
    homotopy_knob : str
        Parameter knob for homotopy check ('G', 'w_EE', 'w_EI')
    homotopy_perturbation_pct : float
        Perturbation percentage for homotopy check (default: 10.0)
        
    Returns:
    --------
    dict : Analysis results for this particle
    """
    result = {'particle_idx': particle_idx}
    
    try:
        # Step 1: Model setup and core computations
        model = setup_particle_model(theta, sc, hmap)
        fixed_point = find_fixed_point(model)
        jacobian = compute_jacobian(model, include_bold=False)
        eigen_results = eigen_analysis(jacobian)
        
        # Store basic model info
        result['G'] = theta[4]
        result['w_EE_bias'] = theta[2]
        result['w_EE_coeff'] = theta[3]
        result['w_EI_bias'] = theta[0]
        result['w_EI_coeff'] = theta[1]
        result['n_regions'] = len(fixed_point['S_E'])
        result['setup_success'] = True
        
        # Step 2: Stability & margin analysis
        stability_results = analyze_stability_and_margin(eigen_results)
        result.update(stability_results)
        
        # Step 3: Homotopy check (only if Step 2 passes)
        if result['passes_stability']:
            homotopy_results = run_homotopy_check(
                theta, 
                sc, 
                hmap,
                knob=homotopy_knob, 
                perturbation_pct=homotopy_perturbation_pct
            )
            result.update(homotopy_results)
        else:
            # If Step 2 fails, skip Step 3 and fill with default values
            result.update({
                'passes_homotopy': False,
                'lambda_max_perturbed': None,
                'perturbations_tested': None,
                'perturbation_pct': homotopy_perturbation_pct,
                'knob_parameter': homotopy_knob,
                'original_knob_value': None,
                'failed_at_perturbation': None,
                'n_perturbations_tested': 0,
                'n_perturbations_failed': 0,
                'has_computation_errors': False,
                'computation_errors': None
            })
        
        # Determine which step failed (0 = passes all, 2 = fails step 2, 3 = fails step 3)
        if result['passes_stability'] and result['passes_homotopy']:
            result['step_failed_at'] = 0  # Passes all implemented steps
        elif not result['passes_stability']:
            result['step_failed_at'] = 2  # Fails step 2
        else:  # passes_stability but not passes_homotopy
            result['step_failed_at'] = 3  # Fails step 3
            
    except Exception as e:
        result['error'] = str(e)
        result['setup_success'] = False
        result['step_failed_at'] = -1  # Error in setup
        
        # Fill in NaN values for missing fields
        result.update({
            'lambda_max': np.nan,
            'min_real': np.nan,
            'max_real': np.nan,
            'fraction_complex': np.nan,
            'slowest_timescale': np.nan,
            'passes_stability': False,
            'stability_regime': 'error',
            'passes_homotopy': False,
            'lambda_max_perturbed': None,
            'perturbations_tested': None,
            'perturbation_pct': homotopy_perturbation_pct,
            'knob_parameter': homotopy_knob,
            'original_knob_value': None,
            'failed_at_perturbation': None,
            'n_perturbations_tested': 0,
            'n_perturbations_failed': 0,
            'has_computation_errors': False,
            'computation_errors': None
        })
    
    return result



def analyze_all_particles(theta_heterogeneous, sc, hmap, save_path=None, verbose=True, 
                         homotopy_knob='G', homotopy_perturbation_pct=10.0):
    """
    Run complete analysis pipeline on all particles.
    
    Parameters:
    -----------
    theta_heterogeneous : array
        Parameter matrix with shape (n_params, n_particles)
    sc : array
        Structural connectivity matrix
    hmap : array  
        Heterogeneity map
    save_path : str, optional
        Path to save results DataFrame (CSV format)
    verbose : bool
        Whether to print progress
    homotopy_knob : str
        Parameter knob for homotopy check ('G', 'w_EE', 'w_EI')
    homotopy_perturbation_pct : float
        Perturbation percentage for homotopy check (default: 10.0)
        
    Returns:
    --------
    pd.DataFrame : Analysis results with one row per particle
    """
    results = []
    n_particles = theta_heterogeneous.shape[1]
    
    if verbose:
        print(f"Analyzing {n_particles} particles...")
    
    for particle_idx in range(n_particles):
        if verbose and (particle_idx + 1) % 50 == 0:
            print(f"  Progress: {particle_idx + 1}/{n_particles}")
            
        # Run analysis for this particle
        particle_result = analyze_single_particle(
            particle_idx, theta_heterogeneous[:, particle_idx], sc, hmap,
            homotopy_knob=homotopy_knob, 
            homotopy_perturbation_pct=homotopy_perturbation_pct
        )
        results.append(particle_result)
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Save if requested
    if save_path:
        df.to_csv(save_path, index=False)
        if verbose:
            print(f"Results saved to {save_path}")
    
    if verbose:
        print(f"Analysis complete!")
        print(f"  Successful setups: {df['setup_success'].sum()}/{len(df)}")
        print(f"  Particles passing stability (Step 2): {df['passes_stability'].sum()}/{len(df)}")
        print(f"  Particles passing homotopy (Step 3): {df['passes_homotopy'].sum()}/{len(df)}")
        print(f"  Particles passing both steps: {(df['passes_stability'] & df['passes_homotopy']).sum()}/{len(df)}")
        print(f"  Stability regimes: {df['stability_regime'].value_counts().to_dict()}")
        print(f"  Step failure distribution: {df['step_failed_at'].value_counts().to_dict()}")
    
    return df
