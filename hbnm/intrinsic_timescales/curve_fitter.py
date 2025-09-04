# hbnm/intrinsic_timescales/curve_fitter.py

"""
Exponential Curve Fitting Module for Intrinsic Timescales Analysis

This module fits exponential decay curves to post-stimulation firing rate data
to extract intrinsic timescales from brain network simulations.

Mathematical model: r_E(t) = A * exp(-D * t) + B
Where:
- A: Amplitude/scaling parameter (Hz) 
- D: Decay rate parameter (1/s) - INVERSE of intrinsic timescale
- B: Baseline/offset parameter (Hz)
- τ = 1/D: Intrinsic timescale (seconds)
"""

import numpy as np
from scipy.optimize import curve_fit
import warnings


class CurveFitter:
    """
    Fits exponential decay curves to extract intrinsic timescales.
    
    Uses nonlinear least squares to fit the exponential decay model:
    r_E(t) = A * exp(-D * t) + B
    
    Provides comprehensive fitting with parameter bounds, uncertainty estimation,
    and quality metrics for each brain region.
    """
    
    def __init__(self, config):
        """Initialize with configuration dictionary."""
        self.config = config
        self.fitting_config = config['curve_fitting']
        
        # Extract parameter bounds and fitting options
        self.param_bounds = self.fitting_config['parameter_bounds']
        self.max_iterations = self.fitting_config['max_iterations']
        self.method = self.fitting_config['method']
        
        # Validate configuration
        self._validate_config()
    
    def fit_decay_curves(self, decay_data):
        """
        Fit exponential decay curves to all brain regions.
        
        Parameters:
        -----------
        decay_data : dict
            Clean output from DataProcessor.extract_decay_period():
            {
                'time_relative': np.ndarray,    # Time points (t=0 at stim end)
                'firing_rates': np.ndarray      # Shape: (n_regions, n_timepoints)
            }
            
        Returns:
        --------
        np.ndarray
            Fitted parameters and quality metrics for each region.
            Shape: (n_regions,) with structured dtype containing:
            - 'A': Amplitude parameter (Hz)
            - 'D': Decay rate parameter (1/s)  
            - 'B': Baseline parameter (Hz)
            - 'A_err': Amplitude uncertainty (Hz)
            - 'D_err': Decay rate uncertainty (1/s)
            - 'B_err': Baseline uncertainty (Hz)
            - 'timescale': Intrinsic timescale τ = 1/D (s)
            - 'timescale_err': Timescale uncertainty (s)
            - 'r_squared': Coefficient of determination
            - 'rmse': Root mean squared error (Hz)
            - 'success': Whether fit converged successfully
        """
        print("=== EXPONENTIAL CURVE FITTING ===")
        
        # Extract and validate input data
        time_points = decay_data['time_relative']
        firing_rates = decay_data['firing_rates']
        
        self._validate_input_data(time_points, firing_rates)
        
        n_regions, n_timepoints = firing_rates.shape
        print(f"Fitting {n_regions} brain regions with {n_timepoints} time points")
        print(f"Time range: {time_points[0]:.3f}s to {time_points[-1]:.3f}s")
        
        # Create output array with structured dtype
        results_dtype = [
            ('A', 'f8'), ('D', 'f8'), ('B', 'f8'),
            ('A_err', 'f8'), ('D_err', 'f8'), ('B_err', 'f8'),
            ('timescale', 'f8'), ('timescale_err', 'f8'),
            ('r_squared', 'f8'), ('rmse', 'f8'),
            ('success', 'bool')
        ]
        
        results = np.zeros(n_regions, dtype=results_dtype)
        
        # Fit each region individually
        print("Fitting exponential decay curves...")
        n_successful = 0
        n_failed = 0
        
        for region_idx in range(n_regions):
            region_data = firing_rates[region_idx, :]
            
            # Fit single region
            region_result = self._fit_single_region(
                time_points, region_data, region_idx
            )
            
            # Store results
            for field in results_dtype:
                field_name = field[0]
                results[region_idx][field_name] = region_result[field_name]
            
            # Track success/failure
            if region_result['success']:
                n_successful += 1
            else:
                n_failed += 1
            
            # Progress reporting for large datasets
            if n_regions > 50 and (region_idx + 1) % 50 == 0:
                print(f"  Progress: {region_idx + 1}/{n_regions} regions processed")
        
        # Final summary
        success_rate = 100 * n_successful / n_regions
        print(f"✓ Curve fitting completed!")
        print(f"  Successful fits: {n_successful}/{n_regions} ({success_rate:.1f}%)")
        
        if n_failed > 0:
            print(f"  Failed fits: {n_failed} regions")
            
        # Summary statistics for successful fits
        if n_successful > 0:
            successful_mask = results['success']
            successful_timescales = results[successful_mask]['timescale']
            
            print(f"  Timescale statistics:")
            print(f"    Mean: {np.mean(successful_timescales):.3f}s")
            print(f"    Std:  {np.std(successful_timescales):.3f}s") 
            print(f"    Range: {np.min(successful_timescales):.3f}s - {np.max(successful_timescales):.3f}s")
        
        return results
    
    def get_summary_statistics(self, fitting_results):
        """
        Compute summary statistics from fitting results.
        
        Parameters:
        -----------
        fitting_results : np.ndarray
            Output from fit_decay_curves()
            
        Returns:
        --------
        dict
            Summary statistics for successful fits
        """
        successful_mask = fitting_results['success']
        n_successful = np.sum(successful_mask)
        n_total = len(fitting_results)
        
        if n_successful == 0:
            return {
                'n_total': n_total,
                'n_successful': 0,
                'success_rate': 0.0
            }
        
        # Extract successful results
        successful = fitting_results[successful_mask]
        
        summary = {
            'n_total': n_total,
            'n_successful': n_successful,
            'success_rate': n_successful / n_total,
            'timescale_stats': {
                'mean': np.mean(successful['timescale']),
                'std': np.std(successful['timescale']),
                'min': np.min(successful['timescale']),
                'max': np.max(successful['timescale']),
                'median': np.median(successful['timescale'])
            },
            'quality_stats': {
                'mean_r_squared': np.mean(successful['r_squared']),
                'mean_rmse': np.mean(successful['rmse'])
            }
        }
        
        return summary
    
    # PRIVATE METHODS
    
    def _validate_config(self):
        """Validate curve fitting configuration."""
        required_keys = ['parameter_bounds', 'max_iterations', 'method']
        for key in required_keys:
            if key not in self.fitting_config:
                raise ValueError(f"Missing curve fitting config: '{key}'")
        
        # Validate parameter bounds
        required_params = ['A', 'D', 'B']
        for param in required_params:
            if param not in self.param_bounds:
                raise ValueError(f"Missing parameter bounds for '{param}'")
            
            bounds = self.param_bounds[param]
            if len(bounds) != 2 or bounds[0] >= bounds[1]:
                raise ValueError(f"Invalid bounds for parameter '{param}': {bounds}")
    
    def _validate_input_data(self, time_points, firing_rates):
        """Validate input data from DataProcessor."""
        # Check time points
        if not isinstance(time_points, np.ndarray) or time_points.ndim != 1:
            raise ValueError("time_points must be 1D numpy array")
        
        if not np.all(np.diff(time_points) > 0):
            raise ValueError("time_points must be monotonically increasing")
        
        if time_points[0] != 0.0:
            raise ValueError("time_points should start at 0.0 (relative to stimulation end)")
        
        # Check firing rates
        if not isinstance(firing_rates, np.ndarray) or firing_rates.ndim != 2:
            raise ValueError("firing_rates must be 2D numpy array")
        
        if firing_rates.shape[1] != len(time_points):
            raise ValueError("firing_rates time dimension must match time_points length")
        
        if np.any(np.isnan(firing_rates)) or np.any(np.isinf(firing_rates)):
            raise ValueError("firing_rates contains NaN or Inf values")
    
    def _fit_single_region(self, time_points, firing_rate_data, region_idx):
        """
        Fit exponential decay to a single brain region.
        
        Returns dict with all fitting results for this region.
        """
        # Initialize result structure
        result = {
            'A': np.nan, 'D': np.nan, 'B': np.nan,
            'A_err': np.nan, 'D_err': np.nan, 'B_err': np.nan,
            'timescale': np.nan, 'timescale_err': np.nan,
            'r_squared': np.nan, 'rmse': np.nan,
            'success': False
        }
        
        try:
            # Get initial parameter estimates
            A_init, D_init, B_init = self._estimate_initial_parameters(
                time_points, firing_rate_data
            )
            
            # Prepare parameter bounds for scipy
            bounds_lower = [
                self.param_bounds['A'][0],
                self.param_bounds['D'][0], 
                self.param_bounds['B'][0]
            ]
            bounds_upper = [
                self.param_bounds['A'][1],
                self.param_bounds['D'][1],
                self.param_bounds['B'][1]
            ]
            
            # Perform nonlinear least squares fitting
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                popt, pcov = curve_fit(
                    self._exponential_decay,
                    time_points,
                    firing_rate_data,
                    p0=[A_init, D_init, B_init],
                    bounds=(bounds_lower, bounds_upper),
                    max_nfev=self.max_iterations,
                    method=self.method
                )
            
            # Extract fitted parameters
            A_fit, D_fit, B_fit = popt
            
            # Calculate parameter uncertainties
            param_errors = np.sqrt(np.diag(pcov))
            A_err, D_err, B_err = param_errors
            
            # Compute intrinsic timescale and its uncertainty
            timescale = 1.0 / D_fit
            timescale_err = D_err / (D_fit ** 2)  # Error propagation: d(1/D)/dD = -1/D²
            
            # Calculate quality metrics
            y_pred = self._exponential_decay(time_points, A_fit, D_fit, B_fit)
            r_squared = self._calculate_r_squared(firing_rate_data, y_pred)
            rmse = np.sqrt(np.mean((firing_rate_data - y_pred) ** 2))
            
            # Store successful results
            result.update({
                'A': A_fit, 'D': D_fit, 'B': B_fit,
                'A_err': A_err, 'D_err': D_err, 'B_err': B_err,
                'timescale': timescale, 'timescale_err': timescale_err,
                'r_squared': r_squared, 'rmse': rmse,
                'success': True
            })
            
        except Exception as e:
            # Fitting failed - keep NaN values but don't raise exception
            # This allows the analysis to continue with other regions
            if region_idx is not None and region_idx < 10:  # Only warn for first few regions
                warnings.warn(f"Fitting failed for region {region_idx}: {str(e)}")
        
        return result
    
    def _exponential_decay(self, t, A, D, B):
        """
        Exponential decay function: r_E(t) = A * exp(-D * t) + B
        
        Parameters:
        -----------
        t : array_like
            Time points (seconds after stimulation end)
        A : float
            Amplitude parameter (Hz)
        D : float
            Decay rate parameter (1/s)
        B : float
            Baseline parameter (Hz)
        """
        return A * np.exp(-D * t) + B
    
    def _estimate_initial_parameters(self, time_points, firing_rate_data):
        """
        Estimate reasonable initial parameters for exponential fitting.
        
        Returns:
        --------
        tuple
            (A_init, D_init, B_init) initial parameter estimates
        """
        # B (baseline): estimate as final value
        B_init = firing_rate_data[-1]
        
        # A (amplitude): difference between initial and final values
        A_init = firing_rate_data[0] - B_init
        
        # D (decay rate): estimate from exponential fit to log-transformed data
        if A_init > 0:  # Only if there's actual decay
            try:
                # Use middle portion to avoid edge effects
                mid_start = len(firing_rate_data) // 4
                mid_end = 3 * len(firing_rate_data) // 4
                
                # Subtract baseline and take log
                r_normalized = firing_rate_data[mid_start:mid_end] - B_init
                r_normalized = np.maximum(r_normalized, 1e-6)  # Avoid log(0)
                log_r = np.log(r_normalized)
                
                # Linear fit: log(r) = log(A) - D*t
                slope, _ = np.polyfit(time_points[mid_start:mid_end], log_r, 1)
                D_init = -slope  # Negative slope becomes positive decay rate
                
                # Ensure reasonable bounds
                D_init = np.clip(D_init, 0.01, 10.0)
                
            except (ValueError, RuntimeWarning):
                D_init = 1.0  # Default fallback
        else:
            # No clear decay pattern
            D_init = 1.0
            A_init = 0.1  # Small positive amplitude
        
        # Ensure initial parameters are within bounds
        A_init = np.clip(A_init, self.param_bounds['A'][0], self.param_bounds['A'][1])
        D_init = np.clip(D_init, self.param_bounds['D'][0], self.param_bounds['D'][1])
        B_init = np.clip(B_init, self.param_bounds['B'][0], self.param_bounds['B'][1])
        
        return A_init, D_init, B_init
    
    def _calculate_r_squared(self, y_true, y_pred):
        """Calculate coefficient of determination (R²)."""
        ss_res = np.sum((y_true - y_pred) ** 2)  # Residual sum of squares
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)  # Total sum of squares
        
        if ss_tot == 0:
            return 0.0  # Perfect fit if no variance
        
        return 1.0 - (ss_res / ss_tot)
