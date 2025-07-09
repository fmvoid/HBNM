"""
Simulation Analysis Module for HBNM

This module provides tools for analyzing simulation outputs across multiple iterations
and creating comprehensive visualizations of optimization progress.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob
import h5py
from scipy import stats
from typing import Dict, List, Tuple, Optional, Union
import pandas as pd
from collections import defaultdict
import seaborn as sns

class SimulationAnalyzer:
    """
    A class for analyzing simulation outputs and creating visualizations.
    
    This class can handle multiple simulation types (homogeneous, heterogeneous, multimap)
    and automatically detects the number of iterations and parameters for each simulation.
    """
    
    def __init__(self, data_handler, verbose=True):
        """
        Initialize the SimulationAnalyzer.
        
        Parameters
        ----------
        data_handler : hbnm.io.Data
            Data handler instance from HBNM
        verbose : bool, optional
            Whether to print progress information
        """
        self.data = data_handler
        self.verbose = verbose
        self.simulation_data = {}
        self.parameter_info = {
            'baseline_homogeneous': {
                'n_params': 3,
                'param_names': ['w_EI', 'w_EE', 'G'],
                'description': 'Homogeneous model (no spatial heterogeneity)'
            },
            'baseline_heterogeneous': {
                'n_params': 5,
                'param_names': ['w_EI_bias', 'w_EI_slope', 'w_EE_bias', 'w_EE_slope', 'G'],
                'description': 'Single-map heterogeneous model (T1w/T2w)'
            },
            'receptor_nmda': {
                'n_params': 5,
                'param_names': ['w_EI_bias', 'w_EI_coeff', 'w_EE_bias', 'w_EE_coeff', 'G'],
                'description': 'Single-map heterogeneous model (NMDA_avg map)'
            },
            'receptor_dopamine_avg': {
                'n_params': 5,
                'param_names': ['w_EI_bias', 'w_EI_coeff', 'w_EE_bias', 'w_EE_coeff', 'G'],
                'description': 'Single-map heterogeneous model (dopamine_avg map)'
            },
            'receptor_serotonin_avg': {
                'n_params': 5,
                'param_names': ['w_EI_bias', 'w_EI_coeff', 'w_EE_bias', 'w_EE_coeff', 'G'],
                'description': 'Single-map heterogeneous model (serotonin_avg map)'
            },
            'receptor_norepinephrine_avg': {
                'n_params': 5,
                'param_names': ['w_EI_bias', 'w_EI_coeff', 'w_EE_bias', 'w_EE_coeff', 'G'],
                'description': 'Single-map heterogeneous model (norepinephrine_avg map)'
            },
            'receptor_nicotinic_avg': {
                'n_params': 5,
                'param_names': ['w_EI_bias', 'w_EI_coeff', 'w_EE_bias', 'w_EE_coeff', 'G'],
                'description': 'Single-map heterogeneous model (nicotinic_avg map)'
            },
            'receptor_muscarinic_avg': {
                'n_params': 5,
                'param_names': ['w_EI_bias', 'w_EI_coeff', 'w_EE_bias', 'w_EE_coeff', 'G'],
                'description': 'Single-map heterogeneous model (muscarinic_avg map)'
            },
            'receptor_gaba_a_avg': {
                'n_params': 5,
                'param_names': ['w_EI_bias', 'w_EI_coeff', 'w_EE_bias', 'w_EE_coeff', 'G'],
                'description': 'Single-map heterogeneous model (gaba_a_avg map)'
            }
        }
    
    def detect_simulations(self, output_dir: Optional[str] = None) -> List[str]:
        """
        Automatically detect available simulation directories.
        
        Parameters
        ----------
        output_dir : str, optional
            Output directory path. If None, uses data handler's output directory.
            
        Returns
        -------
        List[str]
            List of detected simulation names
        """
        if output_dir is None:
            output_dir = self.data.output_dir
            
        output_path = Path(output_dir)
        simulations = []
        
        for sim_dir in output_path.iterdir():
            if sim_dir.is_dir() and any(sim_dir.glob('iteration_*.hdf5')):
                simulations.append(sim_dir.name)
                
        if self.verbose:
            print(f"Detected {len(simulations)} simulations: {simulations}")
            
        return sorted(simulations)
    
    def detect_iterations(self, simulation: str) -> List[int]:
        """
        Detect available iteration numbers for a given simulation.
        
        Parameters
        ----------
        simulation : str
            Simulation name (e.g., 'baseline_homogeneous')
            
        Returns
        -------
        List[int]
            List of available iteration numbers
        """
        pattern = f"/{simulation}/iteration_*.hdf5"
        try:
            # Use the data handler to get file paths
            sim_dir = Path(self.data.output_dir) / simulation
            iteration_files = list(sim_dir.glob('iteration_*.hdf5'))
            
            iterations = []
            for file_path in iteration_files:
                # Extract iteration number from filename
                iteration_num = int(file_path.stem.split('_')[1])
                iterations.append(iteration_num)
                
            iterations.sort()
            
            if self.verbose:
                print(f"Simulation '{simulation}': found {len(iterations)} iterations ({min(iterations)}-{max(iterations)})")
                
            return iterations
            
        except Exception as e:
            if self.verbose:
                print(f"Error detecting iterations for {simulation}: {e}")
            return []
    
    def load_simulation_data(self, simulation: str, iterations: Optional[List[int]] = None) -> Dict:
        """
        Load all iteration data for a given simulation.
        
        Parameters
        ----------
        simulation : str
            Simulation name
        iterations : List[int], optional
            Specific iterations to load. If None, loads all available iterations.
            
        Returns
        -------
        Dict
            Dictionary containing loaded data organized by iteration
        """
        if iterations is None:
            iterations = self.detect_iterations(simulation)
            
        if not iterations:
            if self.verbose:
                print(f"No iterations found for simulation: {simulation}")
            return {}
        
        data = {
            'iterations': [],
            'theta': [],
            'distances': [],
            'weights': [],
            'epsilon': [],
            'ess': [],
            'n_accepted': [],
            'n_total': [],
            'acceptance_rate': [],
            'tau_squared': []
        }
        
        for iteration in iterations:
            try:
                file_path = f"/{simulation}/iteration_{iteration}.hdf5"
                fin = self.data.load(file_path, from_output=True)
                
                data['iterations'].append(iteration)
                data['theta'].append(fin['theta'][:])
                data['distances'].append(fin['distance'][:])
                data['weights'].append(fin['weights'][:])
                data['epsilon'].append(fin['epsilon'][()])
                data['ess'].append(fin['ess'][()])
                data['n_accepted'].append(fin['n_accepted'][()])
                data['n_total'].append(fin['n_total'][()])
                data['acceptance_rate'].append(fin['n_accepted'][()] / fin['n_total'][()])
                data['tau_squared'].append(fin['tau_squared'][:])
                
                fin.close()
                
            except Exception as e:
                if self.verbose:
                    print(f"Error loading iteration {iteration} for {simulation}: {e}")
                continue
        
        if self.verbose:
            print(f"Loaded {len(data['iterations'])} iterations for {simulation}")
            
        return data
    
    def load_all_simulations(self, simulations: Optional[List[str]] = None) -> Dict:
        """
        Load data for all specified simulations.
        
        Parameters
        ----------
        simulations : List[str], optional
            List of simulation names to load. If None, detects and loads all available.
            
        Returns
        -------
        Dict
            Dictionary with simulation names as keys and their data as values
        """
        if simulations is None:
            simulations = self.detect_simulations()
            
        self.simulation_data = {}
        
        for simulation in simulations:
            if self.verbose:
                print(f"\nLoading simulation: {simulation}")
            self.simulation_data[simulation] = self.load_simulation_data(simulation)
            
        return self.simulation_data
    
    def get_parameter_statistics(self, simulation: str) -> Dict:
        """
        Calculate parameter statistics across iterations for a simulation.
        
        Parameters
        ----------
        simulation : str
            Simulation name
            
        Returns
        -------
        Dict
            Dictionary containing parameter statistics
        """
        if simulation not in self.simulation_data:
            self.simulation_data[simulation] = self.load_simulation_data(simulation)
            
        data = self.simulation_data[simulation]
        if not data['iterations']:
            return {}
            
        # Get parameter info
        param_info = self.parameter_info.get(simulation, {})
        n_params = param_info.get('n_params', data['theta'][0].shape[0])
        param_names = param_info.get('param_names', [f'param_{i}' for i in range(n_params)])
        
        stats = {
            'param_names': param_names,
            'iterations': data['iterations'],
            'theta_means': [],
            'theta_stds': [],
            'theta_medians': [],
            'theta_q25': [],
            'theta_q75': []
        }
        
        for theta_array in data['theta']:
            # theta_array shape: (n_params, n_particles)
            stats['theta_means'].append(np.mean(theta_array, axis=1))
            stats['theta_stds'].append(np.std(theta_array, axis=1))
            stats['theta_medians'].append(np.median(theta_array, axis=1))
            stats['theta_q25'].append(np.percentile(theta_array, 25, axis=1))
            stats['theta_q75'].append(np.percentile(theta_array, 75, axis=1))
            
        # Convert to numpy arrays for easier handling
        for key in ['theta_means', 'theta_stds', 'theta_medians', 'theta_q25', 'theta_q75']:
            stats[key] = np.array(stats[key])  # Shape: (n_iterations, n_params)
            
        return stats
    
    def plot_simulation_progress(self, simulation: str, figsize: Tuple[int, int] = (15, 12)) -> plt.Figure:
        """
        Create comprehensive progress plots for a single simulation.
        
        Parameters
        ----------
        simulation : str
            Simulation name
        figsize : Tuple[int, int], optional
            Figure size (width, height)
            
        Returns
        -------
        plt.Figure
            The created figure
        """
        if simulation not in self.simulation_data:
            self.simulation_data[simulation] = self.load_simulation_data(simulation)
            
        data = self.simulation_data[simulation]
        param_stats = self.get_parameter_statistics(simulation)
        
        if not data['iterations']:
            print(f"No data available for simulation: {simulation}")
            return None
            
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle(f'Simulation Progress: {simulation}', fontsize=16, fontweight='bold')
        
        iterations = np.array(data['iterations'])
        
        # 1. Acceptance Rate
        ax = axes[0, 0]
        acceptance_rates = np.array(data['acceptance_rate']) * 100
        ax.plot(iterations, acceptance_rates, 'o-', linewidth=2, markersize=6)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Acceptance Rate (%)')
        ax.set_title('Acceptance Rate Over Iterations')
        ax.grid(True, alpha=0.3)
        
        # 2. Distance Statistics
        ax = axes[0, 1]
        distance_means = [np.mean(d) for d in data['distances']]
        distance_stds = [np.std(d) for d in data['distances']]
        distance_mins = [np.min(d) for d in data['distances']]
        
        ax.plot(iterations, distance_means, 'o-', label='Mean', linewidth=2)
        ax.fill_between(iterations, 
                       np.array(distance_means) - np.array(distance_stds),
                       np.array(distance_means) + np.array(distance_stds),
                       alpha=0.3)
        ax.plot(iterations, distance_mins, 's-', label='Best', alpha=0.7)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Distance')
        ax.set_title('Distance Statistics')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Epsilon (Tolerance)
        ax = axes[0, 2]
        epsilons = np.array(data['epsilon'])
        ax.plot(iterations, epsilons, 'o-', color='red', linewidth=2, markersize=6)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Epsilon (Tolerance)')
        ax.set_title('Tolerance Evolution')
        ax.grid(True, alpha=0.3)
        
        # 4. Effective Sample Size
        ax = axes[1, 0]
        ess_values = np.array(data['ess'])
        ax.plot(iterations, ess_values, 'o-', color='green', linewidth=2, markersize=6)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Effective Sample Size')
        ax.set_title('Effective Sample Size')
        ax.grid(True, alpha=0.3)
        
        # 5. Parameter Evolution (means with error bars)
        ax = axes[1, 1]
        if param_stats:
            n_params = len(param_stats['param_names'])
            colors = plt.cm.tab10(np.linspace(0, 1, n_params))
            
            for i, (param_name, color) in enumerate(zip(param_stats['param_names'], colors)):
                means = param_stats['theta_means'][:, i]
                stds = param_stats['theta_stds'][:, i]
                
                ax.errorbar(iterations, means, yerr=stds, 
                           label=param_name, color=color, 
                           marker='o', linewidth=2, capsize=4)
            
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Parameter Value')
            ax.set_title('Parameter Evolution (Mean ± Std)')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
        
        # 6. Weight Statistics
        ax = axes[1, 2]
        weight_means = [np.mean(w) for w in data['weights']]
        weight_stds = [np.std(w) for w in data['weights']]
        
        ax.plot(iterations, weight_means, 'o-', label='Mean Weight', linewidth=2)
        ax.fill_between(iterations,
                       np.array(weight_means) - np.array(weight_stds),
                       np.array(weight_means) + np.array(weight_stds),
                       alpha=0.3)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Particle Weight')
        ax.set_title('Particle Weights')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_parameter_evolution(self, simulation: str, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Create detailed parameter evolution plots with confidence intervals.
        
        Parameters
        ----------
        simulation : str
            Simulation name
        figsize : Tuple[int, int], optional
            Figure size
            
        Returns
        -------
        plt.Figure
            The created figure
        """
        param_stats = self.get_parameter_statistics(simulation)
        
        if not param_stats:
            print(f"No parameter statistics available for simulation: {simulation}")
            return None
            
        n_params = len(param_stats['param_names'])
        iterations = np.array(param_stats['iterations'])
        
        # Create subplots
        fig, axes = plt.subplots(n_params, 1, figsize=figsize, sharex=True)
        if n_params == 1:
            axes = [axes]
            
        fig.suptitle(f'Parameter Evolution: {simulation}', fontsize=14, fontweight='bold')
        
        colors = plt.cm.tab10(np.linspace(0, 1, n_params))
        
        for i, (param_name, color) in enumerate(zip(param_stats['param_names'], colors)):
            ax = axes[i]
            
            means = param_stats['theta_means'][:, i]
            stds = param_stats['theta_stds'][:, i]
            medians = param_stats['theta_medians'][:, i]
            q25 = param_stats['theta_q25'][:, i]
            q75 = param_stats['theta_q75'][:, i]
            
            # Plot median with IQR
            ax.plot(iterations, medians, 'o-', color=color, linewidth=2, 
                   label=f'{param_name} (median)', markersize=6)
            ax.fill_between(iterations, q25, q75, alpha=0.3, color=color, 
                           label='IQR (25%-75%)')
            
            # Plot mean with std
            ax.plot(iterations, means, 's--', color=color, alpha=0.7, 
                   label=f'{param_name} (mean)', markersize=4)
            
            ax.set_ylabel(param_name)
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            
        axes[-1].set_xlabel('Iteration')
        plt.tight_layout()
        return fig
    
    def plot_comparative_analysis(self, simulations: Optional[List[str]] = None, 
                                figsize: Tuple[int, int] = (16, 10)) -> plt.Figure:
        """
        Create comparative plots across multiple simulations.
        
        Parameters
        ----------
        simulations : List[str], optional
            List of simulations to compare. If None, uses all loaded simulations.
        figsize : Tuple[int, int], optional
            Figure size
            
        Returns
        -------
        plt.Figure
            The created figure
        """
        if simulations is None:
            simulations = list(self.simulation_data.keys())
            
        if not simulations:
            print("No simulations available for comparison")
            return None
            
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Comparative Analysis Across Simulations', fontsize=16, fontweight='bold')
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(simulations)))
        
        # 1. Acceptance Rates
        ax = axes[0, 0]
        for simulation, color in zip(simulations, colors):
            if simulation in self.simulation_data:
                data = self.simulation_data[simulation]
                iterations = np.array(data['iterations'])
                acceptance_rates = np.array(data['acceptance_rate']) * 100
                ax.plot(iterations, acceptance_rates, 'o-', color=color, 
                       label=simulation, linewidth=2, markersize=4)
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Acceptance Rate (%)')
        ax.set_title('Acceptance Rate Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Mean Distance Evolution with Standard Deviation
        ax = axes[0, 1]
        for simulation, color in zip(simulations, colors):
            if simulation in self.simulation_data:
                data = self.simulation_data[simulation]
                iterations = np.array(data['iterations'])
                distance_means = [np.mean(d) for d in data['distances']]
                distance_stds = [np.std(d) for d in data['distances']]
                
                ax.plot(iterations, distance_means, 'o-', color=color, 
                       label=simulation, linewidth=2, markersize=4)
                ax.fill_between(iterations, 
                               np.array(distance_means) - np.array(distance_stds),
                               np.array(distance_means) + np.array(distance_stds),
                               alpha=0.2, color=color)
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Distance')
        ax.set_title('Mean Distance Evolution (Mean ± Std)')
        ax.legend()
        ax.grid(True, alpha=0.3)
                
        # 3. Epsilon Evolution
        ax = axes[1, 0]
        for simulation, color in zip(simulations, colors):
            if simulation in self.simulation_data:
                data = self.simulation_data[simulation]
                iterations = np.array(data['iterations'])
                epsilons = np.array(data['epsilon'])
                ax.plot(iterations, epsilons, 'o-', color=color, 
                       label=simulation, linewidth=2, markersize=4)
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Epsilon (Tolerance)')
        ax.set_title('Tolerance Evolution Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Effective Sample Size
        ax = axes[1, 1]
        for simulation, color in zip(simulations, colors):
            if simulation in self.simulation_data:
                data = self.simulation_data[simulation]
                iterations = np.array(data['iterations'])
                ess_values = np.array(data['ess'])
                ax.plot(iterations, ess_values, 'o-', color=color, 
                       label=simulation, linewidth=2, markersize=4)
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Effective Sample Size')
        ax.set_title('Effective Sample Size Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_summary_report(self, simulations: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Create a summary report of all simulations.
        
        Parameters
        ----------
        simulations : List[str], optional
            List of simulations to include. If None, uses all loaded simulations.
            
        Returns
        -------
        pd.DataFrame
            Summary statistics for all simulations
        """
        if simulations is None:
            simulations = list(self.simulation_data.keys())
            
        summary_data = []
        
        for simulation in simulations:
            if simulation not in self.simulation_data:
                continue
                
            data = self.simulation_data[simulation]
            if not data['iterations']:
                continue
                
            # Calculate summary statistics
            final_acceptance = data['acceptance_rate'][-1] * 100
            final_epsilon = data['epsilon'][-1]
            final_ess = data['ess'][-1]
            best_distance = min([np.min(d) for d in data['distances']])
            final_distance = np.min(data['distances'][-1])
            n_iterations = len(data['iterations'])
            n_params = data['theta'][0].shape[0]
            
            summary_data.append({
                'Simulation': simulation,
                'Iterations': n_iterations,
                'Parameters': n_params,
                'Final_Acceptance_Rate': f"{final_acceptance:.1f}%",
                'Final_Epsilon': f"{final_epsilon:.4f}",
                'Final_ESS': f"{final_ess:.1f}",
                'Best_Distance': f"{best_distance:.4f}",
                'Final_Distance': f"{final_distance:.4f}",
                'Description': self.parameter_info.get(simulation, {}).get('description', 'Unknown')
            })
        
        return pd.DataFrame(summary_data)
    
    def save_all_plots(self, output_dir: str, simulations: Optional[List[str]] = None):
        """
        Save all plots to the specified directory.
        
        Parameters
        ----------
        output_dir : str
            Directory to save plots
        simulations : List[str], optional
            List of simulations to plot. If None, uses all loaded simulations.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if simulations is None:
            simulations = list(self.simulation_data.keys())
            
        # Individual simulation plots
        for simulation in simulations:
            if simulation in self.simulation_data:
                # Progress plot
                fig = self.plot_simulation_progress(simulation)
                if fig is not None:
                    fig.savefig(output_path / f'{simulation}_progress.png', dpi=300, bbox_inches='tight')
                    plt.close(fig)
                
                # Parameter evolution plot
                fig = self.plot_parameter_evolution(simulation)
                if fig is not None:
                    fig.savefig(output_path / f'{simulation}_parameters.png', dpi=300, bbox_inches='tight')
                    plt.close(fig)
        
        # Comparative plot
        if len(simulations) > 1:
            fig = self.plot_comparative_analysis(simulations)
            if fig is not None:
                fig.savefig(output_path / 'comparative_analysis.png', dpi=300, bbox_inches='tight')
                plt.close(fig)
        
        # Summary report
        summary_df = self.create_summary_report(simulations)
        summary_df.to_csv(output_path / 'simulation_summary.csv', index=False)
        
        if self.verbose:
            print(f"All plots and summary saved to: {output_path}") 