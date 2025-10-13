from hbnm.pmc import Pmc
from scipy import stats
from hbnm.model.utils import subdiag, linearize_map, normalize_sc, fisher_z
from scipy.stats import pearsonr
import numpy as np

class MultiMapHeterogeneous(Pmc):
    """
    PMC optimization class supporting arbitrary numbers of biological maps.
    Automatically adapts parameter dimensionality based on provided maps.
    """
    
    def __init__(self, input_directory, output_directory, verbose=True):
        """
        Initialize the optimization class.
        
        Parameters
        ----------
        input_directory : str
            Input directory for data
        output_directory : str  
            Output directory for results
        verbose : bool, optional
            Print diagnostics (default: True)
        """
        self.n_maps = 0  # Will be set during initialize()
        super().__init__(input_directory, output_directory, verbose)
        
    def initialize(self, sc, fc=None, maps=None, map_invert_flags=None, n_particles=10, 
                   rejection_threshold=None, custom_priors=None, *args, **kwargs):
        """
        Initialize the optimization with data and model.
        
        Parameters
        ----------
        sc : ndarray or list
            Structural connectivity matrix
        fc : ndarray
            Empirical functional connectivity  
        maps : ndarray, optional
            Biological maps matrix of shape (n_maps, n_regions)
        map_invert_flags : list of bool, optional
            For each biological map, whether to invert it (True) or use direct (False)
        n_particles : int
            Number of particles for PMC
        rejection_threshold : float
            Initial rejection threshold
        custom_priors : dict, optional
            Custom prior specifications. If None, uses default priors.
            Expected format: {
                'w_EI_bias': (min, max),
                'w_EI_coeff': (min, max),  # For heterogeneous models
                'w_EE_bias': (min, max), 
                'w_EE_coeff': (min, max),  # For heterogeneous models
                'G': (min, max)
            }
        """
        # Determine number of maps
        if maps is not None:
            maps = np.asarray(maps)
            if maps.ndim == 1:
                maps = maps[None, :]
            self.n_maps = maps.shape[0]
        else:
            self.n_maps = 0
            
        # Store for use in other methods
        self.maps = maps
        self.custom_priors = custom_priors
        
        # Update priors now that we know the correct number of maps
        self.set_prior()
        
        # Initialize parent class (this calls set_prior)
        super().initialize(sc, fc=fc, maps=maps, 
                          map_invert_flags=map_invert_flags,
                          n_particles=n_particles,
                          rejection_threshold=rejection_threshold,
                          *args, **kwargs)

    def get_appendices(self, run_id):
        """Required abstract method - no additional data to save"""
        return

    def set_prior(self):
        """
        Set prior distributions based on number of maps.
        
        If called before n_maps is set (from Pmc.__init__), this will set up
        a placeholder that gets overwritten when called properly from initialize().
        """
        # Check if we have valid map information yet
        if not hasattr(self, 'n_maps') or self.n_maps is None:
            # Called too early (from Pmc.__init__) - set up placeholder
            self.prior = [stats.uniform(0, 1)]  # Dummy prior, will be overwritten
            if self.verbose:
                print("Setting placeholder priors (will be updated during initialize)")
            return
        
        # Check if custom priors were provided
        if hasattr(self, 'custom_priors') and self.custom_priors is not None:
            priors = self._build_custom_priors(self.custom_priors)
            if self.verbose:
                print("Using custom priors specified by user")
        else:
            priors = self._build_default_priors()
            if self.verbose:
                print("Using default priors")
        
        self.prior = priors
        
        if self.verbose:
            print(f"Set up priors for {len(self.prior)} parameters:")
            print(f"  - w_EI: 1 bias + {self.n_maps} coefficients")
            print(f"  - w_EE: 1 bias + {self.n_maps} coefficients") 
            print(f"  - G: 1 parameter")
    
    def _build_default_priors(self):
        """Build default prior distributions"""
        priors = []
        if self.n_maps == 0:
            # Homogeneous case logic
            priors.append(stats.uniform(0.001, 5.0))   # w_EI bias
            priors.append(stats.uniform(0.001, 5.0))   # w_EE bias - This should be changed to 0.001, 15.0
            priors.append(stats.uniform(0.001, 5.0))   # G
        else:
            # Multi-map case logic
            # w_EI parameters: bias + coefficients
            priors.append(stats.uniform(0.001, 2.0))   # w_EI bias
            for _ in range(self.n_maps):
                priors.append(stats.uniform(0.0, 2.5))  # w_EI coefficients
            
            # w_EE parameters: bias + coefficients  
            priors.append(stats.uniform(0.001, 5.0))   # w_EE bias
            for _ in range(self.n_maps):
                priors.append(stats.uniform(0.0, 15.0))  # w_EE coefficients
            
            # Global coupling - reasonable range
            priors.append(stats.uniform(0.001, 5.0))     # G
        
        return priors
    
    def _build_custom_priors(self, custom_priors):
        """
        Build prior distributions from custom specifications.
        
        Parameters
        ----------
        custom_priors : dict
            Custom prior specifications with keys:
            - 'w_EI_bias': (min, max)
            - 'w_EI_coeff': (min, max) for heterogeneous models
            - 'w_EE_bias': (min, max)
            - 'w_EE_coeff': (min, max) for heterogeneous models  
            - 'G': (min, max)
        
        Returns
        -------
        list
            List of scipy.stats prior distributions
        """
        priors = []
        
        if self.n_maps == 0:
            # Homogeneous case - need w_EI_bias, w_EE_bias, G
            required_keys = ['w_EI_bias', 'w_EE_bias', 'G']
            for key in required_keys:
                if key not in custom_priors:
                    raise ValueError(f"Missing required prior specification for homogeneous model: '{key}'")
                
                min_val, max_val = custom_priors[key]
                if min_val >= max_val:
                    raise ValueError(f"Invalid prior range for {key}: min ({min_val}) >= max ({max_val})")
                
                priors.append(stats.uniform(min_val, max_val - min_val))
        else:
            # Heterogeneous case - need all coefficient types
            required_keys = ['w_EI_bias', 'w_EI_coeff', 'w_EE_bias', 'w_EE_coeff', 'G']
            for key in required_keys:
                if key not in custom_priors:
                    raise ValueError(f"Missing required prior specification for heterogeneous model: '{key}'")
            
            # w_EI bias
            min_val, max_val = custom_priors['w_EI_bias']
            priors.append(stats.uniform(min_val, max_val - min_val))
            
            # w_EI coefficients  
            min_val, max_val = custom_priors['w_EI_coeff']
            for _ in range(self.n_maps):
                priors.append(stats.uniform(min_val, max_val - min_val))
            
            # w_EE bias
            min_val, max_val = custom_priors['w_EE_bias']  
            priors.append(stats.uniform(min_val, max_val - min_val))
            
            # w_EE coefficients
            min_val, max_val = custom_priors['w_EE_coeff']
            for _ in range(self.n_maps):
                priors.append(stats.uniform(min_val, max_val - min_val))
            
            # Global coupling
            min_val, max_val = custom_priors['G']
            priors.append(stats.uniform(min_val, max_val - min_val))
        
        return priors

    def run_particle(self, theta):
        """
        Apply parameter vector to model.
        
        Parameters
        ----------
        theta : ndarray
            Parameter vector of length 2*(n_maps + 1) + 1
        """
        idx = 0
        
        # Extract w_EI parameters (bias + coefficients)
        w_EI_params = theta[idx:idx + self.n_maps + 1]
        idx += self.n_maps + 1
        
        # Extract w_EE parameters (bias + coefficients)
        w_EE_params = theta[idx:idx + self.n_maps + 1] 
        idx += self.n_maps + 1
        
        # Extract global coupling
        G = theta[idx]
        
        # Apply to model with error handling
        try:
            self.model.set('w_EI', tuple(w_EI_params))
            self.model.set('w_EE', tuple(w_EE_params))
            self.model.set('G', G)
        except Exception as e:
            if self.verbose:
                print(f"Parameter application failed: {e}")
            # Mark as unstable by setting extreme values
            self.model._unstable = True

    def generate_data(self):
        """Generate model functional connectivity"""
        self.model.moments_method(BOLD=True)
        return subdiag(self.model.get('corr_bold'))

    def distance_function(self, synthetic_data):
        """
        Calculate distance between model and empirical FC.
        
        Automatically detects Simple vs Hierarchical PMC based on fc_objective shape:
        - 1D (n_connections,): Simple PMC using pearsonr
        - 2D (n_connections, n_subjects): Hierarchical PMC using vcorrcoef
        
        Parameters
        ----------
        synthetic_data : ndarray
            Model FC upper diagonal (n_connections,)
            
        Returns
        -------
        float
            Distance metric (lower is better)
        """
        # Apply Fisher-z transform to model FC
        synthetic_data_z = fisher_z(synthetic_data)
        
        # Detect PMC mode based on fc_objective dimensionality
        if self.fc_objective.ndim == 2:
            # HIERARCHICAL PMC: Use vcorrcoef for subject-level correlations
            from hbnm.model.utils import vcorrcoef
            
            # vcorrcoef correlates each ROW of X with y (despite misleading docstring)
            # fc_objective has shape (n_connections, n_subjects)
            # Transpose to (n_subjects, n_connections) so each row = one subject
            # synthetic_data_z is Fisher-z transformed; fc_objective was transformed during loading
            fit = vcorrcoef(self.fc_objective.T, synthetic_data_z).mean()
            penalty = (self.fc_objective.mean() - synthetic_data_z.mean()) ** 2
            
        else:
            # SIMPLE PMC: Use pearsonr for group-average correlation
            fit = pearsonr(self.fc_objective, synthetic_data_z)[0]
            penalty = (self.fc_objective.mean() - synthetic_data_z.mean()) ** 2
        
        distance = 1.0 - (fit - penalty)
        return distance


# Backwards compatible classes
class Homogeneous(MultiMapHeterogeneous):
    """Backwards compatible homogeneous optimization (0 maps)"""
    
    def initialize(self, sc, fc=None, n_particles=10, rejection_threshold=None, 
                   custom_priors=None, *args, **kwargs):
        super().initialize(sc, fc=fc, maps=None, n_particles=n_particles,
                          rejection_threshold=rejection_threshold, 
                          custom_priors=custom_priors, *args, **kwargs)


class Heterogeneous(MultiMapHeterogeneous):
    """Backwards compatible single-map optimization"""
    
    def initialize(self, sc, fc=None, gradient=None, n_particles=10, 
                   rejection_threshold=None, custom_priors=None, *args, **kwargs):
        # Convert gradient to maps format for compatibility
        maps = gradient[None, :] if gradient is not None else None
        super().initialize(sc, fc=fc, maps=maps, n_particles=n_particles,
                          rejection_threshold=rejection_threshold, 
                          custom_priors=custom_priors, *args, **kwargs)


def load_data(data):
    """Load and preprocess data - unchanged from original"""
    fin = data.load('demirtas_neuron_2019.hdf5')
    sc = fin['sc'][:]
    fc = fin['fc'][:]
    t1t2 = fin['t1wt2w'][:]
    fin.close()

    # For left hemisphere, use first 180 indices
    sc = normalize_sc(sc[:180,:180])
    fc_obj = fc[:180,:180]
    hmap = linearize_map(t1t2[:180])

    return sc, hmap, fc_obj