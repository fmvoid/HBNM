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
        
    def initialize(self, sc, fc=None, maps=None, n_particles=10, 
                   rejection_threshold=None, *args, **kwargs):
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
        n_particles : int
            Number of particles for PMC
        rejection_threshold : float
            Initial rejection threshold
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
        
        # Update priors now that we know the correct number of maps
        self.set_prior()
        
        # Initialize parent class (this calls set_prior)
        super().initialize(sc, fc=fc, maps=maps, 
                          n_particles=n_particles,
                          rejection_threshold=rejection_threshold,
                          *args, **kwargs)

    def get_appendices(self, run_id):
        """Required abstract method - no additional data to save"""
        return

    def set_prior(self):
        """
        Set prior distributions based on number of maps.
        
        Parameter order in theta:
        [w_EI_bias, w_EI_coeff_1, ..., w_EI_coeff_k,
         w_EE_bias, w_EE_coeff_1, ..., w_EE_coeff_k, 
         G]
        
        Total parameters: 2*(n_maps + 1) + 1
        """
        priors = []
        
        if self.n_maps == 0:
            # Homogeneous case - can use broader ranges
            priors.append(stats.uniform(0.001, 2.0))   # w_EI bias
            priors.append(stats.uniform(0.001, 5.0))   # w_EE bias
            priors.append(stats.uniform(0.001, 5.0))   # G
        else:
            # Multi-map case - use very tight ranges to prevent FIC failures
            # w_EI parameters: bias + coefficients
            priors.append(stats.uniform(0.12, 0.06))   # w_EI bias: 0.12 to 0.18
            for _ in range(self.n_maps):
                priors.append(stats.uniform(-0.02, 0.04))  # w_EI coefficients: -0.02 to +0.02
            
            # w_EE parameters: bias + coefficients  
            priors.append(stats.uniform(0.12, 0.06))   # w_EE bias: 0.12 to 0.18
            for _ in range(self.n_maps):
                priors.append(stats.uniform(-0.02, 0.04))  # w_EE coefficients: -0.02 to +0.02
            
            # Global coupling - reasonable range
            priors.append(stats.uniform(1.0, 2.0))     # G: 1.0 to 3.0
        
        self.prior = priors
        
        if self.verbose:
            print(f"Set up priors for {len(self.prior)} parameters:")
            print(f"  - w_EI: 1 bias + {self.n_maps} coefficients")
            print(f"  - w_EE: 1 bias + {self.n_maps} coefficients") 
            print(f"  - G: 1 parameter")

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
        """Calculate distance between model and empirical FC"""
        fit = pearsonr(self.fc_objective, fisher_z(synthetic_data))[0]
        penalty = (self.fc_objective.mean() - synthetic_data.mean()) ** 2
        distance = 1.0 - (fit - penalty)
        return distance


# Backwards compatible classes
class Homogeneous(MultiMapHeterogeneous):
    """Backwards compatible homogeneous optimization (0 maps)"""
    
    def initialize(self, sc, fc=None, n_particles=10, rejection_threshold=None, 
                   *args, **kwargs):
        super().initialize(sc, fc=fc, maps=None, n_particles=n_particles,
                          rejection_threshold=rejection_threshold, *args, **kwargs)


class Heterogeneous(MultiMapHeterogeneous):
    """Backwards compatible single-map optimization"""
    
    def initialize(self, sc, fc=None, gradient=None, n_particles=10, 
                   rejection_threshold=None, *args, **kwargs):
        # Convert gradient to maps format for compatibility
        maps = gradient[None, :] if gradient is not None else None
        super().initialize(sc, fc=fc, maps=maps, n_particles=n_particles,
                          rejection_threshold=rejection_threshold, *args, **kwargs)


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