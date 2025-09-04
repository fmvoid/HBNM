#!/usr/bin/python

""" Dynamic mean field model base class."""
from .utils import cov_to_corr
from .utils import load_model_params
from .hemo import Balloon
from .sim import Sim

from scipy.optimize import fsolve
from scipy.linalg import solve_lyapunov, eig
import scipy.sparse as spr
import numpy as np
import sympy as sym

class Model(object):

    """
    Class for the large-scale computational model with optional heterogeneous parametrization of w^{EE} and w^{EI}, 
    based on provided heterogeneity map.
    """

    def __init__(self, sc, g=0, norm_sc=True, hmap=None, maps=None,
                 wee=(0.15, 0.), wei=(0.15, 0.),
                 syn_params=None, bold_params='obata',
                 map_invert_flags=None, verbose=True):
        """
        
        Parameters
        ----------
        sc : ndarray
            Structural connectivity matrix
        g : float, optional
            Global coupling parameter to scale structural connectivity matrix
        norm_sc : bool, optional
            Normalize input strengths of the structural connectivity matrix (True by default)
        hmap : ndarray, optional
            DEPRECATED: Use 'maps' instead. Single heterogeneity map to scale local model parameters.
        maps : ndarray, optional
            Biological maps matrix of shape (n_maps, n_regions) or (n_regions,) for single map.
            If None, the model parameters are homogeneous (None by default)
        wee : tuple, optional
            Local recurrent excitatory connectivity weights (w^{EE}). 
            Can be:
            - Single value: homogeneous
            - 2-tuple (bias, slope): single-map heterogeneity (backwards compatible)
            - (n_maps+1)-tuple (bias, c1, c2, ...): multi-map heterogeneity
        wei : tuple, optional
            Local excitatory to inhibitory connectivity weights (w^{EI}). 
            Same format options as wee.
        syn_params : list, optional
            Synaptic dynamical model parameters (None by default)
        bold_params : str, optional
            Hemodynamic model parameters. 'obata' or 'friston' ('obata' by default)
        map_invert_flags : list of bool, optional
            For each biological map, whether to invert it (True) or use direct (False).
            If None, defaults to [True] for backwards compatibility. Length must match number of maps.
        verbose : bool, optional
            If True, prints diagnostics to console (True by default)  
        
        """

        # Structural connectivity / number of cortical areas
        self._SC = sc
        self._nc = sc.shape[0]

        # Global coupling
        self._G = g

        # Print diagnostics to console
        self._verbose = verbose

        # Initialize hemodynamic model (Balloon-Windkessel)
        self.hemo = Balloon(self._nc, parameters=bold_params)

        # Initialize simulation class
        self.sim = Sim()

        # If custom model parameters were provided, load model parameters with corresponding keys
        model_params = load_model_params()
        if syn_params is not None:
            for key in list(syn_params.keys()):
                model_params[key] = syn_params[key]

        # Unstable if Jacobian has eval > 0
        self._unstable = False

        # Initialize model outputs to None
        self._jacobian = None
        self._cov = None
        self._corr = None
        self._cov_bold = None
        self._corr_bold = None
        self._full_cov = None

        # Initialize state members to None
        self._I_E = None
        self._I_I = None
        self._S_E = None
        self._S_I = None
        self._r_E = None
        self._r_I = None

        # Various model parameters
        self._w_II = np.repeat(model_params['w_II'], self._nc)
        self._w_IE = np.repeat(model_params['w_IE'], self._nc)
        self._w_EE = np.repeat(model_params['w_EE'], self._nc)
        self._w_EI = np.repeat(model_params['w_EI'], self._nc)

        self._I0 = np.repeat(model_params['I0'], self._nc)
        self._J_NMDA = np.repeat(model_params['J_NMDA'], self._nc)
        self._sigma = model_params['sigma']
        self._gamma = model_params['gamma']
        self._W_I = model_params['W_I']
        self._W_E = model_params['W_E']
        self._tau_I = model_params['tau_I']
        self._tau_E = model_params['tau_E']
        self._d_I = model_params['d_I']
        self._d_E = model_params['d_E']
        self._b_I = model_params['b_I']
        self._b_E = model_params['b_E']
        self._a_I = model_params['a_I']
        self._a_E = model_params['a_E']
        self._I_ext = np.repeat(model_params['I_ext'], self._nc)

        self._gamma_I = 1.0

        self._tau_E_reset = np.copy(self._tau_E)
        self._tau_I_reset = np.copy(self._tau_I)
        self._gamma_reset = np.copy(self._gamma)
        self._gamma_I_reset = np.copy(self._gamma_I)

        # Baseline input currents
        self._I0_E = self._W_E * self._I0
        self._I0_I = self._W_I * self._I0

        # Steady state values for isolated node
        self._I_E_ss = np.repeat(model_params['I_E_ss'], self._nc)
        self._I_I_ss = np.repeat(model_params['I_I_ss'], self._nc)
        self._S_E_ss = np.repeat(model_params['S_E_ss'], self._nc)
        self._S_I_ss = np.repeat(model_params['S_I_ss'], self._nc)
        self._r_E_ss = np.repeat(model_params['r_E_ss'], self._nc)
        self._r_I_ss = np.repeat(model_params['r_I_ss'], self._nc)

        # Noise covariance matrix
        self._Q = np.identity(2 * self._nc) * self._sigma * self._sigma

        # Add lookup tables for transfer function and its derivatives
        self._phi()

        # Handle backwards compatibility for maps
        if maps is not None and hmap is not None:
            raise ValueError("Cannot specify both 'hmap' and 'maps'. Use 'maps' for new code.")
        
        if hmap is not None:
            # Convert single map to multi-map format
            self._maps = hmap[None, :] if hmap.ndim == 1 else hmap
        else:
            self._maps = maps
        
        # Ensure maps is always 2D: (n_maps, n_regions)
        if self._maps is not None:
            if self._maps.ndim == 1:
                self._maps = self._maps[None, :]  # Shape: (1, n_regions)
            
            # Set up inversion flags
            n_maps = self._maps.shape[0]
            if map_invert_flags is None:
                # Default to True for backwards compatibility (T1w/T2w behavior)
                self._map_invert_flags = [True] * n_maps
                if verbose:
                    print(f"Using default invert flags: {self._map_invert_flags}")
            else:
                if len(map_invert_flags) != n_maps:
                    raise ValueError(f"map_invert_flags length ({len(map_invert_flags)}) must match number of maps ({n_maps})")
                if not all(isinstance(flag, bool) for flag in map_invert_flags):
                    raise ValueError("All map_invert_flags must be boolean values")
                self._map_invert_flags = list(map_invert_flags)
                if verbose:
                    print(f"Using provided invert flags: {self._map_invert_flags}")
            
            # Apply maps to parameters if provided
            self._w_EE = self._apply_maps(wee)
            self._w_EI = self._apply_maps(wei)
        else:
            self._map_invert_flags = None

        # Set SC normalization
        self._sc_norm = 1.0
        if norm_sc:
            sc_norm = 1. / self._SC.sum(1)
            self._sc_norm = np.tile(sc_norm, (self._nc, 1)).T

        return

    def __repr__(self):
        return "dynamic mean field model class"

    def __str__(self):
        msg = ""
        msg += '%-17s %s' % ("\nUnstable:", self._unstable)
        msg += '\n%-16s %s' % ("Coupling (G):", self._G)
        msg += '\n%-16s %s' % ("N areas:", self._nc)
        return msg

    def set_jacobian(self, compute_fic=True):
        """ 
        Set Jacobian matrix given the model parameters. 
         
        Parameters
        ----------
        compute_fic : boolean, optional
            if True, local feedback inhibition parameters (w^{IE}) are adjusted to set the firing rates of
            excitatory populations to ~3Hz
        
        Returns
        -------
        boolean
            If True, the system is stable and linearized covariance can be successfully calculated, otherwise
            moments method fails because the solution is not stable for the given parameters.
        
        Notes
        -----
        This method should be executed before calculating the linearized covariance or performing numerical integration,
        each time the model parameters are modified.
        """

        eye = np.identity(self._nc)

        # Excitatory and inhibitory connection weights

        self._K_EE = (self._w_EE * eye) + (self._G * self._J_NMDA * self._sc_norm * self._SC)
        self._K_EI = (self._w_EI * eye)

        # Local feedback inhbition
        if compute_fic:
            self._w_IE = self._analytic_FIC()

        self._K_IE = -self._w_IE * eye
        self._K_II = -self._w_II * eye

        if np.any(self._w_IE < 0):
            self._unstable = True
            raise ValueError("Warning: FIC calculation led to negative J values!")
        else:
            # Derivatives of transfer function for each cell type
            # at steady state value of current
            dr_E = self.dphi_E(self._I_E_ss) * eye
            dr_I = self.dphi_I(self._I_I_ss) * eye

            # A_{mn} = dS_i^m/dS_j^n
            A_EE = (-1. / self._tau_E - (self._gamma * self._r_E_ss)) * eye + \
                   ((-self._gamma * (self._S_E_ss - 1.)) * eye).dot(dr_E.dot(self._K_EE))

            A_IE = ((self._gamma * (1. - self._S_E_ss)) * eye).dot(dr_E.dot(self._K_IE))
            A_EI = self._gamma_I * dr_I.dot(self._K_EI)
            A_II = (-1. / self._tau_I) * eye + self._gamma_I * dr_I.dot(self._K_II)

            # Stack blocks to form full Jacobian
            col1 = np.vstack((A_EE, A_EI))
            col2 = np.vstack((A_IE, A_II))
            self._jacobian = np.hstack((col1, col2))

            # Eigenvalues of Jacobian matrix
            self._evals, self._evects = eig(self._jacobian)
            self._max_eval = np.real(self._evals.max())

            # Check stability using eigenvalues
            self._unstable = self._max_eval >= 0.0

        return not self._unstable


    def moments_method(self, bold=False, use_lyapunov=False):
        """
        Computes the linearized covariance and the correlation matrices between model variables.
        
        Parameters
        ----------
        bold : boolean, optional
            if True, the covariance and correlation are computed for the extended hemodynamic system (BOLD),
            otherwise it computes only for the synaptic system of equations. (False by default)
        use_lyapunov : boolean, optional
            if True, the builtin function scipy.linalg.solve_lyapunov is used to solve Lyapunov equations.
            (False by default)(not recommended, only for debugging)
        
        Notes
        -----
        The covariance and correlation matrices can be called only after performing this method. 
        """

        if self._jacobian is None: self.set_jacobian()
        self._linearized_cov(use_lyapunov=use_lyapunov, bold=bold)
        self._reset_state()

        return

    def csd(self, freqs, pop='E'):
        """
        Computes cross-spectral density of synaptic variables.

        Parameters
        ----------
        freqs : ndarray or list
            An array or list containing the frequency bins for which the CSD will be computed
        pop : str, optional
            If 'E', the CSD of excitatory populations will return (default). if 'I', the CSD of inhibitory 
            populations will return. Any other string will be ignored and the full CSD will be returned 

        Returns
        -------
        csd : ndarray
            CSD of the synaptic system in shape NxNxf, where N is the number of regions and f is the number
            of frequency bins
        """
        if self._jacobian is None: self.set_jacobian()

        Id = np.identity(self._nc * 2)
        power = np.empty((2 * self._nc, 2 * self._nc, len(freqs)), dtype=complex)
        sig = complex(self._sigma ** 2)
        for i, f in enumerate(freqs):
            w = 2. * np.pi * f
            M1 = np.linalg.inv(self.jacobian + 1.j * w * Id)
            M2 = np.linalg.inv(self.jacobian.T - 1.j * w * Id)
            M3 = np.dot(M1, M2)
            power[:, :, i] = M3 * sig
        if pop == 'E':
            return power[:self._nc, :self._nc, :]
        elif pop == 'I':
            return power[self._nc:, self._nc:, :]
        else:
            return power

    def csd_bold(self, freqs):
        """Computes cross-spectral density of hemodynamic variables.

        Parameters
        ----------
        freqs : ndarray or list
            An array or list containing the frequency bins for which the CSD will be computed

        Returns
        -------
        ndarray
            CSD of the BOLD transformed hemodynamic system in shape NxNxf, where N is the number 
            of regions and f is the number of frequency bins.
            
        Notes
        -----
        The computation is performed for the hemodynamic system, the results are returned for
        BOLD signals. 
        """
        if self._jacobian_bold is None: self.moments_method(bold=True)
        N = self._jacobian_bold.shape[0]

        Id = np.identity(N)
        power = np.empty((self._nc, self._nc, len(freqs)), dtype=complex)
        sig = complex(self._sigma ** 2)
        for i, f in enumerate(freqs):
            w = 2. * np.pi * f
            M1 = np.linalg.inv(self._jacobian_bold + 1.j * w * Id)
            M2 = np.linalg.inv(self._jacobian_bold.T - 1.j * w * Id)
            M3 = np.dot(M1, M2)
            hemo_power = M3 * sig
            power[:,:,i] = (np.dot(np.dot((self.hemo.B), hemo_power), (self.hemo.B.conj().T)))
        return power

    # def integrate(self, t,
    #             dt=1e-4, n_save=10, stimulation=0.0,
    #             delays=False, distance=None, velocity=None,
    #             include_BOLD=True, from_fixed=True,
    #             sim_seed=None, save_mem=False):
    #     """Computes cross-spectral density of hemodynamic variables.

    #     Parameters
    #     ----------
    #     t : int
    #         Total simulation time in seconds.
    #     dt : float, optional
    #         Integration time step in seconds. By default dt is 0.1 msec.
    #     n_save : int, optional
    #         Sampling rate (time points). By default n_save is 10, therefore in dt is 0.1 msec,
    #         all the variables will be sampled at 1 msec.
    #     stimulation : ndarray or float, optional
    #         An array or matrix containing external currents if required.
    #         The size of the array should match the number time points (int(t / dt + 1))
    #         or it may be a scalar (0.0 by default).
    #     delays : bool, optional
    #         If True, delays are included during the integration (False by default).
    #     distance : ndarray, optional
    #         Distance matrix for delays (mm).
    #     velocity : float, optional
    #         Conduction velocity (m/sec).
    #     include_BOLD : boolean, optional
    #         If True, include hemodynamic model and BOLD signals (True by default).
    #     from_fixed : boolean, optional
    #         If True, start from steady‐state; else continue from last values.
    #     sim_seed : int, optional
    #         Seed for random number generator.
    #     save_mem : bool, optional
    #         If True, only keep minimal records for memory savings.

    #     Returns
    #     -------
    #     None

    #     Notes
    #     -----
    #     After simulation, excitatory synaptic variable is in self.sim.S_E
    #     and BOLD signals in self.sim.y (if include_BOLD=True).
    #     """

    #     # Seed RNG
    #     sim_seed = np.random.randint(0, 2**32 - 1) if sim_seed is None else sim_seed
    #     np.random.seed(sim_seed)

    #     # Prep
    #     if self._jacobian is None:
    #         self.set_jacobian()
    #     if from_fixed:
    #         self._reset_state()
    #         self.hemo.reset_state()

    #     # Time discretization
    #     dt_save    = dt * n_save
    #     n_sim_steps  = int(t / dt) + 1
    #     n_save_steps = int(t / dt_save) + 1

    #     # Preprocess stimulation into shape (n_sim_steps, nc)
    #     if np.isscalar(stimulation):
    #         stim_array = np.full((n_sim_steps, self.nc), stimulation)
    #     else:
    #         stim_array = np.array(stimulation)
    #         # 1d → broadcast to all nodes
    #         if stim_array.ndim == 1:
    #             if stim_array.size != n_sim_steps:
    #                 raise ValueError(
    #                     f"stimulation length {stim_array.size} != {n_sim_steps}")
    #             stim_array = np.tile(stim_array[:, None], (1, self.nc))
    #         elif stim_array.shape != (n_sim_steps, self.nc):
    #             raise ValueError(
    #                 f"stimulation shape {stim_array.shape} must be "
    #                 f"(n_sim_steps, nc)={ (n_sim_steps, self.nc) }")

    #     # Allocate storage
    #     if not save_mem:
    #         synaptic_state = np.zeros((6, self.nc, n_save_steps))
    #         synaptic_state[..., 0] = self.state

    #     if include_BOLD:
    #         if save_mem:
    #             hemo_state = np.zeros((self.nc, n_save_steps))
    #         else:
    #             hemo_state = np.zeros((5, self.nc, n_save_steps))
    #             hemo_state[:3, ..., 0] = 1.  # initial conditions

    #     if self._verbose:
    #         print("Beginning simulation.")

    #     # Main integration loop (no delays branch assumed)
    #     for i in range(1, n_sim_steps):
    #         # Inject external input for this time step
    #         self._I_ext = stim_array[i]

    #         # Step synaptic dynamics
    #         self._step(dt)

    #         # On save‐points, record
    #         if (i % n_save) == 0:
    #             i_save = i // n_save
    #             if not save_mem:
    #                 synaptic_state[..., i_save] = self.state

    #             if include_BOLD:
    #                 # hemodynamic update uses S_E deviation from SS
    #                 self.hemo.step(dt * 10., self._S_E - self._S_E_ss)
    #                 if save_mem:
    #                     hemo_state[:, i_save] = self.hemo._y
    #                 else:
    #                     hemo_state[..., i_save] = self.hemo.state

    #             if self._verbose and (i_save % 1000) == 0:
    #                 print(f"Saved step {i_save}/{n_save_steps}")

    #     if self._verbose:
    #         print("Simulation complete.")

    #     # Package results
    #     self.sim.t        = t
    #     self.sim.dt       = dt_save
    #     self.sim.n_save   = n_save
    #     self.sim.t_points = np.linspace(0, t, n_save_steps)
    #     self.sim.seed     = sim_seed

    #     if not save_mem:
    #         (self.sim.I_I, self.sim.I_E,
    #         self.sim.r_I, self.sim.r_E,
    #         self.sim.S_I, self.sim.S_E) = synaptic_state

    #     if include_BOLD:
    #         if save_mem:
    #             self.sim.y = hemo_state
    #         else:
    #             (self.sim.x, self.sim.f,
    #             self.sim.v, self.sim.q,
    #             self.sim.y) = hemo_state

    #     return



    # def integrate(self, t,
    #               dt=1e-4, n_save=10, stimulation=0.0,
    #               delays=False, distance=None, velocity=None,
    #               include_BOLD=True, from_fixed=True,
    #               sim_seed=None, save_mem=False):
    #     """Computes cross-spectral density of hemodynamic variables.

    #     Parameters
    #     ----------
    #     t : int
    #         Total simulation time in seconds.
    #     dt : float, optional
    #         Integration time step in seconds. By default dt is 0.1 msec.
    #     n_save : int, optional
    #         Sampling rate (time points). By default n_save is 10, therefore in dt is 0.1 msec, all the 
    #         variables will be sampled at 1 msec.
    #     stimulation : ndarray or float, optional
    #         An array or matrix containing external currents if required. The size of array should match
    #         to the number time points (i.e. int(t / dt + 1)) (0.0 by default)
    #     delays : bool, optional
    #         If True, delays are included during the integration (False by default)
    #     distance : ndarray, optional
    #         The distance matrix, If delays will be taken into account. The distance matrix should contain
    #         the euclidean or geodesic distance between regions in mm.
    #     velocity : float, optional
    #         The conduction velocity in m/sec, if conduction delays are not ignored.
    #     include_BOLD : boolean, optional
    #         If True, the simulation will also include hemodynamic model and BOLD signals (True by default)
    #     from_fixed : boolean, optional
    #         If True, the simulation will begin using steady state values of the parameters,
    #         otherwise the last available values will be used (i.e. from previous simulations...etc.)
    #     sim_seed : int, optional
    #         The seed for random number generator.
        
        
    #     Returns
    #     -------
    #     None
            
    #     Notes
    #     -----
    #         This method simulates the system for the given simulation time and the parameter values are stored.
    #         After successfull simulation, The excitatory synaptic variables can be obtained by .sim.S_E or 
    #         BOLD signals can be obtained by .sim.y
    #     """

    #     sim_seed = np.random.randint(0, 4294967295) if sim_seed is None else sim_seed
    #     np.random.seed(sim_seed)

    #     # Ensure Jacobian is computed before integration
    #     if self._jacobian is None: 
    #         self.set_jacobian()

    #     # Initialize to fixed point
    #     if from_fixed:
    #         self._reset_state()
    #         self.hemo.reset_state()

    #     # Simulation parameters
    #     dt_save = dt * n_save
    #     n_sim_steps = int(t / dt + 1)
    #     n_save_steps = int(t / dt_save + 1)

    #     # Synaptic state record
    #     if not save_mem:
    #         synaptic_state = np.zeros((6, self.nc, n_save_steps))
    #         synaptic_state[:, :, 0] = self.state
    #     # Synaptic state record
    #     #synaptic_state = np.zeros((6, self.nc, n_save_steps))
    #     #synaptic_state[:, :, 0] = self.state

    #     if self._verbose:
    #         print("Beginning simulation.")

    #     self.delays = delays
    #     if self.delays:
    #         if distance is None or velocity is None:
    #             self.delays = False
    #             msg = "Distance matrix and transmission velocity to implement delays"
    #             raise NotImplementedError(msg)
    #         else:
    #             self.distance = distance
    #             self.velocity = velocity
    #             self.steps_Delay = np.round(self.distance / (self.velocity * 1e-4 * 1e3)).astype(int)
    #             self._S_E_mem = np.tile(self._S_E, (self.steps_Delay.max() + 1, 1)).T
    #             self._S_E_vect = self._S_E_mem[list(range(self._nc)), self.steps_Delay]


    #     # Initialize BOLD variables if required
    #     if include_BOLD:
    #         if save_mem:
    #             # Hemodynamic state record
    #             hemo_state = np.zeros((self._nc, n_save_steps))
    #             #hemo_state[:3, :, 0] = 1.  # ICs
    #         else:
    #             # Hemodynamic state record
    #             hemo_state = np.zeros((5, self._nc, n_save_steps))
    #             hemo_state[:3, :, 0] = 1.  # ICs
    #         # Hemodynamic state record
    #         #hemo_state = np.zeros((5, self._nc, n_save_steps))
    #         #hemo_state[:3, :, 0] = 1.  # ICs

    #     # Main for-loop
    #     for i in range(1, n_sim_steps):
    #         if self.delays:
    #             self._S_E_mem[:, 0] = self._S_E
    #             self._S_E_vect = self._S_E_mem[list(range(self._nc)), self.steps_Delay]

    #         self._step(dt)

    #         if self.delays:
    #             self._S_E_mem[:, 1:] = self._S_E_mem[:, :-1]

    #         # Update state variables
    #         if not (i % n_save):
    #             i_save = i // n_save # integer division 
    #             if not save_mem:
    #                 synaptic_state[:, :, i_save] = self.state

    #             if include_BOLD:
    #                 self.hemo.step(dt*10., self._S_E - self._S_E_ss)
    #                 if save_mem:
    #                     hemo_state[:, i_save] = self.hemo._y
    #                 else:
    #                     hemo_state[:, :, i_save] = self.hemo.state
    #                 #hemo_state[:, :, i_save] = self.hemo.state

    #             if self._verbose:
    #                 if not (i_save % 1000):
    #                     print(i_save)

    #     if self._verbose:
    #         print("Simulation complete.")

    #     self.sim.t = t
    #     self.sim.dt = dt_save
    #     self.sim.n_save = n_save
    #     self.sim.t_points = np.linspace(0, t, n_save_steps)
    #     self.sim.seed = sim_seed

    #     if not save_mem:
    #         self.sim.I_I, self.sim.I_E, self.sim.r_I, self.sim.r_E, \
    #         self.sim.S_I, self.sim.S_E = synaptic_state

    #     if include_BOLD:
    #         if save_mem:
    #             self.sim.y = hemo_state
    #         else:
    #             self.sim.x, self.sim.f, self.sim.v, self.sim.q, self.sim.y = hemo_state

    #     #self.sim.I_I, self.sim.I_E, self.sim.r_I, self.sim.r_E, \
    #     #self.sim.S_I, self.sim.S_E = synaptic_state

    #     #if include_BOLD:
    #     #    self.sim.x, self.sim.f, self.sim.v, self.sim.q, self.sim.y = hemo_state

    #     return


    def integrate(self, t,
                dt=1e-4, n_save=10, stimulation=0.0,
                delays=False, distance=None, velocity=None,
                include_BOLD=True, from_fixed=True,
                sim_seed=None, save_mem=False):
        """Computes cross-spectral density of hemodynamic variables.

        Parameters
        ----------
        t : int
            Total simulation time in seconds.
        dt : float, optional
            Integration time step in seconds. By default dt is 0.1 msec.
        n_save : int, optional
            Sampling rate (time points). By default n_save is 10, therefore in dt is 0.1 msec,
            all the variables will be sampled at 1 msec.
        stimulation : ndarray or float, optional
            An array or matrix containing external currents if required.
            The size of the array should match the number of time points (i.e. int(t/dt)+1)
            or be a scalar (0.0 by default). If providing an array, it must have shape
            (n_sim_steps,) for time-only stimulation or (n_sim_steps, nc) for region-specific
            stimulation, where n_sim_steps = int(t/dt)+1 and nc is number of regions.
        delays : bool, optional
            If True, include conduction delays (False by default).
        distance : ndarray, optional
            Distance matrix (mm) for delays.
        velocity : float, optional
            Conduction velocity (m/sec).
        include_BOLD : bool, optional
            If True, include hemodynamic model and BOLD signals (True by default).
        from_fixed : bool, optional
            If True, start from fixed‐point initial state.
        sim_seed : int, optional
            RNG seed.
        save_mem : bool, optional
            If True, minimize memory usage when recording.

        Returns
        -------
        None
        """
        # 1) Seed RNG
        sim_seed = np.random.randint(0, 2**32 - 1) if sim_seed is None else sim_seed
        np.random.seed(sim_seed)

        # 2) Compute Jacobian & reset state if needed
        if self._jacobian is None:
            self.set_jacobian()
        if from_fixed:
            self._reset_state()
            self.hemo.reset_state()

        # 3) Time discretization
        dt_save     = dt * n_save
        n_sim_steps  = int(t / dt) + 1
        n_save_steps = int(t / dt_save) + 1

        # 4) Preprocess stimulation → array of shape (n_sim_steps, nc)
        if np.isscalar(stimulation):
            stim_array = np.full((n_sim_steps, self.nc), stimulation)
        else:
            stim_array = np.array(stimulation)
            # 1D time only → broadcast across regions
            if stim_array.ndim == 1:
                if stim_array.size != n_sim_steps:
                    raise ValueError(
                        f"stimulation length {stim_array.size} != {n_sim_steps}")
                stim_array = np.tile(stim_array[:, None], (1, self.nc))
            # full (time × regions) must match exactly
            elif stim_array.shape != (n_sim_steps, self.nc):
                raise ValueError(
                    f"stimulation shape {stim_array.shape} must be "
                    f"(n_sim_steps, nc)=({n_sim_steps}, {self.nc})"
                )

        # 5) Allocate storage
        if not save_mem:
            synaptic_state = np.zeros((6, self.nc, n_save_steps))
            synaptic_state[:, :, 0] = self.state

        if include_BOLD:
            if save_mem:
                hemo_state = np.zeros((self.nc, n_save_steps))
            else:
                hemo_state = np.zeros((5, self.nc, n_save_steps))
                hemo_state[:3, :, 0] = 1.  # initial conditions

        if self._verbose:
            print("Beginning simulation.")

        # 6) Set up delays if requested
        self.delays = delays
        if self.delays:
            if distance is None or velocity is None:
                msg = "Distance matrix and transmission velocity are required for delays"
                raise NotImplementedError(msg)
            self.distance    = distance
            self.velocity    = velocity
            # steps of delay in index units
            self.steps_Delay = np.round(
                self.distance / (self.velocity * 1e-4 * 1e3)
            ).astype(int)
            # initialize circular buffer for S_E
            self._S_E_mem  = np.tile(self._S_E, (self.steps_Delay.max() + 1, 1)).T
            self._S_E_vect = self._S_E_mem[
                list(range(self.nc)), self.steps_Delay
            ]

        # 7) Main integration loop
        for i in range(1, n_sim_steps):
            # 7a) update delays buffer
            if self.delays:
                # write newest S_E into buffer
                self._S_E_mem[:, 0] = self._S_E
                self._S_E_vect     = self._S_E_mem[
                    list(range(self.nc)), self.steps_Delay
                ]

            # 7b) inject stimulation for this step
            self._I_ext = stim_array[i]

            # 7c) step synaptic dynamics
            self._step(dt)

            # 7d) shift delay‐buffer
            if self.delays:
                self._S_E_mem[:, 1:] = self._S_E_mem[:, :-1]

            # 7e) record at save points
            if (i % n_save) == 0:
                i_save = i // n_save
                if not save_mem:
                    synaptic_state[:, :, i_save] = self.state

                if include_BOLD:
                    self.hemo.step(dt * 10., self._S_E - self._S_E_ss)
                    if save_mem:
                        hemo_state[:, i_save] = self.hemo._y
                    else:
                        hemo_state[:, :, i_save] = self.hemo.state

                if self._verbose and (i_save % 1000) == 0:
                    print(f"Saved step {i_save}/{n_save_steps}")

        if self._verbose:
            print("Simulation complete.")

        # 8) Pack results into self.sim
        self.sim.t        = t
        self.sim.dt       = dt_save
        self.sim.n_save   = n_save
        self.sim.t_points = np.linspace(0, t, n_save_steps)
        self.sim.seed     = sim_seed

        if not save_mem:
            (self.sim.I_I, self.sim.I_E,
            self.sim.r_I, self.sim.r_E,
            self.sim.S_I, self.sim.S_E) = synaptic_state

        if include_BOLD:
            if save_mem:
                self.sim.y = hemo_state
            else:
                (self.sim.x, self.sim.f,
                self.sim.v, self.sim.q,
                self.sim.y) = hemo_state

        return


    # Auxiliary Methods
    def _reset_state(self):
        """
        Reset state members to steady-state values.
        """
        self._I_I = np.copy(self._I_I_ss)
        self._I_E = np.copy(self._I_E_ss)
        self._r_I = np.copy(self._r_I_ss)
        self._r_E = np.copy(self._r_E_ss)
        self._S_I = np.copy(self._S_I_ss)
        self._S_E = np.copy(self._S_E_ss)

        return

    def _phi(self):
        """
        Generate transfer function and derivatives for Excitatory and Inhibitory populations.
        """
        IE = sym.symbols('IE')
        II = sym.symbols('II')
        phi_E = (self._a_E * IE - self._b_E) / (1. - sym.exp(-self._d_E * (self._a_E * IE - self._b_E)))
        phi_I = (self._a_I * II - self._b_I) / (1. - sym.exp(-self._d_I * (self._a_I * II - self._b_I)))
        dphi_E = sym.diff(phi_E, IE)
        dphi_I = sym.diff(phi_I, II)

        self.phi_E = sym.lambdify(IE, phi_E, "numpy")
        self.phi_I = sym.lambdify(II, phi_I,"numpy")
        self.dphi_E = sym.lambdify(IE, dphi_E, "numpy")
        self.dphi_I = sym.lambdify(II, dphi_I, "numpy")

        return

    def _inh_curr_fixed_pts(self, I):
        """ 
        Auxiliary function to find steady state inhibitory currents when FFI is enabled.

        Parameters
        ----------
        I : Inhibitory current

        Returns
        -------
        ndarray
            The fixed points for inhibitory currents 
        """
        return self._I0_I + self._K_EI.dot(self._S_E_ss) - \
               self._w_II * self._gamma_I * self._tau_I * self.phi_I(I) - I

    def _analytic_FIC(self):
        """ 
        Analytically solves for the strength of feedback inhibition for each cortical area. 

        Returns
        -------
        J : ndarray
            Local feedback inhibition providing excitatory firing rates ~3Hz
        """

        if self._SC is None:
            raise Exception("You must supply a connectivity matrix.")

        # Numerically solve for inhibitory currents first
        I_I_ss, infodict, ier, mesg = fsolve(self._inh_curr_fixed_pts,
                                             x0=self._I_I_ss, full_output=True)

        if ier:  # successful exit from fsolve
            self._I_I = np.copy(I_I_ss)  # needed for self._update_rate_I()
            self._I_I_ss = np.copy(I_I_ss)  # update stored steady state value
            self._r_I = self.phi_I(self._I_I)  # compute new steady state rate
            self._r_I_ss = np.copy(self._r_I)  # update stored steady state value
            self._S_I_ss = np.copy(self._r_I_ss) * self._tau_I * self._gamma_I # update stored val.
        else:
            err_msg = "Failed to find new steady-state currents." + \
                      " Cause of failure: %s" % mesg
            raise Exception(err_msg)

        # Solve for J using the steady state values (fixed points)
        J = (-1. / self._S_I_ss) * \
            (self._I_E_ss -
             self._I_ext - self._I0_E -
             self._K_EE.dot(self._S_E_ss))

        return J

    def _solve_lyapunov(self, jacobian, evals, L, Q, bold=False, builtin=False):
        """
        Solves Lyapunov equation
        
        Parameters
        ----------
        jacobian : ndarray
            The Jacobian of the system
        evals : ndarray
            Eigenvalues of the Jacobian matrix
        L : ndarray
            Eigenvectors of the Jacobian matrix
        Q : ndarray
            Input covariance matrix
        bold : boolean, optional
            If True, the Lyapunov equation is solved for hemodynamic system.
        builtin : boolean, optional
            If True, the builtin function will be used (not recommended)

        Notes
        -----
        This method requires the Jacobian matrix and the eigendecomposition of the Jacobian matrix
        as input for computational efficiency.

        Returns
        -------
        cov : ndarray
            The covariance matrix of the system around stable fix point
        """

        if builtin:
            return solve_lyapunov(jacobian, -Q)
        else:
            n = jacobian.shape[0] // self._nc
            evals_cc = np.conj(evals)
            Q = spr.csc_matrix(Q)

            L_inv = np.linalg.inv(L)

            inv_L_dagger = np.conj(L_inv).T

            QQ = Q.dot(inv_L_dagger)

            Q_tilde = L_inv.dot(QQ)

            denom_lambda_i = np.tile(evals.reshape((1, n * self._nc)).T,
                                     (1, n * self._nc))
            denom_lambda_conj_j = np.tile(evals_cc, (n * self._nc, 1))
            total_denom = denom_lambda_i + denom_lambda_conj_j
            M = -Q_tilde / total_denom

            if bold:
                B = spr.csc_matrix(self.hemo.B)
                X = B.dot(L)
                cov = np.dot(X.dot(M), X.conj().T).real
            else:
                L_dagger = np.conj(L).T
                cov = L.dot(M.dot(L_dagger)).real
            return cov

    def _linearized_cov(self, use_lyapunov=False, bold=False):
        """ 
        Solves for the linearized covariance matrix, using either
        the Lyapunov equation or eigen-decomposition.

        Parameters
        ----------
        use_lyapunov : boolean, optional
            If True, the builtin function will be used (not recommended)
        bold : boolean, optional
            If True, the Lyapunov equation is solved for hemodynamic system.
        """

        if self._unstable:
            if self._verbose: print("System unstable - no solution to Lyapunov equation - exiting")
            self._cov, self._cov_bold, self._corr_bold, self._corr = None, None, None, None
            return
        else:
            if bold:
                self.hemo.linearize_BOLD(self._S_E_ss, self._jacobian, self._Q)
                self._jacobian_bold = self.hemo.full_A
                self._Q_bold = self.hemo.full_Q

                evals, evects = eig(self._jacobian_bold)
                self.evals_bold = evals
                self.evects_bold = evects

                self._cov_bold = self._solve_lyapunov(self._jacobian_bold, evals, evects, self._Q_bold, bold=True,
                                                      builtin=use_lyapunov)
                self._corr_bold = cov_to_corr(self._cov_bold, full_matrix=False)
            else:
                self._cov = self._solve_lyapunov(self._jacobian, self._evals, self._evects, self._Q,
                                                 builtin=use_lyapunov)
                self._corr = cov_to_corr(self._cov)


    def _exc_current(self):
        """
        Excitatory current for each cortical region.
        
        
        Parameters
        ----------
        I_ext : float
            External stimulation at each time step
        
        Returns
        -------
        ndarray
            Excitatory currents at each time step 
        """
        if self.delays:
            return self._I0_E + self._I_ext + (self._K_EE * self._S_E_vect).sum(1) + self._K_IE.dot(self._S_I)
        else:
            return self._I0_E + self._I_ext + self._K_EE.dot(self._S_E) + self._K_IE.dot(self._S_I)

    def _inh_current(self):
        """ 
        Inhibitory current for each cortical region
        
        Returns
        -------
        ndarray
            Inhibitory currents at each time step
        """
        return self._I0_I + self._K_EI.dot(self._S_E) + self._K_II.dot(self._S_I)

    def _step(self, dt):
        """ 
        Advance system synaptic state by time evolving for time dt. 
        
        Parameters
        ----------
        dt : float
            Integration time step in seconds. By default dt is 0.1 msec.
        I_ext : float
            External stimulation at time step
        """
        self._I_E = self._exc_current()
        self._I_I = self._inh_current()

        self._r_E = self.phi_E(self._I_E)
        self._r_I = self.phi_I(self._I_I)

        # Compute change in synaptic gating variables
        dS_E = self._dSEdt() * dt + np.sqrt(dt) * self._sigma * \
               np.random.normal(size=self._nc)

        dS_I = self._dSIdt() * dt + np.sqrt(dt) * self._sigma * \
               np.random.normal(size=self._nc)

        # Update class members S_E, S_I
        self._S_E += dS_E
        self._S_I += dS_I

        # Clip synaptic gating fractions
        self._S_E = np.clip(self._S_E, 0., 1.)
        self._S_I = np.clip(self._S_I, 0., 1.)

        return

    def _dSEdt(self):
        """
        Returns time derivative of excitatory synaptic gating variables in absense of noise.
        
        Returns
        -------
        ndarray
            Derivatives of excitatory synaptic gating variables
        """
        return -(self._S_E / self._tau_E) + (self._gamma * self._r_E) * (1. - self._S_E)

    def _dSIdt(self):
        """Returns time derivative of inhibitory synaptic gating variables in absense of noise.
        
        Returns
        -------
        ndarray
            Derivatives of excitatory synaptic gating variables
        """
        return -(self._S_I / self._tau_I) + self._gamma_I * self._r_I

    def _apply_maps(self, params):
        """
        Apply biological maps to generate region-wise parameter values.
        
        Parameters
        ----------
        params : scalar, array-like, or tuple
            Parameter specification:
            - scalar: homogeneous value across all regions
            - array of length n_regions: explicit per-region values
            - tuple of length 2: (bias, slope) for single-map heterogeneity
            - tuple of length n_maps+1: (bias, c1, c2, ...) for multi-map
        
        Returns
        -------
        ndarray
            Parameter values for each region, shape (n_regions,)
        
        Examples
        --------
        >>> # Homogeneous
        >>> model._apply_maps(0.15)  # -> [0.15, 0.15, ..., 0.15]
        
        >>> # Single-map (backwards compatible)
        >>> model._apply_maps((0.15, 0.05))  # -> bias + slope * maps[0]
        
        >>> # Multi-map
        >>> model._apply_maps((0.15, 0.02, -0.01, 0.03))  # -> bias + sum(coeffs * maps)
        """
        # Handle scalar input
        if np.isscalar(params):
            return np.full(self._nc, params)
        
        # Convert to array for easier handling
        params = np.asarray(params)
        
        # Handle homogeneous case when no maps provided
        if self._maps is None:
            if params.size == 1:
                return np.full(self._nc, params[0])
            elif params.size == self._nc:
                return params  # Explicit per-region values
            else:
                raise ValueError(f"No biological maps provided, but received {params.size} parameters. "
                               f"Expected 1 (homogeneous) or {self._nc} (per-region).")
        
        # Handle map-based heterogeneity (when maps ARE provided)
        n_maps = self._maps.shape[0]
        bias = params[0]
        
        if params.size == 1:
            # Just bias, no heterogeneity
            return np.full(self._nc, bias)
        elif params.size == 2 and n_maps >= 1:
            # Backwards compatible: (bias, slope) uses first map with inversion flags
            slope = params[1]
            
            # Use inversion flag if available, otherwise default to True for backwards compatibility
            should_invert = getattr(self, '_map_invert_flags', [True])[0]
            
            # Validate map for normalization
            map_range = np.ptp(self._maps[0])
            if map_range == 0:
                raise ValueError("Map is constant (range=0) - cannot normalize for heterogeneity")
            if np.any(np.isnan(self._maps[0])):
                raise ValueError("Map contains NaN values")
            
            # Always normalize to [0, 1] first
            map_normalized = (self._maps[0] - np.min(self._maps[0])) / map_range
            
            # Apply inversion if requested
            if should_invert:
                map_normalized = 1.0 - map_normalized
                
            # Apply coefficient (users can specify negative coefficients if desired)
            return bias + slope * map_normalized
            
        elif params.size == n_maps + 1:
            # Multi-map: bias + sum(coefficients * maps with individual inversion)
            coeffs = params[1:]
            result = np.full(self._nc, bias)
            
            # Use inversion flags if available, otherwise default to True for all maps
            invert_flags = getattr(self, '_map_invert_flags', [True] * n_maps)
            
            # Validate invert flags
            if len(invert_flags) != n_maps:
                raise ValueError(f"Invert flags length ({len(invert_flags)}) doesn't match number of maps ({n_maps})")
            
            for i, (coeff, map_vals, should_invert) in enumerate(zip(coeffs, self._maps, invert_flags)):
                # Validate map for normalization
                map_range = np.ptp(map_vals)
                if map_range == 0:
                    raise ValueError(f"Map {i} is constant (range=0) - cannot normalize")
                if np.any(np.isnan(map_vals)):
                    raise ValueError(f"Map {i} contains NaN values")
                    
                # Always normalize to [0, 1] first
                map_normalized = (map_vals - np.min(map_vals)) / map_range
                
                # Apply inversion if requested
                if should_invert:
                    map_normalized = 1.0 - map_normalized
                    
                # Add contribution
                result += coeff * map_normalized
                
            return result
            
        # REMOVED THE OPTION TO SET PARAMETERS PER REGION -- MAY CHANGE THAT LATER
        # elif params.size == self._nc:
        #     # Explicit per-region values (only as fallback when maps present)
        #     return params
        else:
            raise ValueError(f"Parameter size mismatch. Got {params.size} parameters, "
                            f"expected 1 (homogeneous), 2 (single-map), or {n_maps + 1} (multi-map).")
    
    # Properties
    @property
    def Q(self):
        """
        Returns
        -------
        ndarray
            Input covariance matrix
        """
        return self._Q

    @property
    def cov(self, full=False):
        """
        Parameters
        ----------
        full : bool, optional
            If True, returns the full covariance matrix.
        
        Returns
        -------
        ndarray
            Covariance matrix of linearized fluctuations about fixed point.
        """
        if full:
            return self._cov
        else:
            return self._cov[:self._nc, :self._nc]

    @property
    def cov_bold(self):
        """
        Returns
        -------
        ndarray
            Covariance matrix of linearized fluctuations for BOLD
        """
        return self._cov_bold

    @property
    def var(self, full=False):
        """
        Parameters
        ----------
        full : bool, optional
            If True, returns the full variance.

        Returns
        -------
        ndarray
            Variances of linearized fluctuations about fixed point.
        """
        if full:
            return np.diag(self._cov)
        else:
            return np.diag(self._cov[:self._nc, :self._nc])

    @property
    def var_bold(self):
        """
        Returns
        -------
        ndarray
            Variances of linearized fluctuations for BOLD.
        """
        return np.diag(self._cov_bold)

    @property
    def corr(self):
        """
        Returns
        -------
        ndarray
            Correlation matrix (model FC) for synaptic system 
        """
        return self._corr

    @property
    def corr_bold(self):
        """
        Returns
        -------
        ndarray
            Correlation matrix (model FC) for BOLD 
        """
        return self._corr_bold

    @property
    def jacobian(self):
        """
        Returns
        -------
        ndarray
            Jacobian of linearized fluctuations about fixed point. 
        """
        return self._jacobian

    @property
    def evals(self):
        """
        Returns
        -------
        ndarray
            Eigenvalues of Jacobian matrix. 
        """
        return eig(self._jacobian)[0] if self.jacobian is not None else None

    @property
    def evecs(self):
        """
        Returns
        -------
        ndarray
            Left Eigenvextors of Jacobian matrix. 
        """
        return eig(self._jacobian)[1] if self.jacobian is not None else None

    @property
    def nc(self):
        """
        Returns
        -------
        ndarray
            Number of cortical areas. 
        """
        return self._nc

    @property
    def SC(self):
        """
        Returns
        -------
        ndarray
            Empirical structural connectivity. 
        """
        return self._SC

    @property
    def sigma(self):
        """
        Returns
        -------
        ndarray
            Input noise to each area 
        """
        return self._sigma

    @sigma.setter
    def sigma(self, sigma):
        """
        Parameters
        -----------
        sigma : ndarray or float
            Input noise to each area
        """
        self._sigma = sigma
        self._Q = np.identity(2 * self._nc) * self._sigma * self._sigma

    @SC.setter
    def SC(self, SC):
        """
        Parameters
        -----------
        SC : ndarray
            Empirical structural connectivity
        """
        assert(SC.shape[0] == self._nc)
        self._SC = SC
        return

    @property
    def state(self):
        """
        Returns
        -------
        ndarray
            All state variables, 6 rows by len(nodes) columns.
            Rows are I_I, I_E, r_I, r_E, S_I, S_E.  
        """
        return np.vstack((self._I_I, self._I_E, self._r_I,
                          self._r_E, self._S_I, self._S_E))

    @property
    def steady_state(self):
        """
        Returns
        -------
        ndarray
            All steady state variables, shape 6 x nc.
            Rows are, respectively, I_I, I_E, r_I, r_E, S_I, S_E.  
        """
        return np.vstack((self._I_I_ss, self._I_E_ss, self._r_I_ss,
                          self._r_E_ss, self._S_I_ss, self._S_E_ss))

    @property
    def w_EE(self):
        """
        Returns
        -------
        ndarray
            Local recurrenm excitatory strengths.  
        """
        return self._w_EE

    @w_EE.setter
    def w_EE(self, w):
        """
        Parameters
        -------
        w : scalar, array-like, or tuple
            Local recurrent excitatory strengths.
            - scalar: homogeneous value
            - array of length n_regions: explicit per-region values  
            - tuple of length 2: (bias, slope) for single-map heterogeneity
            - tuple of length n_maps+1: (bias, c1, c2, ...) for multi-map
        """
        self._w_EE = self._apply_maps(w)

    @property
    def w_EI(self):
        """
        Returns
        -------
        ndarray
            Local excitatory to inhibitory strengths.  
        """
        return self._w_EI

    @w_EI.setter
    def w_EI(self, w):
        """
        Parameters
        -------
        w : scalar, array-like, or tuple
            Local excitatory to inhibitory strengths.
            - scalar: homogeneous value
            - array of length n_regions: explicit per-region values
            - tuple of length 2: (bias, slope) for single-map heterogeneity  
            - tuple of length n_maps+1: (bias, c1, c2, ...) for multi-map
        """
        self._w_EI = self._apply_maps(w)

    @property
    def G(self):
        """
        Returns
        -------
        ndarray
            Global coupling strength.  
        """
        return self._G

    @G.setter
    def G(self, g):
        """
        Parameters
        ----------
        g : ndarray
            Global coupling strength.  
        """
        self._G = g
        return

    @property
    def w_IE(self):
        """
        Returns
        -------
        ndarray
            Feedback inhibition weights.  
        """
        return self._w_IE

    @w_IE.setter
    def w_IE(self, J):
        """
        Parameters
        ----------
        J : ndarray
            Feedback inhibition weights.  
        """
        self._w_IE = J
        return

    @property
    def hmap(self):
        """
        Returns
        -------
        ndarray
            Heterogeneity map values
        """
        return self._hmap

    @hmap.setter
    def hmap(self, h):
        """
        Parameters
        ----------
        h : ndarray
            Heterogeneity map values
            
        Notes
        -----
        The heterogeneity map values should be normalized between 0 and 1
        """
        self._hmap = h

    @property
    def I_ext(self):
        """
        Returns
        -------
        ndarray
            External current  
        """
        return self._I_ext

    @I_ext.setter
    def I_ext(self, I):
        """
        Parameters
        ----------
        I : ndarray
            External current  
        """
        self._I_ext = I

    @property
    def J_NMDA(self):
        """
        Returns
        -------
        float
            Effective NMDA conductance  
        """
        return self._J_NMDA

    @J_NMDA.setter
    def J_NMDA(self, J):
        """
        Parameters
        ----------
        float
            Effective NMDA conductance  
        """
        self._J_NMDA = J

    @property
    def maps(self):
        """
        Returns
        -------
        ndarray or None
            Biological maps matrix of shape (n_maps, n_regions), or None if homogeneous
        """
        return self._maps

    @maps.setter
    def maps(self, m):
        """
        Parameters
        ----------
        m : ndarray or None
            Biological maps matrix of shape (n_maps, n_regions) or (n_regions,) for single map
            
        Notes
        -----
        Setting new maps will NOT automatically update existing w_EE/w_EI values.
        You must explicitly reset those parameters after changing maps.
        """
        if m is not None:
            m = np.asarray(m)
            if m.ndim == 1:
                m = m[None, :]
            if m.shape[1] != self._nc:
                raise ValueError(f"Maps must have {self._nc} regions, got {m.shape[1]}")
        self._maps = m
