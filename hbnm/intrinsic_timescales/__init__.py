# hbnm/intrinsic_timescales/__init__.py

"""
Intrinsic Timescales Analysis Package

This package provides tools for computing intrinsic timescales in brain network models
using stimulation-based methods as described in Deco et al. (2021).

Main Components:
---------------
- ModelLoader: Load and configure brain network models with Bayesian averaging
- StimulationSimulator: Run parallel stimulation experiments  
- DataProcessor: Process simulation results for curve fitting
- CurveFitter: Fit exponential decay curves to extract intrinsic timescales
- ResultsSaver: Save complete experiment results to HDF5 format
- Configuration system: Python-based experiment configuration
- Patterns module: Stimulation pattern generation utilities

Example Usage:
--------------
from hbnm.intrinsic_timescales import (
    load_config, ModelLoader, StimulationSimulator, DataProcessor, 
    CurveFitter, ResultsSaver
)

# Complete intrinsic timescales analysis workflow
config = load_config("nmda_avg")

# Load and configure model
loader = ModelLoader(config)
model_data = loader.load_model()

# Run stimulation experiment
simulator = StimulationSimulator(config)
raw_results = simulator.run_stimulation_experiment(model_data)

# Process data for curve fitting
processor = DataProcessor(config)
decay_data = processor.extract_decay_period(raw_results)

# Fit exponential decay curves to extract intrinsic timescales
fitter = CurveFitter(config)
timescale_results = fitter.fit_decay_curves(decay_data)

# Save complete experiment results
saver = ResultsSaver(config)
output_file = saver.save_experiment(
    raw_results, decay_data, timescale_results, "experiment_results.hdf5"
)
"""

# Main classes and functions
from .config import load_config
from .model_loader import ModelLoader
from .simulation import StimulationSimulator
from .data_processor import DataProcessor
from .curve_fitter import CurveFitter
from .results_saver import ResultsSaver

# Stimulation pattern utilities
from . import patterns

# Version information
__version__ = "0.1.0"
__author__ = "Frank Djimbouon"

# Make main components easily accessible
__all__ = [
    'load_config',
    'ModelLoader', 
    'StimulationSimulator',
    'DataProcessor',
    'CurveFitter',
    'ResultsSaver',
    'patterns'
]
