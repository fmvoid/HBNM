"""
HBNM: Hierarchical Brain Network Model

A Python package for large-scale brain network modeling with biological heterogeneity.
"""

from .bnm import Bnm
from .io import Data
from .analysis import SimulationAnalyzer

__version__ = "1.0.0"
__all__ = ["Bnm", "Data", "SimulationAnalyzer"]
