# hbnm/intrinsic_timescales/configs/__init__.py

"""
Configuration files for intrinsic timescales experiments.

Available configurations:
- base: Default settings inherited by all experiments
- nmda_avg: NMDA receptor-based heterogeneous model experiment
- (future: dopamine_d1, serotonin, etc.)
"""

from .base import BASE_CONFIG
from .nmda_avg import NMDA_CONFIG

__all__ = [
    'BASE_CONFIG',
    'NMDA_CONFIG'
]
