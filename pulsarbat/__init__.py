"""Top-level package for pulsarbat."""

__author__ = """Nikhil Mahajan"""
__email__ = 'mahajan@astro.utoronto.ca'
__version__ = '0.0.3'

from . import core
from . import transforms
from . import reductions
from . import pulsar
from . import utils

from .core import (Signal, RadioSignal, BasebandSignal, IntensitySignal)
from .predictor import Polyco
from .observation import (Observation, PUPPIObservation)

__all__ = ['core', 'transforms', 'reductions', 'pulsar', 'utils', 'Polyco']
__all__ += ['Signal', 'RadioSignal', 'BasebandSignal', 'IntensitySignal']
__all__ += ['Observation', 'PUPPIObservation']
