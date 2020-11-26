"""Top-level module for pulsarbat."""

# flake8: noqa

__author__ = """Nikhil Mahajan"""
__email__ = 'mahajan@astro.utoronto.ca'
__version__ = '0.0.4'


from . import core
from . import signal_funcs
from . import reader
from . import utils

from .core import *
from .signal_funcs import *

__all__ = []
__all__.extend(core.__all__)
__all__.extend(signal_funcs.__all__)
