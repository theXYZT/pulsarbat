"""Signal-to-signal transforms."""

import numpy as np
import astropy.units as u
from .core import BasebandSignal
from astropy.time import Time
import baseband

try:
    import pyfftw
    fftpack = pyfftw.interfaces.numpy_fft
except ImportError:
    fftpack = np.fft

__all__ = ['verify_quantity']


def verify_quantity(unit, *args):
    for a in args:
        if not isinstance(a, u.Quantity):
            raise TypeError(f'Expected astropy Quantity, got {type(a)}')
        if not a.unit.is_equivalent(unit):
            expected = f'Expected units of {unit.physical_type}'
            raise u.UnitTypeError(f'{expected}, got units of {a.unit}')
    return True


class BasebandReader:
    """Basic baseband file reader."""
    def __init__(self, files, **baseband_kwargs):
        self.files = files
        self.fh = baseband.open(self.files, 'rs', **baseband_kwargs)

        self.start_time = Time(self.fh.start_time, format='isot', precision=9)
        self.stop_time = Time(self.fh.stop_time, format='isot', precision=9)
