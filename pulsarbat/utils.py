"""Signal-to-signal transforms."""

import numpy as np
import astropy.units as u
from .core import BasebandSignal

try:
    import pyfftw
    fftpack = pyfftw.interfaces.numpy_fft
except ImportError:
    fftpack = np.fft

__all__ = ['baseband_signal_like']


class BasebandReader:
    """Basic baseband file reader."""

    def __init__(self, files, **baseband_kwargs):
        self.files = files
        self.fh = baseband.open(self._files, 'rs', **baseband_kwargs)

        self.start_time = Time(self.fh.start_time, format='isot', precision=9)
        self.stop_time = Time(self.fh.stop_time, format='isot', precision=9)
