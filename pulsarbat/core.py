"""Core module consisting of basic building blocks of pulsarbat."""

from pathlib import Path
import numpy as np
import astropy.units as u
from astropy.time import Time
import baseband
try:
    import pyfftw
    fftpack = pyfftw.interfaces.numpy_fft
except ImportError:
    fftpack = np.fft

__all__ = ['BasebandSignal']

class BasebandSignal:
    """Baseband signal class.

    Signals are in their complex baseband representation (the spectrum
    must be centered on zero). Time is axis 0, the rest is the sample
    shape. The first axis of sample shape (axis 1) is frequency.
    """

    def __init__(self, z, sample_rate=None, center_freq=None):
        if not isinstance(sample_rate, u.Quantity):
            raise TypeError('sample_rate must be an astropy Quantity.')
        if not isinstance(center_freq, u.Quantity):
            raise TypeError('center_freq must be an astropy Quantity.')

        if not sample_rate.unit.is_equivalent(u.Hz):
            raise u.UnitTypeError('sample_rate must have units of frequency.')
        if not center_freq.unit.is_equivalent(u.Hz):
            raise u.UnitTypeError('center_freq must have units of frequency.')

        self._z = np.asarray(z).astype(np.complex64)
        self._sample_rate = sample_rate
        self._center_freq = center_freq

    def __repr__(self):
        return f"{self._z.shape} @ {self._sample_rate} [{self._center_freq}]"

    @property
    def dt(self):
        return (1 / self._sample_rate).to(u.s)

    @property
    def data(self):
        return self._z

    @property
    def sample_rate(self):
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, sample_rate):
        self._sample_rate = sample_rate



class BasebandReader:
    """Basic baseband file reader."""

    def __init__(self, files, **baseband_kwargs):
        self.files = files
        self.fh = baseband.open(self._files, 'rs', **baseband_kwargs)

        self.start_time = Time(self.fh.start_time, format='isot', precision=9)
        self.stop_time = Time(self.fh.stop_time, format='isot', precision=9)
        pass
