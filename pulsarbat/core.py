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


class InvalidSignalError(Exception):
    """Used to catch invalid signals."""


class BasebandSignal:
    """Stores complex baseband signals for analysis.

    A complex baseband signal is uniquely described by an array of
    observed samples (`z`), the sampling rate (`sample_rate`), and a
    reference frequency which in this case is the observing frequency at
    the center of the band (`center_freq`).

    `z` must be an array-like object containing a signal in a complex
    baseband representation. The shape of `z` is `(nsamples, nchan, ...)`
    such that the first dimension is always time (samples), and the rest
    is the sample shape. The first dimension of a sample (second
    dimension of `z`) is always frequency (samples). All channels are
    assumed to be Nyquist-sampled (i.e., adjacent channels are separated
    by a bandwidth of `sample_rate`). Thus, the center frequency of
    channel i is given by::

        freq_i = center_freq + sample_rate * (i + 0.5 - nchan/2)

    where i is in [0, ..., nchan - 1].

    Parameters
    ----------
    z : array_like
        The signal being stored as an array. Must follow specifications
        described above.
    sample_rate : :py:class:`~astropy.units.Quantity`
        The number of samples per second. Must be in units of frequency.
    center_freq : `~astropy.units.Quantity`
        The observing frequency at the center of the signal's band. Must
        be in units of frequency.
    double_precision : bool, optional
        Whether the signal is stored as 64-bit (False) or 128-bit (True)
        complex floating point number. Setting this to true will likely
        slow everything down by more than a factor of two. Only do so if
        absolutely certain that precision is needed. Default is False.
    """

    def __init__(self, z, sample_rate, center_freq, double_precision=False):
        if z.ndim < 2:
            raise InvalidSignalError('Signal has less than 2 dimensions.')

        try:
            if double_precision:
                self._z = np.asarray(z).astype(np.complex128)
            else:
                self._z = np.asarray(z).astype(np.complex64)
        except (TypeError, ValueError):
            raise InvalidSignalError('Invalid signal provided.')

        if not isinstance(sample_rate, u.Quantity):
            raise TypeError('sample_rate must be an astropy Quantity.')
        if not sample_rate.unit.is_equivalent(u.Hz):
            raise u.UnitTypeError('sample_rate must have units of frequency.')
        self._sample_rate = sample_rate

        if not isinstance(center_freq, u.Quantity):
            raise TypeError('center_freq must be an astropy Quantity.')
        if not center_freq.unit.is_equivalent(u.Hz):
            raise u.UnitTypeError('center_freq must have units of frequency.')
        self._center_freq = center_freq

    @property
    def dtype(self):
        """Data type of the signal."""
        return self._z.dtype

    @property
    def shape(self):
        """Shape of the signal."""
        return self._z.shape

    @property
    def __len__(self):
        return self.shape[0]

    @property
    def sample_shape(self):
        """Shape of a sample."""
        return self.shape[1:]

    @property
    def nchan(self):
        """Number of frequency channels."""
        return self.sample_shape[0]

    @property
    def sample_rate(self):
        """Sample rate of the signal."""
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, sample_rate):
        self._sample_rate = sample_rate

    @property
    def dt(self):
        """Sample spacing of the signal."""
        return (1 / self.sample_rate).to(u.s)

    @property
    def center_freq(self):
        """Center observing frequency of the signal."""
        return self._center_freq

    @property
    def array(self):
        """Returns the signal as a :py:mod:`numpy` array."""
        return self._z

    def __repr__(self):
        return f"{self.shape} @ {self.sample_rate} [{self.center_freq}]"


class BasebandReader:
    """Basic baseband file reader."""

    def __init__(self, files, **baseband_kwargs):
        self.files = files
        self.fh = baseband.open(self._files, 'rs', **baseband_kwargs)

        self.start_time = Time(self.fh.start_time, format='isot', precision=9)
        self.stop_time = Time(self.fh.stop_time, format='isot', precision=9)
