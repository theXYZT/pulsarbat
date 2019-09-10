"""Core module consisting of basic building blocks of pulsarbat."""

import numpy as np
import astropy.units as u

__all__ = ['BasebandSignal', 'DispersionMeasure']


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
    channel `i` is given by:

        freq_i = center_freq + sample_rate * (i + 0.5 - nchan/2)

    where `i` is in `[0, ..., nchan - 1]`.

    The signal is always stored as 64-bit complex floating point numbers.
    This should be sufficient precision for practically all use cases.

    Parameters
    ----------
    z : `~numpy.ndarray`
        The signal being stored as an array. Must follow specifications
        described above.
    sample_rate : `~astropy.units.Quantity`
        The number of samples per second. Must be in units of frequency.
    center_freq : `~astropy.units.Quantity`
        The observing frequency at the center of the signal's band. Must
        be in units of frequency.
    """

    def __init__(self, z, sample_rate, center_freq):
        if not isinstance(z, np.ndarray):
            raise TypeError('Input signal must be a numpy.ndarray!')
        if z.ndim < 2:
            raise InvalidSignalError('Input data has less than 2 dimensions.')
        try:
            self._z = np.empty(z.shape, dtype=np.complex64, order='F')
            self._z[:] = z
        except (ValueError, TypeError):
            raise InvalidSignalError('Invalid signal provided.')

        if not isinstance(sample_rate, u.Quantity):
            raise TypeError('sample_rate must be an astropy Quantity.')
        if not sample_rate.unit.is_equivalent(u.Hz):
            raise u.UnitTypeError('sample_rate must have units of frequency.')
        self._sample_rate = sample_rate.copy()

        if not isinstance(center_freq, u.Quantity):
            raise TypeError('center_freq must be an astropy Quantity.')
        if not center_freq.unit.is_equivalent(u.Hz):
            raise u.UnitTypeError('center_freq must have units of frequency.')
        self._center_freq = center_freq.copy()

    def __array__(self):
        return self._z

    @property
    def shape(self):
        """Shape of the signal."""
        return self._z.shape

    def __len__(self):
        return len(self._z)

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
        return self._sample_rate.copy()

    @sample_rate.setter
    def sample_rate(self, sample_rate):
        self._sample_rate = sample_rate.copy()

    @property
    def dt(self):
        """Sample spacing of the signal."""
        return (1 / self.sample_rate).to(u.s)

    @property
    def center_freq(self):
        """Center observing frequency of the signal."""
        return self._center_freq.copy()

    def __repr__(self):
        return f"{self.shape} @ {self.sample_rate} [{self.center_freq}]"

    def copy(self):
        """Creates a deep copy of the object."""
        return BasebandSignal(np.copy(self._z, order='F'),
                              self.sample_rate, self.center_freq)

    def center_freqs(self):
        """Returns a list of center frequencies for all channels."""
        chan_ids = [i + 0.5 - self.nchan/2 for i in range(self.nchan)]
        return self.center_freq + self.sample_rate * chan_ids


class DispersionMeasure(u.SpecificTypeQuantity):
    _equivalent_unit = _default_unit = u.pc / u.cm**3
    dispersion_constant = u.s * u.MHz**2 * u.cm**3 / u.pc / 2.41E-4

    def time_delay(self, f, fref):
        coeff = self.dispersion_constant * self
        return coeff * (1/f**2 - 1/fref**2)

    def phase_delay(self, f, fref):
        coeff = self.dispersion_constant * self
        return coeff * f * u.cycle * (1/fref - 1/f)**2

    def phase_factor(self, f, fref):
        return np.exp(-1j * self.phase_delay(f, fref).to_value(u.rad))
