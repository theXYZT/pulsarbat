"""Core module consisting of basic building blocks of pulsarbat."""

import numpy as np
import astropy.units as u
from astropy.time import Time

__all__ = ['BasebandSignal', 'DispersionMeasure']


def verify_quantity(unit, *args):
    for a in args:
        if not isinstance(a, u.Quantity):
            raise TypeError(f'Expected astropy Quantity, got {type(a)}')
        if not a.unit.is_equivalent(unit):
            expected = f'Expected units of {unit.physical_type}'
            raise u.UnitTypeError(f'{expected}, got units of {a.unit}')
    return True


class InvalidSignalError(ValueError):
    """Used to catch invalid signals."""
    pass


class Signal:
    """Base class for all signals.

    A signal is uniquely described by an array of samples (`z`), a
    sampling rate (`sample_rate`), and a start time (`start_time`).

    Parameters
    ----------
    z : `~numpy.ndarray`
        The signal being stored as an array.
    sample_rate : `~astropy.units.Quantity`
        The number of samples per second. Must be in units of frequency.
    start_time : `astropy.time.Time`
        The start time of the signal (the time at the first sample of
        the signal).
    """
    def __init__(self, z: np.ndarray, sample_rate: u.Quantity,
                 start_time: Time):

        if verify_quantity(u.Hz, sample_rate):
            self._sample_rate = sample_rate.copy()

        try:
            self._start_time = Time(start_time, format='isot', precision=9)
        except ValueError:
            raise ValueError('Invalid start time provided.')

        if not isinstance(z, np.ndarray):
            raise TypeError('Input signal must be a numpy.ndarray!')

        try:
            self._z = self._create_signal_template(z)
            self._z[:] = z
        except (ValueError, TypeError):
            raise InvalidSignalError('Invalid signal provided.')

    def _create_signal_template(self, z):
        return np.empty_like(z, order='F')

    def copy(self):
        """Creates a deep copy of the object."""
        return Signal(self._z, self.sample_rate, self.start_time)

    def __array__(self):
        return self._z

    def __repr__(self):
        return (f"Signal Shape: {self.shape}\n"
                f"Sample Rate: {self.sample_rate}\n"
                f"Start time: {self.start_time.isot}\n")

    def __len__(self):
        return len(self._z)

    @property
    def shape(self):
        """Shape of the signal."""
        return self._z.shape

    @property
    def sample_shape(self):
        """Shape of a sample."""
        return self.shape[1:]

    @property
    def sample_rate(self):
        """Sample rate of the signal."""
        return self._sample_rate.copy()

    @sample_rate.setter
    def sample_rate(self, sample_rate):
        if verify_quantity(u.Hz, sample_rate):
            self._sample_rate = sample_rate.copy()

    @property
    def dt(self):
        """Sample spacing of the signal (1 / sample_rate)."""
        return (1 / self.sample_rate).to(u.s)

    @property
    def time_length(self):
        """Length of signal in time units."""
        return (len(self) * self.dt).to(u.s)

    @property
    def start_time(self):
        """Start time of the signal (Time at first sample)."""
        return self._start_time.copy()

    @start_time.setter
    def start_time(self, start_time):
        self._start_time = Time(start_time, format='isot', precision=9)

    @property
    def stop_time(self):
        return self.start_time + self.time_length


class BasebandSignal(Signal):
    """Stores complex baseband signals for analysis.

    A complex baseband signal is uniquely described by an array of
    complex samples (`z`), the sampling rate (`sample_rate`), the start
    time (`start_time`), and a reference frequency which in this case is
    the observing frequency at the center of the band (`center_freq`).

    `z` must be an array-like object containing a signal in a complex
    baseband representation. The shape of `z` is `(nsamples, nchan, ...)`
    such that the first dimension is always time (samples), and the rest
    is the sample shape. The first dimension of a sample (second
    dimension of `z`) is always frequency (channels). All channels are
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
    start_time : `astropy.time.Time`
        The start time of the signal (the time at the first sample of
        the signal).
    center_freq : `~astropy.units.Quantity`
        The observing frequency at the center of the signal's band. Must
        be in units of frequency.
    """
    def __init__(self, z: np.ndarray, sample_rate: u.Quantity,
                 start_time: Time, center_freq: u.Quantity):
        if z.ndim < 2:
            raise InvalidSignalError(f'Expected signal with >= 2 dimensions, '
                                     f'got signal with {z.ndim} dimension(s).')

        if verify_quantity(u.Hz, center_freq):
            self._center_freq = center_freq.copy()

        super().__init__(z, sample_rate, start_time)

    def _create_signal_template(self, z):
        return np.empty_like(z, dtype=np.complex64, order='F')

    def __repr__(self):
        signal_type = ('BasebandSignal\n'
                       '--------------\n')
        return (f"{signal_type}"
                f"{super().__repr__()}"
                f"Center Frequency: {self.center_freq}\n")

    def copy(self, z=None, sample_rate=None, start_time=None,
             center_freq=None):
        """Creates a copy of the object, unless specified otherwise."""
        z = z or self._z
        sample_rate = sample_rate or self.sample_rate
        start_time = start_time or self.start_time
        center_freq = center_freq or self.center_freq

        BasebandSignal(z, sample_rate, start_time, center_freq)

    @property
    def nchan(self):
        """Number of frequency channels."""
        return self.sample_shape[0]

    @property
    def center_freq(self):
        """Center observing frequency of the signal."""
        return self._center_freq.copy()

    @property
    def bandwidth(self):
        """Total bandwidth of signal (nchan * sample_rate)."""
        return self.nchan * self.sample_rate

    @property
    def channel_centers(self):
        """Returns a list of center frequencies for all channels."""
        chan_ids = [i + 0.5 - self.nchan / 2 for i in range(self.nchan)]
        return self.center_freq + self.sample_rate * chan_ids

    @property
    def max_freq(self):
        """Frequency at the top of the band."""
        return self.center_freq + self.bandwidth / 2

    @property
    def min_freq(self):
        """Frequency at the bottom of the band."""
        return self.center_freq - self.bandwidth / 2


class DispersionMeasure(u.SpecificTypeQuantity):
    _equivalent_unit = _default_unit = u.pc / u.cm**3
    dispersion_constant = u.s * u.MHz**2 * u.cm**3 / u.pc / 2.41E-4

    def time_delay(self, f, ref_freq):
        coeff = self.dispersion_constant * self
        delay = coeff * (1 / f**2 - 1 / ref_freq**2)
        return delay.to(u.s)

    def sample_delay(self, f, ref_freq, sample_rate):
        samples = self.time_delay(f, ref_freq) * sample_rate
        samples = samples.to_value(u.one)
        return int(np.copysign(np.ceil(abs(samples)), samples))

    def phase_delay(self, f, ref_freq):
        coeff = self.dispersion_constant * self
        phase = coeff * f * u.cycle * (1 / ref_freq - 1 / f)**2
        return phase.to(u.rad)

    def phase_factor(self, f, ref_freq):
        phase = self.phase_delay(f, ref_freq)
        factor = np.exp(-1j * phase.to_value(u.rad))
        return factor.astype(np.complex64)
