"""Core module consisting of basic building blocks of pulsarbat."""

import numpy as np
import astropy.units as u
from astropy.time import Time

__all__ = [
    'Signal', 'RadioSignal', 'BasebandSignal', 'IntensitySignal',
    'DispersionMeasure', 'verify_scalar_quantity', 'InvalidSignalError'
]


class InvalidSignalError(ValueError):
    """Used to catch invalid signals."""
    pass


def verify_scalar_quantity(a, unit):
    if not isinstance(a, u.Quantity):
        raise TypeError(f'Expected astropy Quantity, got {type(a)}')

    if not a.unit.is_equivalent(unit):
        expected = f'Expected units of {unit.physical_type}'
        raise u.UnitTypeError(f'{expected}, got units of {a.unit}')

    if not a.isscalar:
        raise ValueError(f'Expected a scalar quantity.')

    return True


def not_none(value, default):
    return value if value is not None else default


class Signal:
    """Base class for all signals.

    A signal is sufficiently described by an array of samples (`z`),
    and a constant sampling rate (`sample_rate`).

    `z` must be a `~numpy.ndarray` object where the zeroth axis refers
    to time. That is, z[i] is the `i`th sample of the signal, and the
    sample shape can be arbitrary. `z` must have at least 1 dimension.

    Parameters
    ----------
    z : `~numpy.ndarray`
        The signal being stored as an array.
    sample_rate : `~astropy.units.Quantity`
        The number of samples per second. Must be in units of frequency.
    """
    _min_ndim = 1
    _dtype = None

    def __init__(self, z: np.ndarray, sample_rate: u.Quantity):
        self.sample_rate = sample_rate

        self._verify_signal(z)
        try:
            self._z = self._create_signal_template(z)
            self._z[:] = z
        except (ValueError, TypeError):
            raise InvalidSignalError('Invalid signal provided.')

    def _verify_signal(self, z):
        """Verifies that signal matches specifications."""
        if not isinstance(z, np.ndarray):
            raise TypeError('Input signal must be an ndarray object!')

        if z.ndim < self._min_ndim:
            err = (f'Expected signal with >= {self._min_ndim} dimension(s), '
                   f'got signal with {z.ndim} dimension(s) instead!')
            raise InvalidSignalError(err)

    def _create_signal_template(self, z):
        dtype = self._dtype or z.dtype
        return np.empty_like(z, order='F', dtype=dtype)

    def copy(self, z=None, sample_rate=None):
        """Creates a copy of the object."""
        return type(self)(not_none(z, self._z),
                          not_none(sample_rate, self.sample_rate))

    def __array__(self):
        return self._z

    def __repr__(self):
        signature = f"{self.__class__.__name__} @ {hex(id(self))}"
        return (f"{signature}\n"
                f"{'-' * len(signature)}\n"
                f"Signal shape: {self.shape}\n"
                f"Signal dtype: {self.dtype}\n"
                f"Sample rate: {self.sample_rate}")

    def __len__(self):
        return len(self._z)

    @property
    def data(self):
        """The signal data."""
        return np.array(self)

    @property
    def shape(self):
        """Shape of the signal."""
        return self._z.shape

    @property
    def sample_shape(self):
        """Shape of a sample."""
        return self.shape[1:]

    @property
    def ndim(self):
        """Number of dimensions in data."""
        return self._z.ndim

    @property
    def dtype(self):
        """Data type of the signal."""
        return self._z.dtype

    @property
    def sample_rate(self):
        """Sample rate of the signal."""
        return self._sample_rate.copy()

    @sample_rate.setter
    def sample_rate(self, sample_rate):
        if verify_scalar_quantity(sample_rate, u.Hz):
            self._sample_rate = sample_rate.copy()

    @property
    def dt(self):
        """Sample spacing of the signal (1 / sample_rate)."""
        return (1 / self.sample_rate).to(u.s)

    @property
    def time_length(self):
        """Length of signal in time units."""
        return (len(self) * self.dt).to(u.s)

    def expand_dims(self, ndim):
        """Expand dimensions of signal to provided number of dimensions."""
        if ndim < self.ndim:
            raise ValueError("Given ndim is smaller than signal ndim!")
        else:
            new_shape = self.shape + (1, ) * (ndim - self.ndim)
            self._z = self._z.reshape(new_shape)


class RadioSignal(Signal):
    """Class for observed radio signals.

    A signal is sufficiently described by an array of samples (`z`),
    and a sampling rate (`sample_rate`). A radio signal is a signal
    observed by some antenna or analog receiver of some sort. Thus, it
    comes with additional metadata such as a start time (`start_time`),
    a center frequency (`center_freq`), and a bandwidth (`bandwidth`).

    Radio signals can be channelized (split into adjacent channels in
    frequency). In order to simplify management of these channels, the
    input signal (`z`) is required to be a `~numpy.ndarray` object with
    at least 2 dimensions. The first dimension (`axis = 0`) refers to
    time (as required by the parent class `~Signal`). The second
    dimension (`axis = 1`) refers to frequency, such that z[:, i] is the
    `i`th channel of the signal.

    The shape of `z` must be `(nsamples, nchan, ...)`.

    The channels must be adjacent in frequency and of equal bandwidth,
    such that the center frequency of a channel `i` is given by,

        freq_i = center_freq + (bandwidth / nchan) * (i + 0.5 - nchan/2)

    where `i` is in `[0, ..., nchan - 1]` and `nchan` is the number of
    channels (`z.shape[1]`) and `bandwidth / nchan` is the bandwidth of
    a single channel.

    Input data that is unchannelized must still be treated as data with
    1 channel where `z.shape[1] = 1`.

    Parameters
    ----------
    z : `~numpy.ndarray`
        The signal being stored as an array.
    sample_rate : `~astropy.units.Quantity`
        The number of samples per second. Must be in units of frequency.
    start_time : `~astropy.time.Time`
        The start time of the signal (that is, the time at the first
        sample of the signal).
    center_freq : `~astropy.units.Quantity`
        The observing frequency at the center of the signal's band. Must
        be in units of frequency.
    bandwidth : `~astropy.units.Quantity`
        The total bandwidth of the signal. The channel bandwidth is this
        total bandwidth divided by the number of channels. Must be in
        units of frequency.
    """
    _min_ndim = 2

    def __init__(self, z: np.ndarray, sample_rate: u.Quantity,
                 start_time: Time, center_freq: u.Quantity,
                 bandwidth: u.Quantity):
        super().__init__(z, sample_rate)
        self.center_freq = center_freq
        self.bandwidth = bandwidth
        self.start_time = start_time

    def copy(self,
             z=None,
             sample_rate=None,
             start_time=None,
             center_freq=None,
             bandwidth=None):
        """Creates a copy of the object."""
        return type(self)(not_none(z, self._z),
                          not_none(sample_rate, self.sample_rate),
                          not_none(start_time, self.start_time),
                          not_none(center_freq, self.center_freq),
                          not_none(bandwidth, self.bandwidth))

    def __repr__(self):
        return (f"{super().__repr__()}\n"
                f"Bandwidth: {self.bandwidth}\n"
                f"Center Freq.: {self.center_freq}\n"
                f"Start Time: {self.start_time.isot}\n")

    @property
    def start_time(self):
        """Start time of the signal (Time at first sample)."""
        return self._start_time.copy()

    @start_time.setter
    def start_time(self, start_time):
        self._start_time = Time(start_time, format='isot', precision=9)

        if not self._start_time.isscalar:
            raise ValueError('Start time must be a scalar!')

    @property
    def stop_time(self):
        return self.start_time + self.time_length

    @property
    def nchan(self):
        """Number of frequency channels."""
        return self.shape[1]

    @property
    def center_freq(self):
        """Center observing frequency of the signal."""
        return self._center_freq.copy()

    @center_freq.setter
    def center_freq(self, center_freq):
        if verify_scalar_quantity(center_freq, u.Hz):
            self._center_freq = center_freq.copy()

    @property
    def bandwidth(self):
        """Total bandwidth of signal."""
        return self._bandwidth.copy()

    @bandwidth.setter
    def bandwidth(self, bandwidth):
        if verify_scalar_quantity(bandwidth, u.Hz):
            self._bandwidth = bandwidth.copy()

    @property
    def chan_bandwidth(self):
        """Bandwidth of a single channel."""
        return self.bandwidth / self.nchan

    @property
    def max_freq(self):
        """Frequency at the top of the band."""
        return self.center_freq + self.bandwidth / 2

    @property
    def min_freq(self):
        """Frequency at the bottom of the band."""
        return self.center_freq - self.bandwidth / 2

    @property
    def channel_centers(self):
        """Returns a list of center frequencies for all channels."""
        chan_ids = [i + 0.5 - self.nchan / 2 for i in range(self.nchan)]
        return self.center_freq + self.chan_bandwidth * chan_ids


class IntensitySignal(RadioSignal):
    """Stores intensity signals.

    See the documentation for `~RadioSignal` for specifications.

    Parameters
    ----------
    z : `~numpy.ndarray`
        The signal being stored as an array.
    sample_rate : `~astropy.units.Quantity`
        The number of samples per second. Must be in units of frequency.
    start_time : `~astropy.time.Time`
        The start time of the signal (that is, the time at the first
        sample of the signal).
    center_freq : `~astropy.units.Quantity`
        The observing frequency at the center of the signal's band. Must
        be in units of frequency.
    bandwidth : `~astropy.units.Quantity`
        The total bandwidth of the signal. The channel bandwidth is this
        total bandwidth divided by the number of channels. Must be in
        units of frequency.
    """
    _dtype = np.float32


class BasebandSignal(RadioSignal):
    """Stores complex baseband signals.

    Baseband signals are assumed to be raw radio data which are Nyquist
    sampled analytic signals (the bandwidth of a channel is equal to the
    sampling rate).

    See the documentation for `~RadioSignal` for specifications.

    `z` must be an array-like object containing a signal in a complex
    baseband representation. The signal is always stored as 64-bit
    complex floating point numbers. This should be sufficient precision
    for practically all use cases.

    Parameters
    ----------
    z : `~numpy.ndarray`
        The signal being stored as an array.
    sample_rate : `~astropy.units.Quantity`
        The number of samples per second. Must be in units of frequency.
    start_time : `~astropy.time.Time`
        The start time of the signal (that is, the time at the first
        sample of the signal).
    center_freq : `~astropy.units.Quantity`
        The observing frequency at the center of the signal's band. Must
        be in units of frequency.
    bandwidth : `~astropy.units.Quantity`
        The total bandwidth of the signal. The channel bandwidth is this
        total bandwidth divided by the number of channels. Must be in
        units of frequency.
    """
    _dtype = np.complex64

    def __init__(self, z: np.ndarray, sample_rate: u.Quantity,
                 start_time: Time, center_freq: u.Quantity,
                 bandwidth: u.Quantity):
        super().__init__(z, sample_rate, start_time, center_freq, bandwidth)

        if not np.isclose(self.chan_bandwidth, self.sample_rate):
            err = 'Sample rate is not equal to channel bandwidth!'
            raise InvalidSignalError(err)


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
