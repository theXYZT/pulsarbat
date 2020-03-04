"""Core module consisting of basic building blocks of pulsarbat."""

import numpy as np
import astropy.units as u
from astropy.time import Time
import inspect
from .utils import verify_scalar_quantity

__all__ = [
    'Signal',
    'RadioSignal',
    'BasebandSignal',
    'IntensitySignal',
    'InvalidSignalError',
    'DispersionMeasure',
]


class InvalidSignalError(ValueError):
    """Used to catch invalid signals."""
    pass


class Signal:
    """Base class for all signals.

    A signal is sufficiently described by an array of samples (`data`),
    and a constant sampling rate (`sample_rate`). Optionally, a `start_time`
    can be provided to specify the timestamp at the first sample of the
    signal.

    `data` must be a `~numpy.ndarray` object where the zeroth axis refers
    to time. That is, `data[i]` is the `i`-th sample of the signal, and the
    sample shape can be arbitrary. `data` must have at least 1 dimension.

    Parameters
    ----------
    data : `~numpy.ndarray`
        The signal being stored as an array.
    sample_rate : `~astropy.units.Quantity`
        The number of samples per second. Must be in units of frequency.
    start_time : `~astropy.time.Time`, optional
        The start time of the signal (that is, the time at the first
        sample of the signal). Default is None.
    """
    _min_ndim = 1
    _dtype = None

    def __init__(self,
                 data: np.ndarray,
                 sample_rate: u.Quantity,
                 start_time: Time = None):

        self.sample_rate = sample_rate
        self.start_time = start_time

        if not isinstance(data, np.ndarray):
            raise TypeError('Input signal must be an ndarray object!')

        if data.ndim < self._min_ndim:
            err = (f'Expected signal with >= {self._min_ndim} dimension(s), '
                   f'got signal with {data.ndim} dimension(s) instead!')
            raise InvalidSignalError(err)

        try:
            self._data = self._create_signal_template(data)
            self._data[:] = data
        except (ValueError, TypeError):
            raise InvalidSignalError('Invalid signal provided.')

        self._verification_checks()

    def _verification_checks(self):
        pass

    def _create_signal_template(self, z):
        dtype = self._dtype or z.dtype
        return np.empty_like(z, order='F', dtype=dtype)

    def __array__(self):
        return self._data

    def __repr__(self):
        signature = f"{self.__class__.__name__} @ {hex(id(self))}"
        return (f"{signature}\n"
                f"{'-' * len(signature)}\n"
                f"Signal shape: {self.shape}\n"
                f"Signal dtype: {self.dtype}\n"
                f"Sample rate: {self.sample_rate}\n"
                f"Start Time: {self.start_time.isot}\n"
                f"Time Length: {self.time_length}")

    def __len__(self):
        return len(self._data)

    @property
    def data(self):
        """The signal data."""
        return np.array(self)

    @property
    def shape(self):
        """Shape of the signal."""
        return self._data.shape

    @property
    def sample_shape(self):
        """Shape of a sample."""
        return self.shape[1:]

    @property
    def ndim(self):
        """Number of dimensions in data."""
        return self._data.ndim

    @property
    def dtype(self):
        """Data type of the signal."""
        return self._data.dtype

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

    @property
    def start_time(self):
        """Start time of the signal (Time at first sample)."""
        if self._start_time is not None:
            return self._start_time.copy()
        return None

    @start_time.setter
    def start_time(self, start_time):
        if start_time is not None:
            self._start_time = Time(start_time, format='isot', precision=9)

            if not self._start_time.isscalar:
                raise ValueError('Start time must be a scalar!')
        else:
            self._start_time = start_time

    @property
    def stop_time(self):
        """Stop time of the signal (Time at sample after the last sample)."""
        if self._start_time is not None:
            return self.start_time + self.time_length
        return None

    @classmethod
    def like(cls, signal, *args, **kwargs):
        """Creates an object like `signal` unless overridden by args/kwargs.

        This classmethod inspects the class signature and creates an object
        using given `*args` and `**kwargs`. For all arguments required by
        the signature that are not provided, they are instead acquired
        pulled from attributes of the same name in given object, `signal`.
        """
        sig = inspect.signature(cls)
        params = sig.bind_partial(*args, **kwargs).arguments
        for k, v in sig.parameters.items():
            if k not in params:
                if hasattr(signal, k):
                    params[k] = getattr(signal, k)
                elif v.default is not v.empty:
                    params[k] = v.default
                else:
                    raise TypeError(f'Missing required argument: {k}')
        return cls(**params)


class RadioSignal(Signal):
    """Class for observed radio signals.

    A signal is sufficiently described by an array of samples (`data`),
    and a sampling rate (`sample_rate`). A radio signal is a signal
    observed by some antenna or analog receiver of some sort. Thus, it
    comes with additional metadata such as a start time (`start_time`),
    a center frequency (`center_freq`), and a bandwidth (`bandwidth`).

    Radio signals can be channelized (split into adjacent channels in
    frequency). In order to simplify management of these channels, the
    input signal (`data`) is required to be a `~numpy.ndarray` object with
    at least 2 dimensions. The first dimension (`axis = 0`) refers to
    time (as required by the parent class `~Signal`). The second
    dimension (`axis = 1`) refers to frequency, such that `data[:, i]` is the
    `i`-th channel of the signal.

    The shape of `data` must be `(nsamples, nchan, ...)`.

    The channels must be adjacent in frequency and of equal bandwidth,
    such that the center frequency of a channel `i` is given by,::

        freq_i = center_freq + (bandwidth / nchan) * (i + 0.5 - nchan/2)

    where `i` is in `[0, ..., nchan - 1]` and `nchan` is the number of
    channels (`data.shape[1]`) and `bandwidth / nchan` is the bandwidth of
    a single channel.

    Input data that is unchannelized must still be treated as data with
    1 channel where `data.shape[1] = 1`.

    Parameters
    ----------
    data : `~numpy.ndarray`
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

    def __init__(self, data: np.ndarray, sample_rate: u.Quantity,
                 start_time: Time, center_freq: u.Quantity,
                 bandwidth: u.Quantity):
        super().__init__(data=data,
                         sample_rate=sample_rate,
                         start_time=start_time)
        self.center_freq = center_freq
        self.bandwidth = bandwidth

    def __repr__(self):
        return (f"{super().__repr__()}\n"
                f"Bandwidth: {self.bandwidth}\n"
                f"Center Freq.: {self.center_freq}\n")

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
    data : `~numpy.ndarray`
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

    See the documentation for `~RadioSignal` for other specifications.

    `data` must be an array-like object containing a signal in a complex
    baseband representation. The signal is always stored as 64-bit
    complex floating point numbers. This should be sufficient precision
    for practically all use cases.

    Parameters
    ----------
    data : `~numpy.ndarray`
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

    def _verification_checks(self):
        super()._verification_checks()

        if not np.isclose(self.chan_bandwidth, self.sample_rate):
            err = (f"Sample rate ({self.sample_rate}) !="
                   f"channel bandwidth ({self.chan_bandwidth})!")
            raise InvalidSignalError(err)


class DispersionMeasure(u.SpecificTypeQuantity):
    _equivalent_unit = _default_unit = u.pc / u.cm**3
    dispersion_constant = u.s * u.MHz**2 * u.cm**3 / u.pc / 2.41E-4

    def time_delay(self, f, ref_freq):
        """Time delay of frequencies relative to reference frequency."""
        coeff = self.dispersion_constant * self
        delay = coeff * (1 / f**2 - 1 / ref_freq**2)
        return delay.to(u.s)

    def sample_delay(self, f, ref_freq, sample_rate):
        """Sample delay of frequencies relative to reference frequency."""
        samples = self.time_delay(f, ref_freq) * sample_rate
        samples = samples.to_value(u.one)
        return int(np.copysign(np.ceil(abs(samples)), samples))

    def transfer_function(self, f, ref_freq):
        """Returns the transfer function for dedispersion."""
        coeff = self.dispersion_constant * self
        phase = coeff * f * u.cycle * (1 / ref_freq - 1 / f)**2
        transfer = np.exp(1j * phase.to_value(u.rad))
        return np.asfortranarray(transfer.astype(np.complex64))
