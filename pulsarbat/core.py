"""Core module consisting of basic building blocks of pulsarbat."""

import inspect
import numpy as np
import astropy.units as u
from astropy.time import Time
from itertools import zip_longest
from .utils import verify_scalar_quantity

__all__ = [
    'Signal',
    'RadioSignal',
    'IntensitySignal',
    'BasebandSignal',
    'FullStokesSignal',
    'DualPolarizationSignal',
    'InvalidSignalError',
    'DispersionMeasure',
]


class InvalidSignalError(ValueError):
    """Used to catch invalid signals."""
    pass


class Signal:
    """Base class for all signals.

    A signal is sufficiently described by an array of samples (`z`),
    and a constant sampling rate (`sample_rate`). Optionally, a
    `start_time` can be provided to specify the timestamp at the first
    sample of the signal.

    The signal data (`z`) must be an `~numpy.ndarray` or similar object
    (for example, dask arrays) with at least 1 dimension where the
    zeroth axis (`axis=0`) refers to time. That is, `z[i]` is the `i`-th
    sample of the signal, and the sample shape (`z[i].shape`) can be
    arbitrary.

    Parameters
    ----------
    z : `~numpy.ndarray`-like
        The signal data. Must be at least 1-dimensional with shape
        `(nsample, ...)`, and must have non-zero size.
    sample_rate : `~astropy.units.Quantity`
        The number of samples per second. Must be in units of frequency.
    start_time : `~astropy.time.Time`, optional
        The start time of the signal (that is, the time at the first
        sample of the signal). Default is None.
    """
    _dtype = None
    _shape = (None, )

    def __init__(self, z, /, *, sample_rate, start_time=None):
        self.sample_rate = sample_rate
        self.start_time = start_time

        if z.size == 0:
            raise InvalidSignalError('Expected signal with non-zero size.')

        if z.ndim < len(self._shape):
            err = (f'Expected at least {len(self._shape)} dimension(s), '
                   f'got signal with {z.ndim} dimension(s) instead!')
            raise InvalidSignalError(err)

        zipped = zip_longest(z.shape[:len(self._shape)], self._shape)
        if not all(x == (y or x) for x, y in zipped):
            err = (f'Input signal has invalid shape. Expected {self._shape}, '
                   f'got {z.shape} instead!')
            raise InvalidSignalError(err)

        self._data = z.astype(self._dtype or z.dtype)
        self._verification_checks()

    def _verification_checks(self):
        pass

    def __array__(self):
        return np.asanyarray(self.data)

    def __str__(self):
        signature = f"{self.__class__.__name__} @ {hex(id(self))}"
        c = type(self.data)
        s = f"{signature}\n{'-' * len(signature)}\n"
        s += f"Data Container: {c.__module__}.{c.__name__}"
        s += f"<shape={self.shape}, dtype={self.dtype}>\n"
        s += f"Sample rate: {self.sample_rate}\n"
        s += f"Time length: {self.time_length}\n"
        st = 'N/A' if self.start_time is None else self.start_time.isot
        s += f"Start time: {st}"
        return s

    def __repr__(self):
        info = f"shape={self.shape}, dtype={self.dtype}"
        s = f"pulsarbat.{self.__class__.__name__}<{info}> @ {hex(id(self))}"
        return s

    def __len__(self):
        return len(self._data)

    @property
    def data(self):
        """The signal data."""
        return self._data

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
            self._sample_rate = sample_rate.to(u.MHz)

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
            temp = Time(start_time, precision=9)
            if not temp.isscalar:
                raise ValueError('Start time must be a scalar!')
            self._start_time = temp
        else:
            self._start_time = None

    @property
    def stop_time(self):
        """Stop time of the signal (Time at sample after the last sample)."""
        if self._start_time is not None:
            return self.start_time + self.time_length
        return None

    def __contains__(self, time):
        if self.start_time is not None:
            return self.start_time <= time < self.stop_time
        else:
            return False

    @classmethod
    def like(cls, obj, z, /, **kwargs):
        """Creates a signal object "like" another signal object.

        This classmethod inspects the class signature and creates a
        signal object like the given reference object. The signal data
        (as an array) must always be provided as well as any other
        arguments that are intended to be different than the reference
        object. All arguments required by the class signature that are
        not provided are acquired from attributes of the same name in
        the reference object.

        Parameters
        ----------
        obj : `~pulsarbat.Signal`
            The reference signal object.
        z : `~numpy.ndarray`-like
            The signal data.
        **kwargs
            Additional keyword arguments to pass on to the target class.
        """
        sig = inspect.signature(cls)

        for k, v in sig.parameters.items():
            if v.kind is not v.POSITIONAL_ONLY and k not in kwargs:
                if hasattr(obj, k):
                    kwargs[k] = getattr(obj, k)
                elif v.default is v.empty:
                    raise TypeError(f'Missing required keyword argument: {k}')

        return cls(z, **kwargs)


class RadioSignal(Signal):
    """Class for heterodyned radio signals.

    A signal is sufficiently described by an array of samples (`z`),
    and a constant sampling rate (`sample_rate`). Optionally, a
    `start_time` can be provided to specify the timestamp at the first
    sample of the signal. A radio signal is often heterodyned, so a
    center frequency (`center_freq`) and a bandwidth (`bandwidth`) is
    necessary.

    The signal data (`z`) must a `~numpy.ndarray` or similar object
    (for example, dask arrays) with at least 2 dimensions where the
    first two dimensions refer to time (`axis=0`) and frequency
    (`axis=1`), respectively.

    Parameters
    ----------
    z : `~numpy.ndarray`-like
        The signal data. Must be at least 2-dimensional with shape
        `(nsample, nchan, ...)`.
    sample_rate : `~astropy.units.Quantity`
        The number of samples per second. Must be in units of frequency.
    center_freq : `~astropy.units.Quantity`
        The observing frequency at the center of the signal's band. Must
        be in units of frequency.
    bandwidth : `~astropy.units.Quantity`
        The total bandwidth of the signal. The channel bandwidth is this
        total bandwidth divided by the number of channels. Must be in
        units of frequency.
    start_time : `~astropy.time.Time`, optional
        The start time of the signal (that is, the time at the first
        sample of the signal). Default is None.

    Notes
    -----
    The input signal array must have shape `(nsample, nchan, ...)`,
    where `nsample` is the number of time samples, and `nchan` is the
    number of frequency channels.

    The channels must be adjacent in frequency and of equal channel
    bandwidth, such that the center frequency of a channel `i` is given
    by,::

        freq_i = center_freq + (bandwidth / nchan) * (i + 0.5 - nchan/2)

    where `i` is in `[0, ..., nchan - 1]` and `nchan` is the number of
    channels (`data.shape[1]`) and `bandwidth / nchan` is the bandwidth of
    a single channel.

    Input data that is unchannelized must still be treated as data with
    1 channel where `data.shape[1] = 1`.
    """
    _shape = (None, None, )

    def __init__(self, z, /, *, sample_rate, center_freq, bandwidth,
                 start_time=None):
        self.center_freq = center_freq
        self.bandwidth = bandwidth

        super().__init__(z, sample_rate=sample_rate, start_time=start_time)

    def __str__(self):
        s = super().__str__() + "\n"
        s += f"Total Bandwidth: {self.bandwidth}\n"
        s += f"Center Frequency: {self.center_freq}"
        return s

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
            self._center_freq = center_freq.to(u.MHz)

    @property
    def bandwidth(self):
        """Total bandwidth of signal."""
        return self._bandwidth.copy()

    @bandwidth.setter
    def bandwidth(self, bandwidth):
        if verify_scalar_quantity(bandwidth, u.Hz):
            self._bandwidth = bandwidth.to(u.MHz)

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
    """Class for intensity signals such as Stokes I, Q, U, V, etc.

    See the documentation for `~RadioSignal` for more details.

    Parameters
    ----------
    z : `~numpy.ndarray`-like
        The signal data. Must be at least 2-dimensional with shape
        `(nsample, nchan, ...)`.
    sample_rate : `~astropy.units.Quantity`
        The number of samples per second. Must be in units of frequency.
    center_freq : `~astropy.units.Quantity`
        The observing frequency at the center of the signal's band. Must
        be in units of frequency.
    bandwidth : `~astropy.units.Quantity`
        The total bandwidth of the signal. The channel bandwidth is this
        total bandwidth divided by the number of channels. Must be in
        units of frequency.
    start_time : `~astropy.time.Time`, optional
        The start time of the signal (that is, the time at the first
        sample of the signal). Default is None.

    See Also
    --------
    Signal
    RadioSignal

    Notes
    -----
    Signal data is converted to and stored with `dtype=np.float64`, as
    intensity signals are necessarily real-valued, and more
    floating-point precision is rarely required.
    """
    _dtype = np.float32


class BasebandSignal(RadioSignal):
    """Class for complex baseband signals.

    See the documentation for `~RadioSignal` for more details.

    Parameters
    ----------
    z : `~numpy.ndarray`-like
        The signal data. Must be at least 2-dimensional with shape
        `(nsample, nchan, ...)`.
    sample_rate : `~astropy.units.Quantity`
        The number of samples per second. Must be in units of frequency.
    center_freq : `~astropy.units.Quantity`
        The observing frequency at the center of the signal's band. Must
        be in units of frequency.
    bandwidth : `~astropy.units.Quantity`
        The total bandwidth of the signal. Must be equal to
        `nchan * sample_rate`. Must be in units of frequency.
    start_time : `~astropy.time.Time`, optional
        The start time of the signal (that is, the time at the first
        sample of the signal). Default is None.

    See Also
    --------
    Signal
    RadioSignal

    Notes
    -----
    Baseband signals are Nyquist-sampled analytic signals (the bandwidth
    of a channel is equal to the sampling rate). The signal data is
    converted to and stored with `dtype=np.complex128` as baseband
    signals are necessarily complex-valued and more floating-point
    precision is rarely required.

    Since, the signal is expected to be Nyquist-sampled and frequency
    channels must be contiguous, the following must be true::

        sample_rate = bandwidth / nchan

    All frequency channels are assumed to be upper side-band (that is,
    the signal isn't reversed in frequency space).
    """
    _dtype = np.complex64

    def _verification_checks(self):
        super()._verification_checks()

        if not u.isclose(self.chan_bandwidth, self.sample_rate):
            err = (f"Sample rate ({self.sample_rate}) != "
                   f"channel bandwidth ({self.chan_bandwidth})!")
            raise InvalidSignalError(err)

    def to_intensity(self):
        """Converts baseband signal to intensities.

        Returns
        -------
        out : `~pulsarbat.BasebandSignal`
            The converted signal.
        """
        z = self._data.real**2 + self._data.imag**2
        return IntensitySignal.like(self, z)


class FullStokesSignal(IntensitySignal):
    """Class for full Stokes (I, Q, U, V) signals.

    See the documentation for `~RadioSignal` and `~IntensitySignal` for
    more details.

    Parameters
    ----------
    z : `~numpy.ndarray`-like
        The signal data. Must be at least 3-dimensional with shape
        `(nsample, nchan, nstokes, ...)` where `nstokes = 4`, and the
        order of components is `[I, Q, U, V]`
    sample_rate : `~astropy.units.Quantity`
        The number of samples per second. Must be in units of frequency.
    center_freq : `~astropy.units.Quantity`
        The observing frequency at the center of the signal's band. Must
        be in units of frequency.
    bandwidth : `~astropy.units.Quantity`
        The total bandwidth of the signal. The channel bandwidth is this
        total bandwidth divided by the number of channels. Must be in
        units of frequency.
    start_time : `~astropy.time.Time`, optional
        The start time of the signal (that is, the time at the first
        sample of the signal). Default is None.

    See Also
    --------
    Signal
    RadioSignal
    IntensitySignal

    References
    ----------
    .. [1] Wikipedia, "Stokes Parameters",
           https://en.wikipedia.org/wiki/Stokes_parameters
    """
    _shape = (None, None, 4)


class DualPolarizationSignal(BasebandSignal):
    """Class for dual-polarization complex baseband signals.

    See the documentation for `~RadioSignal` and `~BasebandSignal` for
    more details.

    Parameters
    ----------
    z : `~numpy.ndarray`-like
        The signal data. Must be at least 3-dimensional with shape
        `(nsample, nchan, npol, ...)` where `npol = 2`.
    sample_rate : `~astropy.units.Quantity`
        The number of samples per second. Must be in units of frequency.
    center_freq : `~astropy.units.Quantity`
        The observing frequency at the center of the signal's band. Must
        be in units of frequency.
    bandwidth : `~astropy.units.Quantity`
        The total bandwidth of the signal. The channel bandwidth is this
        total bandwidth divided by the number of channels. Must be in
        units of frequency.
    pol_type : {'linear', 'circular'}
        The polarization type of the signal. Accepted values are
        'linear' for linearly polarized signals (with basis `[X, Y]`)
        or 'circular' for circularly polarized signals (with basis
        `[R, L]`).
    start_time : `~astropy.time.Time`, optional
        The start time of the signal (that is, the time at the first
        sample of the signal). Default is None.

    See Also
    --------
    Signal
    RadioSignal
    BasebandSignal

    Notes
    -----
    Linear polarization is assumed to have basis `[X, Y]` and circular
    polarization is assumed to have basis `[R, L]`. For example, with
    `pol_type='circular'`, `z[:, :, 0]` refers to the right-handed
    circular polarization component.
    """
    _shape = (None, None, 2)

    def __init__(self, z, /, *, sample_rate, center_freq, bandwidth, pol_type,
                 start_time=None):
        self.pol_type = pol_type
        super().__init__(z, sample_rate=sample_rate, center_freq=center_freq,
                         bandwidth=bandwidth, start_time=start_time)

    def __str__(self):
        s = super().__str__() + "\n"
        s += f"Polarization Type: {self.pol_type}"
        return s

    @property
    def pol_type(self):
        return self._pol_type

    @pol_type.setter
    def pol_type(self, pol_type):
        if pol_type in ['linear', 'circular']:
            self._pol_type = pol_type
        else:
            raise ValueError("pol_type must be in {'linear', 'circular'}")

    def to_linear(self):
        """Converts the dual-polarization signal to linear basis.

        If polarization basis is already linear, simply returns a copy of
        the signal object.

        Returns
        -------
        out : `~pulsarbat.DualPolarizationSignal`
            The converted signal.
        """
        if self.pol_type == 'circular':
            R, L = self.data[:, :, 0], self.data[:, :, 1]
            X, Y = R + L, 1j * (R - L)
            z = np.stack([X, Y], axis=2) / np.sqrt(2)
        else:
            z = self.data

        return type(self).like(self, z, pol_type='linear')

    def to_circular(self):
        """Converts the dual-polarization signal to circular basis.

        If polarization basis is already circular, simply returns a copy of
        the signal object.

        Returns
        -------
        out : `~pulsarbat.DualPolarizationSignal`
            The converted signal.
        """
        if self.pol_type == 'linear':
            X, Y = self.data[:, :, 0], self.data[:, :, 1]
            R, L = X - 1j * Y, X + 1j * Y
            z = np.stack([R, L], axis=2) / np.sqrt(2)
        else:
            z = self.data

        return type(self).like(self, z, pol_type='circular')

    def to_stokes(self):
        """Converts signal to IQUV Stokes representation.

        Returns
        -------
        out : `~pulsarbat.StokesSignal`
            Signal in Stokes IQUV representation.

        References
        ----------
        .. [1] Wikipedia, "Stokes Parameters",
               https://en.wikipedia.org/wiki/Stokes_parameters
        """
        if self.pol_type == 'linear':
            X, Y = self.data[:, :, 0], self.data[:, :, 1]

            i = X.real**2 + X.imag**2 + Y.real**2 + Y.imag**2
            Q = X.real**2 + X.imag**2 - Y.real**2 - Y.imag**2
            U = +2 * (X * Y.conj()).real
            V = -2 * (X * Y.conj()).imag

        elif self.pol_type == 'circular':
            R, L = self.data[:, :, 0], self.data[:, :, 1]

            i = R.real**2 + R.imag**2 + L.real**2 + L.imag**2
            Q = +2 * (R * L.conj()).real
            U = -2 * (R * L.conj()).imag
            V = R.real**2 + R.imag**2 - L.real**2 - L.imag**2

        z = np.stack([i, Q, U, V], axis=2)
        return FullStokesSignal.like(self, z)


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
        samples = np.ceil(samples) if samples > 0 else np.floor(samples)
        return samples.astype(np.int64)

    def phase_delay(self, f, ref_freq):
        coeff = self.dispersion_constant * self
        phase = coeff * f * u.cycle * (1 / ref_freq - 1 / f)**2
        return phase.to_value(u.rad)

    def transfer_function(self, f, ref_freq):
        """Returns the transfer function for dedispersion."""
        transfer = np.exp(1j * self.phase_delay(f, ref_freq))
        return transfer.astype(np.complex64)
