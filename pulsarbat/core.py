"""Core module consisting of basic building blocks of pulsarbat."""

import inspect
import numpy as np
import astropy.units as u
from astropy.time import Time
from itertools import zip_longest

__all__ = [
    'Signal',
    'RadioSignal',
    'IntensitySignal',
    'BasebandSignal',
    'FullStokesSignal',
    'DualPolarizationSignal',
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

    The signal data (`z`) must have at least 1 dimension where the
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
        try:
            self._sample_rate = sample_rate.to(u.MHz)
            assert self._sample_rate.isscalar
        except Exception:
            err = ("Invalid sample_rate! Must be a scalar astropy "
                   "Quantity with units of Hz or equivalent.")
            raise ValueError(err)

        self._start_time = None
        if start_time is not None:
            try:
                self._start_time = Time(start_time, format='isot', precision=9)
                assert self._start_time.isscalar
            except Exception:
                raise ValueError('Invalid start_time!')

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

        self._data = z.astype(self._dtype or z.dtype, copy=False)
        self._verification_checks()

    def _verification_checks(self):
        pass

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
        return len(self.data)

    def __array__(self):
        return np.asanyarray(self.data)

    def _time_slice(self, index):
        s = slice(*index.indices(self.shape[0]))
        assert s.step > 0, "Time axis slicing does not support negative step"
        assert s.stop > s.start, "Empty time slice!"

        kw = dict()
        if s.step > 1:
            kw['sample_rate'] = self.sample_rate / s.step
        if self.start_time is not None:
            kw['start_time'] = self.start_time + s.start / self.sample_rate
        return kw

    def __getitem__(self, index):
        if not isinstance(index, tuple):
            index = (index,)

        if not all(isinstance(a, slice) for a in index[:1]):
            err = "Only supports slicing on time axis."
            raise IndexError(err)

        kw = dict()
        kw.update(self._time_slice(index[0]))
        return type(self).like(self, self.data[index], **kw)

    @property
    def data(self):
        """The signal data."""
        return self._data

    @property
    def shape(self):
        """Shape of the signal."""
        return self.data.shape

    @property
    def sample_shape(self):
        """Shape of a sample."""
        return self.shape[1:]

    @property
    def ndim(self):
        """Number of dimensions in data."""
        return self.data.ndim

    @property
    def dtype(self):
        """Data type of the signal."""
        return self.data.dtype

    @property
    def sample_rate(self):
        """Sample rate of the signal."""
        return self._sample_rate

    @property
    def dt(self):
        """Sample spacing of the signal (1 / sample_rate)."""
        return (1 / self.sample_rate).to(u.s)

    @property
    def time_length(self):
        """Length of signal in time units."""
        return (len(self) / self.sample_rate).to(u.s)

    @property
    def start_time(self):
        """Start time of the signal (Time at first sample)."""
        return self._start_time

    @property
    def stop_time(self):
        """Stop time of the signal (Time at sample after the last sample)."""
        if self.start_time is None:
            return None
        return self.start_time + self.time_length

    def __contains__(self, time):
        """Tell if `time` is within the bounds of the signal."""
        if self.start_time is None:
            return False
        return self.start_time <= time + 0.1 * u.ns < self.stop_time

    @classmethod
    def like(cls, obj, z=None, /, **kwargs):
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
                    raise ValueError(f'Missing required keyword argument: {k}')

        if z is None:
            z = obj.data
        return cls(z, **kwargs)


class RadioSignal(Signal):
    """Class for heterodyned radio signals.

    A radio signal is often heterodyned, so a center frequency
    (`center_freq`) and a bandwidth (`bandwidth`) must be provided to
    indicate the "true frequency" at the center of the band. Optionally,
    a `freq_align` is provided to indicate how the "channel frequencies"
    are positioned. Usually, the signal data here would be the result of
    some sort of filterbank.

    The signal data (`z`) must have at least 2 dimensions where the
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
    freq_align : {'bottom', 'center', 'top'}, optional
        The alignment of channels relative to the `center_freq`. Default
        is `'center'` (as with odd-length complex DFTs). `'bottom'` and
        `'top'` only have an effect when `nchan` is even.
    start_time : `~astropy.time.Time`, optional
        The start time of the signal (that is, the time at the first
        sample of the signal). Default is None.

    Notes
    -----
    The input signal array must have shape `(nsample, nchan, ...)`,
    where `nsample` is the number of time samples, and `nchan` is the
    number of frequency channels.

    The channels must be adjacent in frequency and of equal channel
    bandwidth, such that the frequency of a channel `i` is given by,::

        freq_i = center_freq + (bandwidth / nchan) * (i + a - nchan/2)

    where `i` is in `[0, ..., nchan - 1]`, `nchan` is the number of
    channels (`z.shape[1]`), `bandwidth / nchan` is the bandwidth of a
    single channel, and `a` is in `{0, 0.5, 1}` depending on the value
    of `freq_align`.

    An even-length complex-valued DFT (as implemented in `~numpy.fft`)
    would have `freq_align = 'bottom'` with a center frequency of 0,
    whereas an odd-length DFT would have `freq_align = 'center'` due to
    symmetry. `freq_align = 'top'` is used in cases where the DFT was
    conducted on the lower sideband of the signal instead.

    When `nchan` is odd, the `freq_align` attribute will internally be
    fixed to `'center'` since the channels must be necessarily centered
    on the `center_freq`.
    """
    _shape = (None, None, )

    def __init__(self, z, /, *, sample_rate, center_freq, bandwidth,
                 freq_align='bottom', start_time=None):

        try:
            self._center_freq = center_freq.to(u.MHz)
            assert self._center_freq.isscalar
        except Exception:
            err = ("Invalid center_freq! Must be a scalar astropy "
                   "Quantity with units of Hz or equivalent.")
            raise ValueError(err)

        try:
            self._bandwidth = bandwidth.to(u.MHz)
            assert self._bandwidth.isscalar
        except Exception:
            err = ("Invalid bandwidth! Must be a scalar astropy "
                   "Quantity with units of Hz or equivalent.")
            raise ValueError(err)

        super().__init__(z, sample_rate=sample_rate, start_time=start_time)

        if freq_align in ['bottom', 'center', 'top']:
            if self.nchan % 2:
                self._freq_align = 'center'
            else:
                self._freq_align = freq_align
        else:
            choices = "{'bottom', 'center', 'top'}"
            raise ValueError(f'Invalid freq_align. Expected: {choices}')

    def __str__(self):
        s = super().__str__() + "\n"
        s += f"Total Bandwidth: {self.bandwidth}\n"
        s += f"Center Frequency: {self.center_freq}"
        return s

    def _freq_slice(self, index):
        s = slice(*index.indices(self.shape[1]))
        assert s.step == 1, "Does not support slice step for frequency axis"
        assert s.stop > s.start, "Empty frequency slice!"
        kw = dict()
        if s.stop - s.start < self.nchan:
            f = self.channel_freqs[s]
            kw['bandwidth'] = self.chan_bandwidth * (s.stop - s.start)
            kw['center_freq'] = (f[0] + f[-1]) / 2
            kw['freq_align'] = 'center'
        return kw

    def __getitem__(self, index):
        if not isinstance(index, tuple):
            index = (index,)

        if not all(isinstance(a, slice) for a in index[:2]):
            err = "Only supports slicing on time and frequency axes."
            raise IndexError(err)

        kw = dict()
        kw.update(self._time_slice(index[0]))
        if len(index) > 1:
            kw.update(self._freq_slice(index[1]))
        return type(self).like(self, self.data[index], **kw)

    @property
    def nchan(self):
        """Number of frequency channels."""
        return self.shape[1]

    @property
    def center_freq(self):
        """Center observing frequency of the signal."""
        return self._center_freq

    @property
    def bandwidth(self):
        """Total bandwidth of signal."""
        return self._bandwidth

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
    def freq_align(self):
        """Alignment of channel frequencies."""
        return self._freq_align

    @property
    def channel_freqs(self):
        """Returns a list of frequencies corresponding to all channels."""
        _align = {'bottom': 0, 'center': 0.5, 'top': 1}[self.freq_align]
        chan_ids = np.arange(self.nchan) + _align - self.nchan / 2
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
    freq_align : {'bottom', 'center', 'top'}, optional
        The alignment of frequencies relative to channels. Default is
        `'bottom'` (as with even-length FFTs).
    start_time : `~astropy.time.Time`, optional
        The start time of the signal (that is, the time at the first
        sample of the signal). Default is None.

    See Also
    --------
    Signal
    RadioSignal

    Notes
    -----
    Signal data is converted to and stored with `dtype=np.float32`, as
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
    freq_align : {'bottom', 'center', 'top'}, optional
        The alignment of frequencies relative to channels. Default is
        `'bottom'` (as with even-length FFTs).
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
    converted to and stored with `dtype=np.complex64` as baseband
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
        out : `~pulsarbat.IntensitySignal`
            The converted signal.
        """
        z = self.data.real**2 + self.data.imag**2
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
    freq_align : {'bottom', 'center', 'top'}, optional
        The alignment of frequencies relative to channels. Default is
        `'bottom'` (as with even-length FFTs).
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
    Wikipedia, "Stokes Parameters",
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
    freq_align : {'bottom', 'center', 'top'}, optional
        The alignment of frequencies relative to channels. Default is
        `'bottom'` (as with even-length FFTs).
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

    def __init__(self, z, /, *, sample_rate, center_freq, bandwidth,
                 freq_align='bottom', pol_type, start_time=None, ):
        self.pol_type = pol_type
        super().__init__(z, sample_rate=sample_rate, center_freq=center_freq,
                         bandwidth=bandwidth, freq_align=freq_align,
                         start_time=start_time)

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
