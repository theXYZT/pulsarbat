"""Core module defining Signal classes."""

import inspect
import pprint
import numpy as np
import astropy.units as u
from astropy.time import Time
from numpy.core.overrides import set_module

__all__ = [
    "Signal",
    "RadioSignal",
    "IntensitySignal",
    "FullStokesSignal",
    "BasebandSignal",
    "DualPolarizationSignal",
]


class InvalidSignalError(ValueError):
    """Used to catch invalid signals."""

    pass


@set_module("pulsarbat")
class Signal:
    """Base class for all signals.

    A signal is sufficiently described by an array of samples (`z`),
    and a sampling rate (`sample_rate`). Optionally, a `start_time` can
    be provided to specify the time stamp at the first sample of the
    signal.

    The signal data (`z`) must have at least 1 dimension where the
    zeroth axis (`axis=0`) refers to time. That is, `z[i]` is the `i`-th
    sample of the signal, and the sample shape (`z[i].shape`) can be
    arbitrary.

    Parameters
    ----------
    z : array
        The signal data. Must be at least 1-dimensional with shape
        `(nsample, ...)`, and must have non-zero size.
    sample_rate : :py:class:`astropy.units.Quantity`
        The number of samples per second. Must be in units of frequency.
    start_time : :py:class:`astropy.time.Time`, optional
        The start time of the signal (that is, the time at the first
        sample of the signal). Default is None.
    meta : dict, optional
        Any metadata that the user might want to attach to the signal.
    """

    _req_dtype = ()
    _req_shape = (None,)

    def __init__(self, z, /, *, sample_rate, start_time=None, meta=None):
        min_ndim = len(self._req_shape)
        if z.ndim < min_ndim:
            err = (f"Expected signal with at least {min_ndim} dimension(s), "
                   f"got signal with {z.ndim} dimension(s) instead.")
            raise InvalidSignalError(err)

        zipped = zip(z.shape[:min_ndim], self._req_shape)
        if not all(x == (y or x) for x, y in zipped):
            err = (f"Signal has invalid shape. Expected {self._req_shape}, "
                   f"got {z.shape} instead.")
            raise InvalidSignalError(err)

        if np.prod(z.shape[1:]) == 0:
            raise InvalidSignalError("Sample shape must have non-zero size!")

        _temp = None
        if z.dtype in self._req_dtype or not self._req_dtype:
            _temp = z
        else:
            for d in self._req_dtype:
                try:
                    _temp = z.astype(self._req_dtype[0], casting="safe")
                except TypeError:
                    pass
                else:
                    break

        if _temp is None:
            err = f"Expected {self._req_dtype}, got {z.dtype}."
            raise InvalidSignalError(f"Invalid dtype. {err}")

        self._data = _temp
        self.sample_rate = sample_rate
        self.start_time = start_time
        self.meta = meta

    def _attr_repr(self):
        st = "N/A" if self.start_time is None else self.start_time.isot
        return (f"Sample rate: {self.sample_rate}\n"
                f"Time length: {self.time_length}\n"
                f"Start time: {st}\n")

    def __str__(self):
        signature = f"{self.__class__.__name__} @ {hex(id(self))}"
        c = type(self.data)
        s = f"{signature}\n{'-' * len(signature)}\n"
        s += f"Data Container: {c.__module__}.{c.__name__}"
        s += f"<shape={self.shape}, dtype={self.dtype}>\n"
        s += self._attr_repr()
        if self.meta is not None:
            s += "\nMeta\n----\n"
            s += pprint.pformat(self.meta, sort_dicts=False, depth=2)
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

        kw = dict()
        if s.step > 1:
            kw["sample_rate"] = self.sample_rate / s.step
        if self.start_time is not None:
            kw["start_time"] = self.start_time + s.start / self.sample_rate
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
    def meta(self):
        """Signal metadata."""
        return self._meta

    @meta.setter
    def meta(self, meta):
        try:
            self._meta = None if meta is None else dict(meta)
        except Exception:
            raise ValueError("meta must be a dict.")

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

    @sample_rate.setter
    def sample_rate(self, sample_rate):
        try:
            temp = sample_rate.to(u.Hz)
            assert temp.isscalar and temp > 0
        except Exception:
            raise ValueError("Invalid sample_rate. Must be a positive scalar "
                             "Quantity with units of Hz or equivalent.")
        else:
            self._sample_rate = sample_rate

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

    @start_time.setter
    def start_time(self, start_time):
        try:
            temp = None
            if start_time is not None:
                temp = Time(start_time, format="isot", precision=9)
                assert temp.isscalar
        except Exception:
            err = "Invalid start_time. Must be a scalar astropy Time object."
            raise ValueError(err)
        else:
            self._start_time = temp

    @property
    def stop_time(self):
        """Stop time of the signal (Time at sample after the last sample)."""
        if self.start_time is None:
            return None
        return self.start_time + self.time_length

    def contains(self, t, /):
        """Whether time(s) are within the bounds of the signal."""
        if self.start_time is None:
            return np.zeros(t.shape, bool) if t.shape else False

        t0, t1 = self.start_time, self.stop_time
        edge = ~Time.isclose(t, t1) | Time.isclose(t, t0)
        return edge & (t0 <= t) & (t < t1)

    def __contains__(self, t):
        """Whether time is within the bounds of the signal."""
        return self.contains(t)

    def compute(self):
        """Returns a signal with data as a numpy.ndarray (or subclass).

        If the data is contained in a dask array, then this will compute
        it as a consequence.
        """
        return type(self).like(self, np.asanyarray(self))

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
        obj : `Signal`
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
                    raise ValueError(f"Missing required keyword argument: {k}")

        if z is None:
            z = obj.data
        return cls(z, **kwargs)


@set_module("pulsarbat")
class RadioSignal(Signal):
    """Class for heterodyned radio signals.

    A radio signal is often heterodyned, so a center frequency
    (`center_freq`) and a channel bandwidth (`chan_bw`) must be provided
    to determine the frequencies of each channel. Optionally,
    `freq_align` is provided to indicate how the channel frequencies are
    positioned. Usually, the signal data here would be the result of
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
    start_time : `~astropy.time.Time`, optional
        The start time of the signal (that is, the time at the first
        sample of the signal). Default is None.
    center_freq : `~astropy.units.Quantity`
        The frequency at the center of the signal's band. Must be in
        units of frequency.
    chan_bw : `~astropy.units.Quantity`
        The bandwidth of a channel. The total bandwidth is `chan_bw *
        nchan`. Must be in units of frequency.
    freq_align : {'bottom', 'center', 'top'}, optional
        The alignment of channels relative to the `center_freq`. Default
        is `'center'` (as with odd-length complex DFTs). `'bottom'` and
        `'top'` only have an effect when `nchan` is even.
    meta : dict, optional
        Any metadata that the user might want to attach to the signal.

    Notes
    -----
    The input signal array must have shape `(nsample, nchan, ...)`,
    where `nsample` is the number of time samples, and `nchan` is the
    number of frequency channels.

    The channels must be adjacent in frequency and of equal channel
    bandwidth, such that the frequency of a channel `i` is given by,::

        freq_i = center_freq + chan_bw * (i + a - nchan/2)

    where `i` is in `[0, ..., nchan - 1]`, `nchan` is the number of
    channels (`z.shape[1]`), `chan_bw` is the bandwidth of a single
    channel, and `a` is in `{0, 0.5, 1}` depending on the value of
    `freq_align`.

    An even-length complex-valued DFT (as implemented in `~numpy.fft`)
    would have `freq_align = 'bottom'` with a center frequency of 0,
    whereas an odd-length DFT would have `freq_align = 'center'` due to
    symmetry. `freq_align = 'top'` is used in cases where the DFT was
    conducted on the lower sideband of the signal instead.

    When `nchan` is odd, the `freq_align` attribute will internally be
    fixed to `'center'` since the channels must be necessarily centered
    on the `center_freq`.
    """

    _req_shape = (None, None,)

    def __init__(self, z, /, *, sample_rate, start_time=None, center_freq,
                 chan_bw, freq_align="center", meta=None):

        super().__init__(z, sample_rate=sample_rate, start_time=start_time,
                         meta=meta)

        self.center_freq = center_freq
        self.chan_bw = chan_bw
        self.freq_align = freq_align

    def _attr_repr(self):
        s = super()._attr_repr()
        s += f"Channel Bandwidth: {self.chan_bw}\n"
        s += f"Total Bandwidth: {self.bandwidth}\n"
        s += f"Center Frequency: {self.center_freq}\n"
        return s

    def _freq_slice(self, index):
        s = slice(*index.indices(self.shape[1]))
        assert s.step == 1, "Does not support slice step for frequency axis"
        assert s.stop > s.start, "Empty frequency slice!"
        f = self.channel_freqs[s]
        return {"center_freq": (f[0] + f[-1])/2, "freq_align": "center"}

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
        """Center frequency."""
        return self._center_freq

    @center_freq.setter
    def center_freq(self, center_freq):
        try:
            temp = center_freq.to(u.Hz)
            assert temp.isscalar
        except Exception:
            raise ValueError("Invalid center_freq. Must be a scalar "
                             "Quantity with units of Hz or equivalent.")
        else:
            self._center_freq = center_freq

    @property
    def bandwidth(self):
        """Total bandwidth."""
        return self.chan_bw * self.nchan

    @property
    def chan_bw(self):
        """Channel bandwidth."""
        return self._chan_bw

    @chan_bw.setter
    def chan_bw(self, chan_bw):
        try:
            temp = chan_bw.to(u.Hz)
            assert temp.isscalar and temp > 0
        except Exception:
            raise ValueError("Invalid chan_bw. Must be a positive scalar "
                             "Quantity with units of Hz or equivalent.")
        else:
            self._chan_bw = chan_bw

    @property
    def max_freq(self):
        """Frequency at top of the band."""
        return self.center_freq + self.bandwidth / 2

    @property
    def min_freq(self):
        """Frequency at bottom of the band."""
        return self.center_freq - self.bandwidth / 2

    @property
    def freq_align(self):
        """Alignment of channel frequencies."""
        return self._freq_align

    @freq_align.setter
    def freq_align(self, freq_align):
        if freq_align in {"bottom", "center", "top"}:
            self._freq_align = "center" if self.nchan % 2 else freq_align
        else:
            choices = "{'bottom', 'center', 'top'}"
            raise ValueError(f"Invalid freq_align. Expected: {choices}")

    @property
    def channel_freqs(self):
        """Returns a list of frequencies corresponding to all channels."""
        _align = {"bottom": 0, "center": 0.5, "top": 1}[self.freq_align]
        chan_ids = np.arange(self.nchan) + _align - self.nchan / 2
        return self.center_freq + self.chan_bw * chan_ids


@set_module("pulsarbat")
class IntensitySignal(RadioSignal):
    """Class for intensity signals.

    See the documentation for `~RadioSignal` for more details.

    Parameters
    ----------
    z : `~numpy.ndarray`-like
        The signal data. Must be at least 3-dimensional with shape
        `(nsample, nchan, npol, ...)`.
    sample_rate : `~astropy.units.Quantity`
        The number of samples per second. Must be in units of frequency.
    start_time : `~astropy.time.Time`, optional
        The start time of the signal (that is, the time at the first
        sample of the signal). Default is None.
    center_freq : `~astropy.units.Quantity`
        The frequency at the center of the signal's band. Must be in
        units of frequency.
    chan_bw : `~astropy.units.Quantity`
        The bandwidth of a channel. The total bandwidth is `chan_bw *
        nchan`. Must be in units of frequency.
    freq_align : {'bottom', 'center', 'top'}, optional
        The alignment of channels relative to the `center_freq`. Default
        is `'center'` (as with odd-length complex DFTs). `'bottom'` and
        `'top'` only have an effect when `nchan` is even.
    meta : dict, optional
        Any metadata that the user might want to attach to the signal.

    See Also
    --------
    Signal
    RadioSignal
    """

    _req_dtype = (np.float64, np.float32)


@set_module("pulsarbat")
class FullStokesSignal(IntensitySignal):
    """Class for full Stokes (I, Q, U, V) signals.

    See the documentation for `~RadioSignal` and `~IntensitySignal` for
    more details.

    Parameters
    ----------
    z : `~numpy.ndarray`-like
        The signal data. Must be at least 3-dimensional with shape
        `(nsample, nchan, nstokes, ...)` where `nstokes = 4`.
        The order of the Stokes components is `[I, Q, U, V]`.
    sample_rate : `~astropy.units.Quantity`
        The number of samples per second. Must be in units of frequency.
    start_time : `~astropy.time.Time`, optional
        The start time of the signal (that is, the time at the first
        sample of the signal). Default is None.
    center_freq : `~astropy.units.Quantity`
        The frequency at the center of the signal's band. Must be in
        units of frequency.
    chan_bw : `~astropy.units.Quantity`
        The bandwidth of a channel. The total bandwidth is `chan_bw *
        nchan`. Must be in units of frequency.
    freq_align : {'bottom', 'center', 'top'}, optional
        The alignment of channels relative to the `center_freq`. Default
        is `'center'` (as with odd-length complex DFTs). `'bottom'` and
        `'top'` only have an effect when `nchan` is even.
    meta : dict, optional
        Any metadata that the user might want to attach to the signal.

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

    _req_shape = (None, None, 4)


@set_module("pulsarbat")
class BasebandSignal(RadioSignal):
    """Class for complex baseband signals.

    Baseband signals are Nyquist-sampled analytic signals. In the
    complex baseband representation, the signal is shifted to be
    centered around zero frequency and is a complex-valued signal. As a
    consequence, `sample_rate == chan_bw`. All frequency channels are
    assumed to be upper side-band (that is, the signal isn't spectrally
    flipped).

    See the documentation for `~RadioSignal` for more details.

    Parameters
    ----------
    z : `~numpy.ndarray`-like
        The signal data. Must be at least 2-dimensional with shape
        `(nsample, nchan, ...)`.
    sample_rate : `~astropy.units.Quantity`
        The number of samples per second. Must be in units of frequency.
    start_time : `~astropy.time.Time`, optional
        The start time of the signal (that is, the time at the first
        sample of the signal). Default is None.
    center_freq : `~astropy.units.Quantity`
        The frequency at the center of the signal's band. Must be in
        units of frequency.
    freq_align : {'bottom', 'center', 'top'}, optional
        The alignment of channels relative to the `center_freq`. Default
        is `'center'` (as with odd-length complex DFTs). `'bottom'` and
        `'top'` only have an effect when `nchan` is even.
    meta : dict, optional
        Any metadata that the user might want to attach to the signal.

    See Also
    --------
    Signal
    RadioSignal
    """

    _req_dtype = (np.complex128, np.complex64)

    def __init__(self, z, /, *, sample_rate, start_time=None, center_freq,
                 freq_align="center", meta=None):

        super().__init__(z, sample_rate=sample_rate, start_time=start_time,
                         center_freq=center_freq, chan_bw=sample_rate,
                         freq_align=freq_align, meta=None)

    def to_intensity(self):
        """Converts baseband signal to intensities.

        Returns
        -------
        out : `~pulsarbat.IntensitySignal`
            The converted signal.
        """
        z = self.data.real**2 + self.data.imag**2
        return IntensitySignal.like(self, z)


@set_module("pulsarbat")
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
    start_time : `~astropy.time.Time`, optional
        The start time of the signal (that is, the time at the first
        sample of the signal). Default is None.
    center_freq : `~astropy.units.Quantity`
        The frequency at the center of the signal's band. Must be in
        units of frequency.
    freq_align : {'bottom', 'center', 'top'}, optional
        The alignment of channels relative to the `center_freq`. Default
        is `'center'` (as with odd-length complex DFTs). `'bottom'` and
        `'top'` only have an effect when `nchan` is even.
    pol_type : {'linear', 'circular'}
        The polarization type of the signal. `'linear'` for linearly
        polarized signals (with basis `[X, Y]`) or `'circular'` for
        circularly polarized signals (with basis `[R, L]`).
    meta : dict, optional
        Any metadata that the user might want to attach to the signal.

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

    _req_shape = (None, None, 2)

    def __init__(self, z, /, *, sample_rate, start_time=None, center_freq,
                 freq_align="center", pol_type, meta=None):

        super().__init__(z, sample_rate=sample_rate, start_time=start_time,
                         center_freq=center_freq, freq_align=freq_align,
                         meta=meta)

        self.pol_type = pol_type

    def _attr_repr(self):
        s = super()._attr_repr()
        s += f"Polarization Type: {self.pol_type}\n"
        return s

    @property
    def pol_type(self):
        """Polarization type (linear or circular)."""
        return self._pol_type

    @pol_type.setter
    def pol_type(self, pol_type):
        if pol_type in {"linear", "circular"}:
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
        if self.pol_type == "circular":
            R, L = self.data[:, :, 0], self.data[:, :, 1]
            X, Y = R + L, 1j * (R - L)
            z = np.stack([X, Y], axis=2) / np.sqrt(2)
        else:
            z = self.data

        return type(self).like(self, z, pol_type="linear")

    def to_circular(self):
        """Converts the dual-polarization signal to circular basis.

        If polarization basis is already circular, simply returns a copy of
        the signal object.

        Returns
        -------
        out : `~pulsarbat.DualPolarizationSignal`
            The converted signal.
        """
        if self.pol_type == "linear":
            X, Y = self.data[:, :, 0], self.data[:, :, 1]
            R, L = X - 1j * Y, X + 1j * Y
            z = np.stack([R, L], axis=2) / np.sqrt(2)
        else:
            z = self.data

        return type(self).like(self, z, pol_type="circular")

    def to_stokes(self):
        """Converts signal to IQUV Stokes representation.

        Returns
        -------
        out : `~pulsarbat.StokesSignal`
            Signal in Stokes IQUV representation.
        """
        if self.pol_type == "linear":
            X, Y = self.data[:, :, 0], self.data[:, :, 1]

            i = X.real**2 + X.imag**2 + Y.real**2 + Y.imag**2
            Q = X.real**2 + X.imag**2 - Y.real**2 - Y.imag**2
            U = +2 * (X * Y.conj()).real
            V = -2 * (X * Y.conj()).imag

        elif self.pol_type == "circular":
            R, L = self.data[:, :, 0], self.data[:, :, 1]

            i = R.real**2 + R.imag**2 + L.real**2 + L.imag**2
            Q = +2 * (R * L.conj()).real
            U = -2 * (R * L.conj()).imag
            V = R.real**2 + R.imag**2 - L.real**2 - L.imag**2

        z = np.stack([i, Q, U, V], axis=2)
        return FullStokesSignal.like(self, z)
