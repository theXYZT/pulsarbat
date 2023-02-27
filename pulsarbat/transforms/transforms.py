"""Core signal transforms."""

import operator
import numpy as np
import astropy.units as u
from astropy.time import Time
import pulsarbat as pb
import functools
import dask.array as da

__all__ = [
    "signal_transform",
    "concatenate",
    "snippet",
    "time_shift",
    "freq_shift",
    "fast_len",
]


def signal_transform(func):
    """Wraps an array function and returns a signal transform.

    The function being decorated must accept an array and return an array
    with function signature::

        func(x, *args, /, **kwargs)

    where ``x`` is the array being transformed. The decorated function will
    accept a Signal object in place of ``x`` and by default return a Signal object
    of the same type (unless a different ``signal_type`` is specified). The
    properties of the returned signal can be modified via ``signal_kwargs`` if needed.

    If the Signal data is contained within a Dask array, ``func`` will be applied
    to every chunk independently using :py:func:`dask.array.map_blocks`.
    Keyword arguments specific to :py:func:`~dask.array.map_blocks` can be passed
    via ``dask_kwargs``.
    """

    @functools.wraps(func)
    def wrapper(
        x, *args, signal_type=None, signal_kwargs=dict(), dask_kwargs=dict(), **kwargs
    ):

        sig_class = type(x) if signal_type is None else signal_type
        if not issubclass(sig_class, pb.Signal):
            raise TypeError("Signal type must be a subclass of pulsarbat.Signal!")

        if isinstance(x.data, da.Array):
            z = da.map_blocks(func, x.data, **dask_kwargs, **kwargs)
        else:
            z = func(x.data, **kwargs)

        return sig_class.like(x, z, **signal_kwargs)

    return wrapper


def concatenate(signals, /, axis=0):
    """Concatenates multiple signals along given axis.

    Signals must be contiguous along the axis of concatenation. The
    concatenated signal will inherit attributes from given ``kwargs`` and
    then from the first signal in the sequence, ``signals[0]``, except for
    ``center_freq`` and ``freq_align`` when concatenating along frequency
    (which must be computed accordingly).

    Parameters
    ----------
    signals : sequence of Signal
        Sequence of signals to concatenate. All signals must be Signal
        objects and have the same type and ``sample_rate``. If concatenating
        along frequency, then they must also have the same ``chan_bw``.
    axis : int or 'time' or 'freq', optional
        Axis along which to concatenate signals. Default is 0. ``time``
        is an alias for 0 (concatenating along time). ``freq`` implies
        ``axis=1`` (concatenating along frequency) and requires that
        signals are instances of RadioSignal.

    Returns
    -------
    Signal
        Concatenated signal of same type as input signals.
    """
    try:
        sig_type = type(signals[0])
    except IndexError:
        raise ValueError("Need at least one signal to concatenate.")
    else:
        if not issubclass(sig_type, pb.Signal):
            raise TypeError("Signals must be pulsarbat.Signal objects.")

        if not all(type(s) == sig_type for s in signals):
            raise TypeError("All signals must have same type!")

    ref_sr = signals[0].sample_rate
    if not all(u.isclose(ref_sr, s.sample_rate) for s in signals):
        raise ValueError("Signals must have the same sample_rate!")

    ref_st = None
    if axis in {0, "time"}:
        n = 0
        for s in signals:
            if s.start_time is not None:
                if ref_st is None:
                    ref_st = s.start_time - (n / ref_sr)
                elif not Time.isclose(ref_st + (n / ref_sr), s.start_time):
                    raise ValueError("Signals not contiguous in time.")
            n += len(s)
        axis = 0
    else:
        for s in signals:
            if s.start_time is not None:
                if ref_st is None:
                    ref_st = s.start_time
                elif not Time.isclose(ref_st, s.start_time):
                    raise ValueError("Signals have different start_time.")

    kw = {"start_time": ref_st}

    if isinstance(signals[0], pb.RadioSignal):
        ref_cbw = signals[0].chan_bw
        if not all(u.isclose(ref_cbw, s.chan_bw) for s in signals):
            raise ValueError("RadioSignals must have the same chan_bw!")

        if axis in {1, "freq"}:
            for x, y in zip(signals, signals[1:]):
                chan_diff = y.channel_freqs[0] - x.channel_freqs[-1]
                if not u.isclose(chan_diff, ref_cbw):
                    raise ValueError("Signals not contiguous in frequency.")

            f0, f1 = signals[0].channel_freqs[0], signals[-1].channel_freqs[-1]
            axis = 1
        else:
            ref_cfs = signals[0].channel_freqs
            if not all(u.allclose(ref_cfs, s.channel_freqs) for s in signals):
                raise ValueError("Signals have different frequency channels.")
            f0, f1 = ref_cfs[0], ref_cfs[-1]

        kw["center_freq"] = (f0 + f1) / 2
        kw["freq_align"] = "center"

    elif axis == "freq":
        err = "Signals must be pb.RadioSignal objects when axis is 'freq'."
        raise TypeError(err)

    z = np.concatenate([s.data for s in signals], axis=axis)
    return sig_type.like(signals[0], z, **kw)


def snippet(z, /, t, n):
    """Extracts a snippet of a signal in time.

    If ``t`` corresponds to non-integer number of samples from the
    start of ``z``, time-shifting via FFT (by applying a phase gradient
    in the Fourier domain) is used. This usually only makes sense if
    ``z`` is a :py:class:`.BasebandSignal`. For non-baseband signals,
    the output might not be meaningful.

    Parameters
    ----------
    z : Signal
        Input signal.
    t : int, float, Quantity, or Time
        Start location of snippet. Given as either a number of
        samples (int or float) or a Quantity (units of time) relative
        to the start of the signal, or a Time object specifying the
        start time of the snippet.
    n : int
        Length of snippet in number of samples. Must be an integer.

    Returns
    -------
    Signal
        Snippet of ``z`` starting at ``t`` with length ``n``.

    Notes
    -----
    Since an FFT is used, it is efficient to provide a signal with a
    fast FFT length via :py:func:`pulsarbat.fast_len`.
    """
    if (n := operator.index(n)) < 0:
        raise ValueError("n must be a non-negative integer.")

    if isinstance(t, Time):
        if z.start_time is None:
            raise ValueError("t is a Time object, but signal has no start time.")

        t = (t - z.start_time).to(u.s)

    if isinstance(t, u.Quantity):
        t = (t * z.sample_rate).to_value(u.one)

    if (t < 0) or (len(z) < t + n):
        raise ValueError("Requested snippet goes out of bounds.")

    if (i := int(t)) < t:
        shift = i - t

        if z.start_time is None:
            new_start = None
        else:
            new_start = z.start_time - shift * z.dt

        shifted = pb.time_shift(z, shift, crop=True).data
        z = type(z).like(z, shifted, start_time=new_start)

    return z[i : i + n]


def time_shift(z, /, shift, crop=False):
    """Shift signal data by given number of samples or time.

    This function shifts the signal data in time via FFT by multiplying by
    a phase gradient in frequency domain. This usually only makes sense if
    ``z`` is a :py:class:`.BasebandSignal`. For non-baseband signals,
    the output might not be meaningful.

    Parameters
    ----------
    z : Signal
        Input signal.
    shift : int, float, array-like or Quantity
        Shift amount. If a number (int or float), the signal is shifted
        by that number of samples. An astropy Quantity with units of
        time can also be passed, in which case the signal will be
        shifted by `dt * z.sample_rate` samples. If an array, must have
        shape such that axes with length more than 1 match ``z.sample_shape``.
    crop : bool, optional
        Whether the returned signal is cropped to eliminate out-of-bounds
        data. Default is False.

    Returns
    -------
    out : Signal
        Shifted signal. If the ``crop`` parameter is ``False``, will have
        the same shape and ``start_time`` as input signal. If ``crop`` is
        ``True``, ``start_time`` will change by ``max(0, shift.max()) * z.dt``.

    Notes
    -----
    Since an FFT is used, it is efficient to provide a signal with a
    fast FFT length via :py:func:`pulsarbat.fast_len`.
    """
    if isinstance(shift, u.Quantity):
        shift = (shift * z.sample_rate).to_value(u.one)

    shift = np.array(shift)

    if shift.ndim >= z.ndim:
        raise ValueError(
            f"shift has too many dimensions. Expected <= {z.ndim - 1} dimensions, "
            f"got {shift.ndim} dimensions!"
        )

    # If shifts are zero, do nothing
    if np.allclose(shift, 0):
        return z

    if shift.ndim > 0:
        ix = (slice(None),) * shift.ndim + (None,) * (z.ndim - shift.ndim - 1)
        shift = shift[ix]

    f_ix = tuple(slice(None) if j == 0 else None for j in range(z.ndim))
    if isinstance(z.data, da.Array):
        f = da.fft.fftfreq(len(z), 1, chunks=(-1,))[f_ix]
    else:
        f = np.fft.fftfreq(len(z), 1)[f_ix]

    ph = np.exp(-2j * np.pi * shift * f).astype(np.complex64)
    shifted = pb.fft.ifft(pb.fft.fft(z.data, axis=0) * ph, axis=0)
    shifted = shifted if np.iscomplexobj(z.data) else shifted.real

    start, stop = 0, 0
    it = np.nditer(shift, flags=["multi_index"])
    for a in it:
        if a < 0:
            a = int(np.floor(a))
            ix = (np.s_[a:],) + it.multi_index
            stop = min(stop, a)
        else:
            a = int(np.ceil(a))
            ix = (np.s_[:a],) + it.multi_index
            start = max(start, a)

        shifted[ix] = 0

    x = type(z).like(z, shifted)

    if crop:
        x = x[start:len(x) + stop]

    return x


def freq_shift(z, /, shift):
    """Shift signal data in frequency by given amount.

    A frequency shift is achieved by mixing the signal with a sinusoid.
    The "out-of-band" portion of the signal is filled with zeros after
    the frequency shift is applied to prevent erroneous data from
    appearing in the wrong places due to wrap-around effects.

    Shifting by more than a channel bandwidth will not return an error,
    but a zero signal instead (since all the data shifted out of band).

    Parameters
    ----------
    z : BasebandSignal
        Input signal.
    shift : Quantity
        Shift amount in units of frequency. Should be a scalar or have
        shape that such that axes with length more than 1 match ``z.sample_shape``.

    Returns
    -------
    BasebandSignal
        Frequency-shifted signal.
    """
    if not isinstance(z, pb.BasebandSignal):
        raise TypeError("Signal must be a BasebandSignal object.")

    try:
        shift = shift.to(u.Hz)
    except Exception:
        raise ValueError("shift must be a Quantity with units of frequency.")

    if shift.isscalar:
        shift = shift[None]

    if shift.ndim >= z.ndim:
        raise ValueError(
            f"shift has too many dimensions. Expected <= {z.ndim - 1} dimensions, "
            f"got {shift.ndim} dimensions!"
        )

    ix = (slice(None),) * shift.ndim + (None,) * (z.ndim - shift.ndim - 1)
    ft = (shift[ix] * z.dt).to_value(u.one)

    if isinstance(z.data, da.Array):
        n = da.arange(len(z), chunks=(-1,))
    else:
        n = np.arange(len(z))

    ix = tuple(slice(None) if j == 0 else None for j in range(z.ndim))
    ph = np.exp(2j * np.pi * ft * n[ix]).astype(z.dtype)

    x = np.fft.fftshift(pb.fft.fft(z.data * ph, axis=0), axes=(0,))

    it = np.nditer(ft * len(x), flags=["multi_index"])
    for a in it:
        if a < 0:
            a = int(np.floor(a))
            ix = (np.s_[a:],) + it.multi_index
        else:
            a = int(np.ceil(a))
            ix = (np.s_[:a],) + it.multi_index

        x[ix] = 0

    return type(z).like(z, pb.fft.ifft(np.fft.ifftshift(x, axes=(0,)), axis=0))


def fast_len(z, /):
    """Crops signal to an efficient length for FFTs.

    Output signal is cropped to a length of the largest 7-smooth number
    less than or equal to the length of the input signal.

    Parameters
    ----------
    z : Signal
        Input signal.

    Returns
    -------
    Signal
        Cropped signal.
    """
    N = len(z)
    fast_N = pb.utils.prev_fast_len(N)
    return z[:fast_N]
