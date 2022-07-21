"""Core signal transforms."""

import numpy as np
import astropy.units as u
from astropy.time import Time
import pulsarbat as pb
import functools
import dask.array as da

__all__ = [
    "signal_transform",
    "concatenate",
    "time_shift",
    "fast_len",
]


def signal_transform(func):
    """Wraps an array function and returns a signal transform.

    The function being decorated must accept an array and return an array
    with function signature::

        func(x, *args, /, **kwargs)

    where `x` is the array being transformed. The decorated function will
    accept a Signal object in place of `x` and by default return a Signal object
    of the same type (unless a different `signal_type` is specified). The
    properties of the returned signal can be modified via `signal_kwargs` if needed.

    If the Signal data is contained within a Dask array, `func` will be applied
    to every chunk independently using `dask.array.map_blocks`. Keyword arguments
    specific to `map_blocks` can be passed via `dask_kwargs`.
    """
    @functools.wraps(func)
    def wrapper(x, *args, signal_type=None, signal_kwargs=dict(),
                dask_kwargs=dict(), **kwargs):

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
    concatenated signal will inherit attributes from given `kwargs` and
    then from the first signal in the sequence, `signals[0]`, except for
    `center_freq` and `freq_align` when concatenating along frequency
    (which must be computed accordingly).

    Parameters
    ----------
    signals : sequence of `~Signal`-like
        Sequence of signals to concatenate. All signals must be Signal
        objects and have the same type and `sample_rate`. If concatenating
        along frequency, then they must also have the same `chan_bw`.
    axis : int or 'time' or 'freq', optional
        Axis along which to concatenate signals. Default is `0`. `time`
        is an alias for `0` (concatenating along time). `freq` implies
        `axis=1` (concatenating along frequency) and requires that
        signals are instances of RadioSignal.

    Returns
    -------
    out : `~Signal`-like
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


def time_shift(z, /, t):
    """Shift signal by given number of samples or time.

    This function shifts signals in time via FFT by multiplying by
    a phase gradient in frequency domain.

    Parameters
    ----------
    z : `~Signal`
        Input signal.
    t : int, float or `~astropy.units.Quantity`
        Shift amount. If a number (int or float), the signal is shifted
        by that number of samples. An astropy Quantity with units of
        time can also be passed, in which case the signal will be
        shifted by `t * z.sample_rate` samples.

    Returns
    -------
    out : `~Signal`
        Shifted signal.
    """
    if t == 0:
        return z

    if isinstance(t, u.Quantity):
        n = np.float64((t * z.sample_rate).to_value(u.one))
    else:
        n = np.float64(t)

    if isinstance(z.data, da.Array):
        f = da.fft.fftfreq(len(z), 1)
    else:
        f = np.fft.fftfreq(len(z), 1)

    ix = tuple(slice(None) if i == 0 else None for i in range(z.ndim))
    ph = np.exp(-2j * np.pi * n * f).astype(np.complex64)[ix]
    shifted = pb.fft.ifft(pb.fft.fft(z.data, axis=0) * ph, axis=0)

    if np.iscomplexobj(z.data):
        shifted = shifted.astype(z.dtype)
    else:
        shifted = shifted.real.astype(z.dtype)

    x = type(z).like(z, shifted, start_time=z.start_time - n * z.dt)

    if n >= 0:
        i = np.int64(np.ceil(n))
        x = x[i:]
    else:
        i = np.int64(np.floor(n))
        x = x[:i]

    return x


def fast_len(z, /):
    """Crops signal to an efficient length for FFTs.

    Output signal is cropped to a length of the largest 7-smooth number
    less than or equal to the length of the input signal.

    Parameters
    ----------
    z : `~Signal`
        Input signal.

    Returns
    -------
    out : `~Signal`
        Cropped signal.
    """
    N = len(z)
    fast_N = pb.utils.prev_fast_len(N)
    return z[:fast_N]
