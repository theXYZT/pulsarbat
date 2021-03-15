"""Core signal functions."""

import numpy as np
import astropy.units as u
from astropy.time import Time
import pulsarbat as pb

__all__ = [
    'concatenate',
    'time_shift',
]


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
    if axis in {0, 'time'}:
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

    kw = {'start_time': ref_st}

    if isinstance(signals[0], pb.RadioSignal):
        ref_cbw = signals[0].chan_bw
        if not all(u.isclose(ref_cbw, s.chan_bw) for s in signals):
            raise ValueError("RadioSignals must have the same chan_bw!")

        if axis in {1, 'freq'}:
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

        kw['center_freq'] = (f0 + f1) / 2
        kw['freq_align'] = 'center'

    elif axis == 'freq':
        err = "Signals must be pb.RadioSignal objects when axis is 'freq'."
        raise TypeError(err)

    z = np.concatenate([s.data for s in signals], axis=axis)
    return sig_type.like(signals[0], z, **kw)


def time_shift(z, t, /):
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
    def transfer_func(shape, shift):
        ndim = len(shape)
        ix = tuple(slice(None) if i == 0 else None for i in range(ndim))
        ph = np.exp(-2j * np.pi * shift * np.fft.fftfreq(shape[0], 1))[ix]
        return ph.astype(np.complex128)

    if isinstance(t, u.Quantity):
        n = np.float64((t * z.sample_rate).to_value(u.one))
    else:
        n = np.float64(t)

    try:
        import dask
        import dask.array as da
    except ImportError:
        use_dask = False
    else:
        use_dask = isinstance(z.data, da.Array)

    if use_dask:
        delayed_tf = dask.delayed(transfer_func, pure=True)
        ph = da.from_delayed(delayed_tf(z.shape, n), dtype=np.complex64,
                             shape=z.shape[:1] + (1,)*(z.ndim - 1))
    else:
        ph = transfer_func(z.shape, n)

    shifted = np.fft.ifft(np.fft.fft(z.data, axis=0) * ph, axis=0)
    if np.iscomplexobj(z.data):
        shifted = shifted.astype(z.dtype)
    else:
        shifted = shifted.real.astype(z.dtype)

    x = type(z).like(z, shifted, start_time=z.start_time - n*z.dt)

    if n >= 0:
        i = np.int64(np.ceil(n))
        return x[i:]
    else:
        i = np.int64(np.floor(n))
        return x[:i]
