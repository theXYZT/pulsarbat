"""Core signal functions."""

import numpy as np
import astropy.units as u
from astropy.time import Time
import pulsarbat as pb

__all__ = [
    'concatenate',
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
