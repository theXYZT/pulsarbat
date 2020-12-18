"""Core signal functions."""

import operator
import numpy as np
import astropy.units as u
from astropy.time import Time
from ..core import Signal, RadioSignal

__all__ = ['concatenate', ]


def concatenate(signals, /, axis=0, **kwargs):
    """Concatenates multiple signals in time or frequency.

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
    **kwargs
        Keyword arguments to pass to the output signal.

    Returns
    -------
    out : `~Signal`-like
        Concatenated signal of same type as input signals.
    """
    try:
        axis = operator.index(axis)
    except Exception:
        if axis not in ('time', 'freq'):
            raise ValueError("axis must be an integer, 'time', or 'freq'.")

    _type = type(signals[0])
    if not all(type(s) is _type for s in signals):
        raise TypeError("All signals must have same type!")

    if not isinstance(signals[0], Signal):
        raise TypeError("Signals must be pulsarbat.Signal objects.")

    ref_sr = signals[0].sample_rate
    if not all(u.isclose(ref_sr, s.sample_rate) for s in signals):
        raise ValueError("Signals don't have the same sample_rate!")

    if axis in {0, 'time'}:
        ref_st, n = None, 0
        for s in signals:
            if s.start_time is not None:
                if ref_st is None:
                    ref_st = s.start_time - (n / ref_sr)
                else:
                    if not Time.isclose(ref_st + (n / ref_sr), s.start_time):
                        raise ValueError("Signals not contiguous in time.")
            n += len(s)

        axis = 0

    elif axis in {1, 'freq'} and isinstance(signals[0], RadioSignal):
        chan_bw = signals[0].chan_bw
        if not all(u.isclose(chan_bw, s.chan_bw) for s in signals):
            raise ValueError("Signals don't have the same chan_bw!")

        if not all(u.isclose(j.channel_freqs[0] - i.channel_freqs[-1], chan_bw)
                   for i, j in zip(signals, signals[1:])):
            raise ValueError("Signals not contiguous in frequency.")

        f0, f1 = signals[0].channel_freqs[0], signals[-1].channel_freqs[-1]
        kwargs['center_freq'] = (f0 + f1) / 2
        kwargs['freq_align'] = 'center'

        axis = 1

    elif axis == 'freq':
        raise TypeError("Signals must be pb.RadioSignal objects when "
                        "axis is 'freq'.")

    z = np.concatenate([s.data for s in signals], axis=axis)
    return _type.like(signals[0], z, **kwargs)
