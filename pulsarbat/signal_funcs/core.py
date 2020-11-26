"""Core signal functions."""

import numpy as np
import astropy.units as u
from ..core import Signal, RadioSignal

__all__ = ['concatenate', ]


def concatenate(signals, axis=0):
    """Concatenates multiple signals in time or frequency.

    All signals must have the same type. Signals must also have the same
    `sample_rate` when concatenating along time or the same `chan_bw`
    when concatenating along frequency. The concatenated signal will
    inherit all other attributes from the first signal provided
    (`signals[0]`) except for `center_freq` when concatenating along
    frequency, of course (there is no problem when concatenating along
    time as the `start_time` is used as a reference which is naturally
    extracted from `signals[0]`).

    Parameters
    ----------
    signals : sequence of `~Signal`-like
        Ordered sequence of signals to concatenate.
    axis : 0 or 1, optional
        Axis along which to concatenate signals. `0` (default)
        indicates concatenation along time, `1` indicates concatenation
        along frequency.

    Returns
    -------
    out : `~Signal`-like
        Concatenated signal of same type as input signals.
    """
    def times_are_equal(t0, t1):
        if t0 is None:
            return t1 is None
        else:
            return abs(t1 - t0) < 0.1 * u.ns

    if axis not in (0, 1):
        raise ValueError("Invalid axis. Must be either 0 or 1.")

    if not isinstance(signals[0], {0: Signal, 1: RadioSignal}[axis]):
        err = {0: "Must be Signal object when axis=0.",
               1: "Must be RadioSignal object when axis=1."}
        raise TypeError(err[axis])

    _type = type(signals[0])
    if not all(type(s) is _type for s in signals):
        raise TypeError("All signals not of same type!")

    _kwargs = dict()
    if axis == 0:
        ref = signals[0].sample_rate
        if not all(u.isclose(ref, s.sample_rate) for s in signals):
            raise ValueError("Signals don't have the same sample_rate!")

        if not all(times_are_equal(i.stop_time, j.start_time)
                   for i, j in zip(signals, signals[1:])):
            raise ValueError("Signals not contiguous in time.")
    else:
        ref = signals[0].chan_bw
        if not all(u.isclose(ref, s.chan_bw) for s in signals):
            raise ValueError("Signals don't have the same chan_bw!")

        if not all(u.isclose(j.channel_freqs[0] - i.channel_freqs[-1], ref)
                   for i, j in zip(signals, signals[1:])):
            raise ValueError("Signals not contiguous in frequency.")

        f0, f1 = signals[0].channel_freqs[0], signals[-1].channel_freqs[-1]
        _kwargs['center_freq'] = (f0 + f1) / 2
        _kwargs['freq_align'] = 'center'

    z = np.concatenate([s.data for s in signals], axis=axis)
    return _type.like(signals[0], z, **_kwargs)
