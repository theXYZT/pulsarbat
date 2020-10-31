"""Signal-to-signal transforms."""

import numpy as np
import astropy.units as u

from .core import RadioSignal


__all__ = ['stack', ]


def stack(sigs):
    """Stacks multiple signals in frequency.

    All signals must have the same type and the same attributes except
    for `center_freq` and `bandwidth`, which must be consistent to be
    stackable (no overlapping channels or gaps in frequency). The output
    signal will be contiguous in frequency.

    Parameters
    ----------
    signals : sequence of `~pulsarbat.RadioSignal`
        The signals being stacked.

    Returns
    -------
    out : `~pulsarbat.RadioSignal`
        The stacked signal.
    """
    if not all(isinstance(s, RadioSignal) for s in sigs):
        raise ValueError('Some signals are not pb.RadioSignal')

    _sig_type = type(sigs[0])
    if not all(type(s) is _sig_type for s in sigs):
        raise ValueError('Signals are not of same type!')

    _attrs = set(sigs[0].__dict__) | {'chan_bandwidth'}
    _attrs -= {'_data', '_bandwidth', '_center_freq'}
    for a in _attrs:
        temp = getattr(sigs[0], a)
        if not all(temp == getattr(s, a) for s in sigs):
            raise ValueError(f"Signals don't have the same {a}")

    sigs = sorted(sigs, key=lambda s: s.center_freq)
    channels = np.concatenate([s.channel_centers for s in sigs])
    df = np.diff(channels).to_value(u.Hz)

    if not np.allclose(df, df[0]):
        raise ValueError("Signals don't align nicely in frequency!")

    fmin, fmax = sigs[0].min_freq, sigs[-1].max_freq
    bw, fcen = fmax - fmin, (fmin + fmax) / 2

    x = np.concatenate([s.data for s in sigs], axis=1)
    return _sig_type.like(sigs[0], x, bandwidth=bw, center_freq=fcen)
