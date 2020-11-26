"""Tests for `pulsarbat.RadioSignal` and subclasses."""

import pytest
import numpy as np
import astropy.units as u
import pulsarbat as pb
from astropy.time import Time


@pytest.mark.parametrize("start_time", [None, Time.now()])
def test_concatenate_time(start_time):
    shape = (4096, 32, 2)
    SR = (100 * u.MHz) / shape[1]
    x = np.random.normal(0, 1, shape) + np.random.normal(0, 1, shape) * 1j
    z = pb.DualPolarizationSignal(x, sample_rate=SR, center_freq=400*u.MHz,
                                  pol_type='linear', start_time=start_time)

    y = pb.concatenate([z], axis=0)
    assert np.allclose(np.array(z), np.array(y))
    for a in ('center_freq', 'chan_bw', 'sample_rate'):
        assert u.isclose(getattr(z, a), getattr(y, a))

    if start_time is None:
        assert y.start_time is None
    else:
        assert abs(y.start_time - z.start_time) < 0.1 * u.ns

    idx = [0, 16, 128, 512, 2000, 3000, 4096]
    s = [z[i:j] for i, j in zip(idx, idx[1:])]
    y = pb.concatenate(s, axis=0)
    assert np.allclose(np.array(z), np.array(y))
    for a in ('center_freq', 'chan_bw', 'sample_rate'):
        assert u.isclose(getattr(z, a), getattr(y, a))

    if start_time is None:
        assert y.start_time is None
    else:
        assert abs(y.start_time - z.start_time) < 0.1 * u.ns


@pytest.mark.parametrize("freq_align", ['bottom', 'center', 'top'])
def test_concatenate_freq(freq_align):
    shape = (16, 2048, 4)
    fcen = 7*u.GHz
    chan_bw = (2 * u.GHz) / shape[1]
    x = np.random.normal(0, 1, shape)
    z = pb.FullStokesSignal(x, sample_rate=1*u.Hz, chan_bw=chan_bw,
                            center_freq=fcen, freq_align=freq_align)

    cen = {'bottom': fcen - chan_bw / 2,
           'center': fcen,
           'top': fcen + chan_bw / 2}

    y = pb.concatenate([z], axis=1)
    assert np.allclose(np.array(z), np.array(y))
    assert y.freq_align == 'center'
    for a in ('chan_bw', 'sample_rate'):
        assert u.isclose(getattr(z, a), getattr(y, a))
    assert u.allclose(z.channel_freqs, y.channel_freqs)
    assert u.isclose(y.center_freq, cen[freq_align])

    idx = [0, 2, 128, 129, 512, 1234, 1999, 2048]
    s = [z[:, i:j] for i, j in zip(idx, idx[1:])]
    y = pb.concatenate(s, axis=1)
    assert y.freq_align == 'center'
    for a in ('chan_bw', 'sample_rate'):
        assert u.isclose(getattr(z, a), getattr(y, a))
    assert u.allclose(z.channel_freqs, y.channel_freqs)
    assert u.isclose(y.center_freq, cen[freq_align])


def test_concatenate_errors():
    x = np.random.normal(0, 1, (32, 32, 4))
    z = pb.FullStokesSignal(x, sample_rate=1*u.Hz, chan_bw=1*u.MHz,
                            center_freq=1*u.GHz, freq_align='center',
                            start_time=Time.now())

    with pytest.raises(TypeError):
        _ = pb.concatenate(["!", z])

    with pytest.raises(TypeError):
        _ = pb.concatenate([z, "!"])

    with pytest.raises(TypeError):
        _ = pb.concatenate([pb.Signal.like(z)], axis=1)

    with pytest.raises(ValueError):
        _ = pb.concatenate([z], axis=2)

    with pytest.raises(ValueError):
        _ = pb.concatenate([z, z], axis=0)

    with pytest.raises(ValueError):
        _ = pb.concatenate([z, z], axis=1)

    with pytest.raises(ValueError):
        y = pb.FullStokesSignal.like(z, sample_rate=2*u.Hz)
        _ = pb.concatenate([z, y], axis=0)

    with pytest.raises(ValueError):
        y = pb.FullStokesSignal.like(z, chan_bw=z.chan_bw/2)
        _ = pb.concatenate([z, y], axis=1)
