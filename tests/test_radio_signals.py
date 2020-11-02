"""Tests for `pulsarbat.RadioSignal` and subclasses."""

import pytest
import numpy as np
import dask.array as da
import astropy.units as u
from astropy.time import Time
import pulsarbat as pb


def times_are_close(t1, t2):
    return np.all(np.abs(t1 - t2) < 0.1 * u.ns)


@pytest.mark.parametrize("nchan, channel_centers",
                         [(5, [520., 560., 600., 640., 680.] * u.MHz),
                          (4, [525., 575., 625., 675.] * u.MHz),
                          (2, [550., 650.] * u.MHz),
                          (1, [600] * u.MHz)])
def test_radiosignal(nchan, channel_centers):
    center_freq = 600 * u.MHz
    bandwidth = 200 * u.MHz
    chan_bandwidth = bandwidth / nchan

    min_freq = 500 * u.MHz
    max_freq = 700 * u.MHz

    z = pb.RadioSignal(np.random.standard_normal((16, nchan, 2)),
                       sample_rate=1 * u.Hz,
                       center_freq=center_freq,
                       bandwidth=bandwidth)

    assert z.nchan == nchan
    assert z.center_freq == center_freq
    assert z.bandwidth == bandwidth
    assert u.isclose(z.min_freq, min_freq)
    assert u.isclose(z.max_freq, max_freq)
    assert u.isclose(z.chan_bandwidth, chan_bandwidth)

    for a, b in zip(z.channel_centers, channel_centers):
        assert u.isclose(a, b)


def test_basebandsignal():
    shape = (16, 4)

    r = np.random.default_rng(42)
    x = r.normal(0, 1, shape) + 1j * r.normal(0, 1, shape)
    sample_rate = 1 * u.MHz
    center_freq = 400 * u.MHz

    z = pb.BasebandSignal(x, sample_rate=sample_rate, center_freq=center_freq,
                          bandwidth=4 * u.MHz)

    assert u.isclose(z.chan_bandwidth, sample_rate)

    with pytest.raises(ValueError):
        z = pb.BasebandSignal(x, sample_rate=sample_rate,
                              center_freq=center_freq,
                              bandwidth=10 * u.MHz)


@pytest.mark.parametrize("use_dask", [True, False])
@pytest.mark.parametrize("A", [1, 2, 4, 10])
def test_baseband_to_intensity(A, use_dask):
    shape = (1024, 4, 2)

    if use_dask:
        x = da.random.uniform(-np.pi, np.pi, shape)
    else:
        x = np.random.uniform(-np.pi, np.pi, shape)
    z_phasor = np.exp(1j * x).astype(np.complex64)

    z = pb.BasebandSignal(A * z_phasor, sample_rate=1 * u.Hz,
                          center_freq=100 * u.Hz, bandwidth=4 * u.Hz)

    zi = z.to_intensity()
    assert isinstance(zi.data, type(z.data))
    assert np.allclose(A**2, np.array(zi))


def stack_data(nchan, use_dask=False):
    shape = (64, nchan, 2)
    f = da.random.normal if use_dask else np.random.normal
    return f(0, 1, shape) + 1j * f(0, 1, shape)


@pytest.mark.parametrize("use_dask", [True, False])
def test_stack_basic(use_dask):
    sigs = [pb.RadioSignal(stack_data(4, use_dask), sample_rate=1*u.Hz,
                           center_freq=fcen, bandwidth=100*u.MHz)
            for fcen in [250 * u.MHz, 350 * u.MHz, 450 * u.MHz, 550 * u.MHz]]

    fcens = np.concatenate([s.channel_centers for s in sigs])
    z = pb.stack(sigs)
    assert u.allclose(fcens, z.channel_centers)
    assert u.isclose(z.bandwidth, 400 * u.MHz)


def test_stack_comprehensive():
    st = Time.now()

    a1 = pb.DualPolarizationSignal(stack_data(4), sample_rate=10*u.MHz,
                                   center_freq=420*u.MHz, bandwidth=40*u.MHz,
                                   pol_type='linear', start_time=st)

    a2 = pb.DualPolarizationSignal(stack_data(6), sample_rate=10*u.MHz,
                                   center_freq=470*u.MHz, bandwidth=60*u.MHz,
                                   pol_type='linear', start_time=st)

    x = pb.stack([a1, a2])
    assert u.isclose(x.bandwidth, 100 * u.MHz)
    assert x.pol_type == 'linear'
    assert times_are_close(st, x.start_time)
    assert u.isclose(x.center_freq, 450 * u.MHz)

    b = pb.DualPolarizationSignal(stack_data(6), sample_rate=10*u.MHz,
                                  center_freq=470*u.MHz, bandwidth=60*u.MHz,
                                  pol_type='circular', start_time=st)

    with pytest.raises(ValueError):
        _ = pb.stack([a1, b])

    c = pb.Signal(stack_data(4), sample_rate=10*u.MHz)

    with pytest.raises(ValueError):
        _ = pb.stack([b, c])

    d = pb.RadioSignal(stack_data(6), sample_rate=10*u.MHz,
                       center_freq=470*u.MHz, bandwidth=60*u.MHz)

    with pytest.raises(ValueError):
        _ = pb.stack([a1, d])

    with pytest.raises(ValueError):
        _ = pb.stack([a1, a2, a2])
