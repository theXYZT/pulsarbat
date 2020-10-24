"""Tests for `pulsarbat.RadioSignal` and subclasses."""

import pytest
import numpy as np
import dask.array as da
import astropy.units as u
import pulsarbat as pb


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
