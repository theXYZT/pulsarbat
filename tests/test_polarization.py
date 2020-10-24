"""Tests for `pulsarbat.RadioSignal` and subclasses."""

import pytest
import numpy as np
import dask.array as da
import astropy.units as u
import pulsarbat as pb

SHAPE = (1024, 4, 2)


@pytest.fixture(params=[True, False])
def use_dask(request):
    return request.param


@pytest.fixture
def complex_noise(use_dask):
    f = da.random.standard_normal if use_dask else np.random.standard_normal
    return f(SHAPE) + 1j * f(SHAPE)


@pytest.fixture
def signal_kwargs():
    return {'sample_rate': 50 * u.MHz,
            'center_freq': 400 * u.MHz,
            'bandwidth': 200 * u.MHz}


def test_invalid_poltype(complex_noise, signal_kwargs):
    with pytest.raises(ValueError):
        _ = pb.DualPolarizationSignal(complex_noise, pol_type='incorrect',
                                      **signal_kwargs)


def test_pol_reversibility(complex_noise, signal_kwargs):
    z = pb.DualPolarizationSignal(complex_noise, pol_type='linear',
                                  **signal_kwargs)
    x = z.to_linear()
    y = z.to_circular().to_linear()
    assert np.allclose(np.array(z), np.array(x))
    assert np.allclose(np.array(z), np.array(y))
    assert z.pol_type == x.pol_type
    assert z.pol_type == y.pol_type

    z = pb.DualPolarizationSignal(complex_noise, pol_type='circular',
                                  **signal_kwargs)
    x = z.to_circular()
    y = z.to_linear().to_circular()
    assert np.allclose(np.array(z), np.array(x))
    assert np.allclose(np.array(z), np.array(y))
    assert z.pol_type == x.pol_type
    assert z.pol_type == y.pol_type


def test_to_stokes(complex_noise, signal_kwargs):
    z = pb.DualPolarizationSignal(complex_noise, pol_type='linear',
                                  **signal_kwargs)
    x = z.to_linear().to_stokes()
    y = z.to_circular().to_stokes()
    assert np.allclose(np.array(x), np.array(y))

    z = pb.DualPolarizationSignal(complex_noise, pol_type='circular',
                                  **signal_kwargs)
    x = z.to_linear().to_stokes()
    y = z.to_circular().to_stokes()
    assert np.allclose(np.array(x), np.array(y))
