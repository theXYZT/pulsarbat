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


@pytest.fixture(params=['linear', 'circular'])
def pol_type(request):
    return request.param


@pytest.fixture
def signal_kwargs():
    return {'sample_rate': 50 * u.MHz,
            'center_freq': 400 * u.MHz,
            'bandwidth': 200 * u.MHz}


def test_invalid_poltype(complex_noise, signal_kwargs):
    with pytest.raises(ValueError):
        _ = pb.DualPolarizationSignal(complex_noise, pol_type='incorrect',
                                      **signal_kwargs)


def test_str_poltype(complex_noise, signal_kwargs, pol_type):
    z = pb.DualPolarizationSignal(complex_noise, pol_type=pol_type,
                                  **signal_kwargs)
    print(z)
    repr(z)


def test_pol_reversibility(complex_noise, signal_kwargs, pol_type):
    z = pb.DualPolarizationSignal(complex_noise, pol_type=pol_type,
                                  **signal_kwargs)

    if pol_type == 'linear':
        x = z.to_linear()
        y = z.to_circular().to_linear()
    else:
        x = z.to_circular()
        y = z.to_linear().to_circular()

    assert z.pol_type == x.pol_type
    assert z.pol_type == y.pol_type

    x, y, z = np.array(x), np.array(y), np.array(z)
    assert np.allclose(z.real, x.real, atol=1E-5)
    assert np.allclose(z.real, y.real, atol=1E-5)
    assert np.allclose(z.imag, x.imag, atol=1E-5)
    assert np.allclose(z.imag, y.imag, atol=1E-5)


def test_to_stokes(complex_noise, signal_kwargs, pol_type):
    z = pb.DualPolarizationSignal(complex_noise, pol_type=pol_type,
                                  **signal_kwargs)
    x = z.to_linear().to_stokes()
    y = z.to_circular().to_stokes()
    x, y = np.array(x), np.array(y)
    assert np.allclose(x.real, y.real, atol=1E-5)
    assert np.allclose(x.imag, y.imag, atol=1E-5)
