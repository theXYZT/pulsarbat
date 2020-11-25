"""Tests for `pulsarbat.RadioSignal` and subclasses."""

import pytest
import numpy as np
import dask.array as da
import astropy.units as u
import pulsarbat as pb

SHAPE = (4096, 8, 2)


@pytest.fixture(params=[True, False])
def use_dask(request):
    return request.param


@pytest.fixture
def complex_noise(use_dask):
    f = da.random.standard_normal if use_dask else np.random.standard_normal
    return (f(SHAPE) + 1j * f(SHAPE)).astype(np.complex128)


@pytest.mark.parametrize("pol_type", ['linear', 'circular'])
def test_pol_reversibility(complex_noise, pol_type):
    z = pb.DualPolarizationSignal(complex_noise, pol_type=pol_type,
                                  sample_rate=1*u.MHz, center_freq=1*u.GHz)

    if pol_type == 'linear':
        x = z.to_linear()
        y = z.to_circular().to_linear()
    else:
        x = z.to_circular()
        y = z.to_linear().to_circular()

    assert z.pol_type == x.pol_type
    assert z.pol_type == y.pol_type

    x, y, z = np.array(x), np.array(y), np.array(z)
    assert np.allclose(z.real, x.real)
    assert np.allclose(z.real, y.real)
    assert np.allclose(z.imag, x.imag)
    assert np.allclose(z.imag, y.imag)


@pytest.mark.parametrize("pol_type", ['linear', 'circular'])
def test_to_stokes(complex_noise, pol_type):
    z = pb.DualPolarizationSignal(complex_noise, pol_type=pol_type,
                                  sample_rate=1*u.MHz, center_freq=1*u.GHz)
    x = z.to_linear().to_stokes()
    y = z.to_circular().to_stokes()
    x, y = np.array(x), np.array(y)
    assert np.allclose(x.real, y.real)
    assert np.allclose(x.imag, y.imag)
