"""Tests for polarization features."""

import pytest
import numpy as np
import dask.array as da
import astropy.units as u
import pulsarbat as pb


@pytest.mark.parametrize("use_dask", [True, False])
@pytest.mark.parametrize("pol_type", ['linear', 'circular'])
def test_pol_reversibility(use_dask, pol_type):
    shape = (4096, 16, 2)
    kw = {'sample_rate': 1*u.MHz, 'center_freq': 1*u.GHz}

    p = da if use_dask else np
    sig = p.exp(1j * p.random.uniform(-np.pi, +np.pi, shape))

    z = pb.DualPolarizationSignal(sig, pol_type=pol_type, **kw)

    if pol_type == 'linear':
        x = z.to_linear()
        y = z.to_circular().to_linear()
    else:
        x = z.to_circular()
        y = z.to_linear().to_circular()

    for a in [x, y]:
        assert isinstance(a.data, type(sig))
        assert a.pol_type == pol_type
        assert np.allclose(np.array(z), np.array(a))


@pytest.mark.parametrize("use_dask", [True, False])
def test_to_stokes(use_dask):
    kw = {'sample_rate': 1*u.MHz, 'center_freq': 1*u.GHz}

    pol_comp = [np.complex128([[[1 + 1j, 1 + 0j]]]),
                np.complex128([[[1 + 2j, 2 + 1j]]]),
                np.complex128([[[0 + 1j, 3 + 1j]]])]

    lin_stokes = [np.float64([[[3, 1, 2, -2]]]),
                  np.float64([[[10, 0, 8, -6]]]),
                  np.float64([[[11, -9, 2, -6]]])]

    cir_stokes = [np.float64([[[3, 2, -2, 1]]]),
                  np.float64([[[10, 8, -6, 0]]]),
                  np.float64([[[11, 2, -6, -9]]])]

    for x, lin, cir in zip(pol_comp, lin_stokes, cir_stokes):
        if use_dask:
            x = da.from_array(x)

        z = pb.DualPolarizationSignal(x, pol_type='linear', **kw)
        y1 = z.to_linear().to_stokes().data
        y2 = z.to_circular().to_stokes().data
        assert isinstance(y1, type(x))
        assert isinstance(y2, type(x))
        assert np.allclose(np.array(y1), lin)
        assert np.allclose(np.array(y2), lin)

        z = pb.DualPolarizationSignal(x, pol_type='circular', **kw)
        y1 = z.to_linear().to_stokes().data
        y2 = z.to_circular().to_stokes().data
        assert isinstance(y1, type(x))
        assert isinstance(y2, type(x))
        assert np.allclose(np.array(y1), cir)
        assert np.allclose(np.array(y2), cir)
