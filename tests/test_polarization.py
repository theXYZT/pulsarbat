"""Tests for polarization features."""

import pytest
import numpy as np
import dask.array as da
import astropy.units as u
import pulsarbat as pb


@pytest.mark.parametrize("use_dask", [True, False])
@pytest.mark.parametrize("pol_type", ["linear", "circular"])
def test_pol_reversibility(use_dask, pol_type):
    shape = (4096, 16, 2)
    kw = {"sample_rate": 1 * u.MHz, "center_freq": 1 * u.GHz}

    p = da if use_dask else np
    sig = p.exp(1j * p.random.uniform(-np.pi, +np.pi, shape))

    z = pb.DualPolarizationSignal(sig, pol_type=pol_type, **kw)

    if pol_type == "linear":
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
def test_stokes(use_dask):
    kw = {"sample_rate": 1 * u.MHz, "center_freq": 1 * u.GHz}

    x = np.array(
        [[[1 + 1j, 2 + 1j]], [[3 + 0j, 0 + 4j]], [[0 + 2j, 3 + 1j]]],
        dtype=np.complex128,
    )

    if use_dask:
        x = da.from_array(x)

    lin_stokes = np.array([[[7, -3, 6, -2]], [[25, -7, 0, 24]], [[14, -6, 4, -12]]])

    cir_stokes = np.array([[[7, 6, -2, -3]], [[25, 0, 24, -7]], [[14, 4, -12, -6]]])

    for pol_type, stokes in zip(["linear", "circular"], [lin_stokes, cir_stokes]):

        z = pb.DualPolarizationSignal(x, pol_type=pol_type, **kw)
        y_lin = z.to_linear().to_stokes()
        y_cir = z.to_circular().to_stokes()

        print(pol_type)
        print(y_lin.data)
        print(y_cir.data)

        for y in [y_lin, y_cir]:
            assert isinstance(y, pb.FullStokesSignal)
            assert isinstance(y.data, type(x))
            assert np.allclose(np.asarray(y), stokes)

            for i, s in enumerate(["I", "Q", "U", "V"]):
                for a in [y[s], getattr(y, "stokes" + s)]:
                    assert isinstance(a, pb.IntensitySignal)
                    assert np.allclose(np.asarray(a), stokes[..., i])


def test_stokes_getitem():
    kw = {"sample_rate": 1 * u.MHz, "center_freq": 1 * u.GHz, "chan_bw": 1 * u.MHz}

    x = np.array([[[7, -3, 6, -2]], [[25, -7, 0, 24]], [[14, -6, 4, -12]]])

    z = pb.FullStokesSignal(x, **kw)

    y = z[:2]
    assert isinstance(y, type(z))
    assert np.allclose(np.array(y), x[:2])

    with pytest.raises(KeyError):
        _ = z["fish"]
