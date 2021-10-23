"""Tests for readers in `pulsarbat.reader`."""

import pytest
from pathlib import Path
import numpy as np
import astropy.units as u
from astropy.time import Time
import pulsarbat as pb
import pulsarbat.readers as pbr

DATA_DIR = Path(__file__).parent.absolute() / "data"


class TestBasebandReader:
    def test_vdif(self):
        r = pbr.BasebandReader(DATA_DIR / "sample.vdif")
        assert r.shape == (20000, 8)
        t0 = Time("2014-06-16T05:56:07.000", format="isot", precision=9)
        assert Time.isclose(r.start_time, t0)

        z = r.read(0, 16)
        assert u.isclose(z.sample_rate, 16 * u.MHz)

        r = pbr.BasebandReader(DATA_DIR / "sample.vdif", squeeze=False)
        assert r.shape == (20000, 8, 1)

    def test_dada(self):
        r = pbr.BasebandReader(DATA_DIR / "sample.dada")
        assert r.shape == (16000, 2)
        t0 = Time("2013-07-02T01:39:20.000", format="isot", precision=9)
        assert Time.isclose(r.start_time, t0)

        z = r.read(0, 16)
        assert u.isclose(z.sample_rate, 16 * u.MHz)

        r = pbr.BasebandReader(DATA_DIR / "sample.dada", squeeze=False)
        assert r.shape == (16000, 2, 1)

    def test_sideband(self):
        f = DATA_DIR / "sample.vdif"
        r1 = pbr.BasebandReader(f, lower_sideband=False)
        r2 = pbr.BasebandReader(f, lower_sideband=True)

        z1, z2 = r1.read(0, 16), r2.read(0, 16)
        assert np.allclose(np.array(z1), np.array(z2).conj())

        lsb = (np.arange(8) % 3).astype(bool)
        r3 = pbr.BasebandReader(f, lower_sideband=lsb)
        z3 = r3.read(0, 16)
        assert np.allclose(np.array(z1)[:, ~lsb], np.array(z3)[:, ~lsb])
        assert np.allclose(np.array(z1)[:, lsb], np.array(z3)[:, lsb].conj())

        with pytest.raises(ValueError):
            _ = pbr.BasebandReader(f, lower_sideband=[True, False])

    def test_intensity_contraints(self):
        with pytest.raises(ValueError):
            _ = pbr.BasebandReader(DATA_DIR / "sample.dada", intensity=True)

        sig_kw = dict(center_freq=1.4*u.GHz, chan_bw=16*u.MHz)

        r = pbr.BasebandReader(DATA_DIR / "sample.vdif", signal_kwargs=sig_kw,
                               signal_type=pb.IntensitySignal)
        assert r.intensity

        with pytest.raises(ValueError):
            _ = pbr.BasebandReader(DATA_DIR / "sample.vdif", signal_kwargs=sig_kw,
                                   signal_type=pb.IntensitySignal, intensity=False)

        sig_kw = dict(center_freq=1.4*u.GHz)

        r = pbr.BasebandReader(DATA_DIR / "sample.vdif", signal_kwargs=sig_kw,
                               signal_type=pb.BasebandSignal)
        assert not r.intensity

        with pytest.raises(ValueError):
            _ = r = pbr.BasebandReader(DATA_DIR / "sample.vdif", signal_kwargs=sig_kw,
                                       signal_type=pb.BasebandSignal, intensity=True)


class TestGUPPIRawReader:
    @pytest.mark.parametrize("use_dask", [True, False])
    def test_basic(self, use_dask):
        fs = sorted(DATA_DIR.glob("fake.*.raw"))
        r = pbr.GUPPIRawReader(fs)
        assert len(r) == 8192 * len(fs)

        assert u.isclose(r.center_freq, 344.1875 * u.MHz)
        assert r.pol_type == "linear"

        z = r.read(0, 8, use_dask=use_dask)
        assert len(z) == 8
        assert u.isclose(z.sample_rate, 3.125 * u.MHz)
        assert u.isclose(z.center_freq, 344.1875 * u.MHz)
        assert u.isclose(z.bandwidth, 12.5 * u.MHz)

        t0 = Time("1997-07-11T12:34:56.000", format="isot", precision=9)
        assert Time.isclose(z.start_time, t0)
        assert z.pol_type == "linear"

    @pytest.mark.parametrize("use_dask", [True, False])
    def test_sequential(self, use_dask):
        fs = sorted(DATA_DIR.glob("fake.*.raw"))

        r1 = pbr.GUPPIRawReader(fs)
        z1 = r1.read(16384 + 16, 32, use_dask=use_dask)

        r2 = pbr.GUPPIRawReader(fs[2])
        z2 = r2.read(16, 32, use_dask=use_dask)

        assert np.allclose(np.array(z1), np.array(z2))
        assert Time.isclose(z1.start_time, z2.start_time)


class TestDADAStokesReader:
    @pytest.mark.parametrize("use_dask", [True, False])
    def test_basic(self, use_dask):
        r = pbr.DADAStokesReader(DATA_DIR / "stokes_ef.dada")
        assert r.shape == (16, 2048, 4)

        z = r.read(0, 4, use_dask=use_dask)
        assert type(z) == pb.FullStokesSignal
        assert u.isclose(z.center_freq, 7 * u.GHz)
        assert u.isclose(z.bandwidth, 2 * u.GHz)
        assert u.isclose(z.dt, 131072 * u.ns)
        assert z.freq_align == "top"
        assert z.nchan == 2048

    def test_not_stokes(self):
        with pytest.raises(ValueError):
            _ = pbr.DADAStokesReader(DATA_DIR / "sample.dada")
