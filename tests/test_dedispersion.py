"""Tests for dedispersion."""

import pytest
import numpy as np
import astropy.units as u
import pulsarbat as pb
from astropy.time import Time
import scipy.signal
import dask.array as da


class TestDispersionMeasure:
    def test_basic(self):
        DM = pb.DispersionMeasure(2.41e-4)

        for f in [0.1, 1.0, 10.0]:
            dt = DM.time_delay(f * u.MHz, np.inf)
            assert u.isclose(dt, (1 / f / f) * u.s)

            dt = DM.time_delay(np.inf, f * u.MHz)
            assert u.isclose(dt, -(1 / f / f) * u.s)

        dt = DM.time_delay(2 * u.MHz, 1 * u.MHz)
        assert u.isclose(dt, -0.75 * u.s)

        for SR in [1 * u.MHz, 10 * u.MHz, 1 * u.kHz]:
            dn = DM.sample_delay(1 * u.MHz, np.inf, SR)
            assert np.isclose(dn, (SR * u.s).to_value(u.one))

        for a in [10, 20, 100]:
            DM = pb.DispersionMeasure(2.41e-4 * a)
            assert u.isclose(DM.time_delay(1 * u.MHz, np.inf), a * u.s)


class TestCoherentDedispersion:
    @pytest.mark.parametrize("dm", [10.0, 50.0, 100.0])
    @pytest.mark.parametrize("use_dask", [True, False])
    def test_basic(self, dm, use_dask):
        shape = (8192, 4)
        fcen = 1 * u.GHz
        SR = 1 * u.MHz
        DM = pb.DispersionMeasure(dm)

        if use_dask:
            f = da.random.standard_normal
        else:
            f = np.random.standard_normal

        x = f(shape) + 1j * f(shape)
        with pytest.raises(TypeError):
            z = pb.Signal(x, sample_rate=SR)
            _ = pb.coherent_dedispersion(z, DM)

        with pytest.raises(TypeError):
            z = pb.Signal(x, sample_rate=SR)
            _ = DM.chirp_from_signal(z)

        z = pb.BasebandSignal(
            x, sample_rate=SR, center_freq=fcen, start_time=Time.now()
        )

        for ref_freq in [z.min_freq, z.center_freq, z.max_freq]:
            y = pb.coherent_dedispersion(z, DM, ref_freq=ref_freq)

            assert len(z) - len(y) >= DM.sample_delay(
                z.min_freq, z.max_freq, z.sample_rate
            )

            d_st = y.start_time - z.start_time
            toffset = int(np.rint((d_st * z.sample_rate).to_value(u.one)))
            assert toffset >= DM.sample_delay(ref_freq, z.max_freq, z.sample_rate)

    @pytest.mark.parametrize("seed", [4, 8, 15, 16, 23, 42])
    def test_reversibility(self, seed):
        ref_freq = 600 * u.MHz
        SR = 400 * u.MHz
        t0 = Time(56000.0, format="mjd")
        DM = pb.DispersionMeasure(0.01)

        N, M = 2 ** 18, 2 ** 12
        R = np.random.default_rng(seed=seed)
        x = R.standard_normal(N) + 1j * R.standard_normal(N)
        x *= np.exp(-(((np.arange(N) - N // 2) / M) ** 2))

        sos = scipy.signal.butter(10, 0.45, "lowpass", fs=1.0, output="sos")
        x = scipy.signal.sosfilt(sos, x)

        sig = pb.BasebandSignal(
            x.reshape(-1, 1), sample_rate=SR, center_freq=ref_freq, start_time=t0
        )
        temp = pb.coherent_dedispersion(sig, DM)
        sig2 = pb.coherent_dedispersion(temp, -DM)

        toffset = sig2.start_time - sig.start_time
        noffset = int(np.rint((toffset * sig.sample_rate).to(u.one)))
        sig1 = sig[noffset : noffset + len(sig2)]
        res = np.array(sig1) - np.array(sig2)
        assert np.allclose(res, 0, atol=3e-8)

    @pytest.mark.parametrize("DM", [pb.DM(0.01), pb.DM(0.02)])
    def test_correctness(self, DM):
        def gabor_wavelet(t, t0, f0, ts):
            a = 2j * np.pi * (t - t0) * f0 - ((t - t0) / ts) ** 2
            return np.exp(a.to_value(u.one))

        ref_freq = 600 * u.MHz
        SR = 400 * u.MHz

        index = 100000
        N = 2 ** 18
        t = np.arange(N) / SR
        t0 = t[index]

        width = 256
        x = np.zeros(N, dtype=np.complex128)
        for df in np.linspace(-3 * SR / 8, 3 * SR / 8, 13):
            dt = DM.time_delay(ref_freq + df, ref_freq)
            x += gabor_wavelet(t, t0 + dt, df, width / SR)

        sig1 = pb.BasebandSignal(
            x.reshape(-1, 1),
            sample_rate=SR,
            center_freq=ref_freq,
            start_time=Time.now(),
        )
        sig2 = pb.coherent_dedispersion(sig1, DM)

        toffset = sig2.start_time - sig1.start_time
        noffset = int(np.rint((toffset * sig2.sample_rate).to(u.one)))

        sig1, sig2 = np.array(sig1), np.array(sig2)
        id1, id2 = index - noffset - 8 * width, index - noffset + 8 * width

        sig1_power = (np.abs(sig1) ** 2).sum()
        sig2_power = (np.abs(sig2[id1:id2]) ** 2).sum()

        assert np.allclose(sig1_power, sig2_power)
        assert np.allclose(sig2[id2:], 0)
        assert np.allclose(sig2[:id1], 0)

    @pytest.mark.parametrize("use_dask", [True, False])
    @pytest.mark.parametrize("DM", [pb.DM(10), pb.DM(20), pb.DM(50)])
    def test_chirp(self, use_dask, DM):
        shape = (8192, 4)

        if use_dask:
            f = da.random.standard_normal
        else:
            f = np.random.standard_normal

        x = f(shape) + 1j * f(shape)
        z = pb.BasebandSignal(x, sample_rate=1 * u.MHz, center_freq=1 * u.GHz)

        chirp = DM.chirp_from_signal(z)

        y1 = pb.coherent_dedispersion(z, DM)
        y2 = pb.coherent_dedispersion(z, DM, chirp=chirp)

        for rf in [z.min_freq, z.max_freq]:
            chirp = DM.chirp_from_signal(z, ref_freq=rf)
            y1 = pb.coherent_dedispersion(z, DM, ref_freq=rf)
            y2 = pb.coherent_dedispersion(z, DM, ref_freq=rf, chirp=chirp)
            assert np.allclose(y1, y2)


class TestIncoherentDedispersion:
    @pytest.mark.parametrize("use_dask", [True, False])
    @pytest.mark.parametrize("dm", [50.0, 100.0, 200.0])
    def test_basic(self, dm, use_dask):
        SR = 1 * u.kHz
        ref_freq = 1 * u.GHz
        shape = (8192, 32, 4)
        t0 = Time.now()

        DM = pb.DispersionMeasure(dm)
        if use_dask:
            x = da.random.standard_normal(shape)
        else:
            x = np.random.standard_normal(shape)

        z1 = pb.FullStokesSignal(
            x, sample_rate=SR, start_time=t0, center_freq=ref_freq, chan_bw=8 * u.MHz
        )
        z2 = pb.incoherent_dedispersion(z1, DM)

        delay_top = DM.sample_delay(ref_freq, z1.channel_freqs[-1], SR).round()
        delay_bot = DM.sample_delay(z1.channel_freqs[0], ref_freq, SR).round()
        assert len(z1) - len(z2) == int(delay_top + delay_bot)
