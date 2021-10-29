"""Tests for core signal functions."""

import pytest
import numpy as np
import dask.array as da
import astropy.units as u
from astropy.time import Time
import pulsarbat as pb


def assert_equal_signals(x, y):
    assert np.allclose(np.array(x), np.array(y)), f"{x.shape}, {y.shape}"
    assert Time.isclose(x.start_time, y.start_time)
    assert u.isclose(x.sample_rate, y.sample_rate)


def assert_equal_radiosignals(x, y):
    assert_equal_signals(x, y)
    assert u.allclose(x.channel_freqs, y.channel_freqs)


class TestSTFT:
    @pytest.mark.parametrize("shape", [(4224, 4, 2), (4233, 3, 2)])
    @pytest.mark.parametrize("use_dask", [True, False])
    def test_reversibility(self, use_dask, shape):
        p = da if use_dask else np
        kw = {
            "sample_rate": 4 * u.MHz,
            "center_freq": 400 * u.MHz,
            "pol_type": "linear",
            "start_time": Time.now(),
        }

        x = p.exp(1j * p.random.uniform(-np.pi, +np.pi, shape))
        z = pb.DualPolarizationSignal(x, **kw)

        for n in [33, 32, shape[0]]:
            y = pb.contrib.istft(pb.contrib.stft(z, nperseg=n), nperseg=n)
            assert isinstance(y, type(z))
            assert isinstance(y.data, type(z.data))
            assert_equal_radiosignals(z[: len(y)], y)

    def test_single_tone(self):
        x = np.exp(2j * np.pi * np.arange(1024) * 0.25)[:, None]
        z = pb.BasebandSignal(x, sample_rate=1 * u.MHz, center_freq=1 * u.GHz)

        for n in [32, 64, 512, 1024]:
            y = pb.contrib.stft(z, nperseg=n)
            a = np.zeros_like(y)
            a[:, 3 * n // 4] = 1.0
            assert np.allclose(a, y.data)
