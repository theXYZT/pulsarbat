"""Tests for core signal functions."""

import math
import pytest
import itertools
import numpy as np
import dask.array as da
import astropy.units as u
import pulsarbat as pb
from astropy.time import Time


def assert_equal_signals(x, y):
    assert np.allclose(np.array(x), np.array(y))
    assert Time.isclose(x.start_time, y.start_time)
    assert u.isclose(x.sample_rate, y.sample_rate)


def assert_equal_radiosignals(x, y):
    assert_equal_signals(x, y)
    assert u.allclose(x.channel_freqs, y.channel_freqs)


class TestConcatenate:
    @pytest.mark.parametrize("use_dask", [True, False])
    def test_basic(self, use_dask):
        """Basic concatenation functionality."""
        if use_dask:
            f = da.random.standard_normal
        else:
            f = np.random.standard_normal

        shape = (16, 16)
        z = pb.Signal(f(shape), sample_rate=1*u.Hz, start_time=Time.now())

        x, y = z[:10], z[10:]
        for axis in [0, 'time']:
            z2 = pb.concatenate([x, y], axis=axis)
            assert_equal_signals(z, z2)

            with pytest.raises(ValueError):
                _ = pb.concatenate([y, x], axis=axis)

        y.sample_rate = x.sample_rate * 2
        with pytest.raises(ValueError):
            _ = pb.concatenate([x, y], axis=0)

        x, y = z[:, :10], z[:, 10:]
        z2 = pb.concatenate([x, y], axis=1)
        assert_equal_signals(z, z2)

        with pytest.raises(TypeError):
            _ = pb.concatenate([x, y], axis='freq')

        z = pb.RadioSignal(f(shape), sample_rate=1*u.Hz, start_time=Time.now(),
                           chan_bw=1*u.MHz, center_freq=1*u.GHz)

        x, y = z[:, :10], z[:, 10:]
        for axis in [1, 'freq']:
            z2 = pb.concatenate([x, y], axis=axis)
            assert_equal_radiosignals(z, z2)

            with pytest.raises(ValueError):
                _ = pb.concatenate([y, x], axis=axis)

        y.chan_bw = x.chan_bw * 2
        with pytest.raises(ValueError):
            _ = pb.concatenate([x, y], axis=1)

    def test_valid_signal(self):
        """
             🡒 t
           ┏━━━┳━━━┓
           ┃ A ┃ B ┃
         ⭣ ┣━━━╋━━━┫
         x ┃ C ┃ D ┃
           ┗━━━┻━━━┛
        valid along t = AB, AD, CB, CD
        valid along x = AA, AC, BB, BD, CC, CA, DD, DB
        """
        shape = (16, 16)
        z = pb.Signal(np.random.default_rng().standard_normal(shape),
                      sample_rate=1*u.Hz, start_time=Time.now())

        A, B, C, D = z[:8, :8], z[8:, :8], z[:8, 8:], z[8:, 8:]

        y = pb.concatenate([pb.concatenate([A, B], axis=0),
                            pb.concatenate([C, D], axis=0)], axis=1)
        assert_equal_signals(z, y)

        y = pb.concatenate([pb.concatenate([A, C], axis=1),
                            pb.concatenate([B, D], axis=1)], axis=0)
        assert_equal_signals(z, y)

        for XY in itertools.product([A, B, C, D], repeat=2):
            if XY in [(A, B), (A, D), (C, B), (C, D)]:
                _ = pb.concatenate(XY, axis=0)
            else:
                with pytest.raises(ValueError):
                    _ = pb.concatenate(XY, axis=0)

            if XY in [(A, A), (A, C), (B, B), (B, D),
                      (C, C), (C, A), (D, D), (D, B)]:
                _ = pb.concatenate(XY, axis=1)
            else:
                with pytest.raises(ValueError):
                    _ = pb.concatenate(XY, axis=1)

    def test_valid_radiosignal(self):
        """
             🡒 t
           ┏━━━┳━━━┓
           ┃ A ┃ B ┃
         ⭣ ┣━━━╋━━━┫
         f ┃ C ┃ D ┃
           ┗━━━┻━━━┛
        valid along t = AB, CD
        valid along f = AC, BD
        """
        shape = (16, 16)
        z = pb.RadioSignal(np.random.default_rng().standard_normal(shape),
                           sample_rate=1*u.Hz, start_time=Time.now(),
                           chan_bw=1*u.MHz, center_freq=1*u.GHz)

        A, B, C, D = z[:8, :8], z[8:, :8], z[:8, 8:], z[8:, 8:]

        y = pb.concatenate([pb.concatenate([A, B], axis='time'),
                            pb.concatenate([C, D], axis='time')], axis='freq')
        assert_equal_radiosignals(z, y)

        y = pb.concatenate([pb.concatenate([A, C], axis='freq'),
                            pb.concatenate([B, D], axis='freq')], axis='time')
        assert_equal_radiosignals(z, y)

        for XY in itertools.product([A, B, C, D], repeat=2):
            if XY in [(A, B), (C, D)]:
                _ = pb.concatenate(XY, axis='time')
            else:
                with pytest.raises(ValueError):
                    _ = pb.concatenate(XY, axis='time')

            if XY in [(A, C), (B, D)]:
                _ = pb.concatenate(XY, axis='freq')
            else:
                with pytest.raises(ValueError):
                    _ = pb.concatenate(XY, axis='freq')

    def test_start_time_none(self):
        shape = (32, 32)
        k = (0, 13, 19, 19, 23, 32)

        z = pb.Signal(np.random.default_rng().standard_normal(shape),
                      sample_rate=1*u.Hz, start_time=Time.now())

        for c in range(len(k) - 1):
            zs = [z[i:j] for i, j in zip(k, k[1:])]

            for j, x in enumerate(zs):
                if j != c:
                    x.start_time = None

            y = pb.concatenate(zs, axis='time')
            assert_equal_signals(z, y)

        zs = [z[i:j] for i, j in zip(k, k[1:])]
        for x in zs:
            x.start_time = None
        y = pb.concatenate(zs, axis='time')
        assert y.start_time is None

    def test_edge_cases(self):
        shape = (16, 16)

        z = pb.Signal(np.random.default_rng().standard_normal(shape),
                      sample_rate=1*u.Hz, start_time=Time.now())

        with pytest.raises(ValueError):
            _ = pb.concatenate([])

        with pytest.raises(TypeError):
            _ = pb.concatenate([z.data])

        with pytest.raises(TypeError):
            _ = pb.concatenate([z[:8], z[8:].data])


class TestTimeShift:
    @pytest.mark.parametrize("use_dask", [True, False])
    @pytest.mark.parametrize("use_complex", [True, False])
    def test_int_roll(self, use_dask, use_complex):
        if use_dask:
            f = da.random.standard_normal
        else:
            f = np.random.standard_normal

        shape = (4096, 4, 2)

        if use_complex:
            x = (f(shape) + 1j*f(shape)).astype(np.complex128)
        else:
            x = f(shape).astype(np.float64)

        z = pb.Signal(x, sample_rate=1*u.kHz, start_time=Time.now())

        for n in [1, 10, 55, 211]:
            assert_equal_signals(z[:-n], pb.time_shift(z, n))
            assert_equal_signals(z[:-n], pb.time_shift(z, n * u.ms))

            assert_equal_signals(z[n:], pb.time_shift(z, -n))
            assert_equal_signals(z[n:], pb.time_shift(z, -n * u.ms))

    def test_subsample_roll(self):
        def impulse(N, t0):
            """Generate noisy impulse at t0, with given S/N."""
            n = (np.arange(N) - N // 2) / N
            x = np.exp(-2j * np.pi * t0 * n)
            return np.fft.ifft(np.fft.ifftshift(x)).astype(np.complex128)

        N = 1024
        for shift in [16.5, 32.25, 50.1, 60.9, 466.666]:
            imp1 = impulse(N, shift)
            imp2 = impulse(N - math.ceil(shift), 0)
            x = pb.Signal(imp1, sample_rate=1*u.kHz, start_time=Time.now())
            y = pb.time_shift(x, -shift)
            assert np.allclose(np.array(y), imp2)
            z = pb.time_shift(x, -shift * u.ms)
            assert np.allclose(np.array(z), imp2)
