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
    assert np.allclose(np.array(x), np.array(y), atol=1e-6)
    assert Time.isclose(x.start_time, y.start_time)
    assert u.isclose(x.sample_rate, y.sample_rate)


def assert_equal_radiosignals(x, y):
    assert_equal_signals(x, y)
    assert u.allclose(x.channel_freqs, y.channel_freqs)


def impulse(N, t0):
    """Generate noisy impulse at t0, with given S/N."""
    n = (np.arange(N) - N // 2) / N
    x = np.exp(-2j * np.pi * t0 * n)
    return np.fft.ifft(np.fft.ifftshift(x)).astype(np.complex64)


def sinusoid(N, f0):
    """Generate a complex sinusoid at frequency f0."""
    n = np.arange(N) / N
    return np.exp(2j * np.pi * f0 * n).astype(np.complex128)


class TestConcatenate:
    @pytest.mark.parametrize("use_dask", [True, False])
    def test_basic(self, use_dask):
        """Basic concatenation functionality."""
        if use_dask:
            f = da.random.standard_normal
        else:
            f = np.random.standard_normal

        shape = (16, 16)
        z = pb.Signal(f(shape), sample_rate=1 * u.Hz, start_time=Time.now())

        x, y = z[:10], z[10:]
        for axis in [0, "time"]:
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
            _ = pb.concatenate([x, y], axis="freq")

        z = pb.RadioSignal(
            f(shape),
            sample_rate=1 * u.Hz,
            start_time=Time.now(),
            chan_bw=1 * u.MHz,
            center_freq=1 * u.GHz,
        )

        x, y = z[:, :10], z[:, 10:]
        for axis in [1, "freq"]:
            z2 = pb.concatenate([x, y], axis=axis)
            assert_equal_radiosignals(z, z2)

            with pytest.raises(ValueError):
                _ = pb.concatenate([y, x], axis=axis)

        y.chan_bw = x.chan_bw * 2
        with pytest.raises(ValueError):
            _ = pb.concatenate([x, y], axis=1)

    def test_valid_signal(self):
        """
              → t
           ┏━━━┳━━━┓
           ┃ A ┃ B ┃
         ↓ ┣━━━╋━━━┫
         x ┃ C ┃ D ┃
           ┗━━━┻━━━┛
        valid along t = AB, AD, CB, CD
        valid along x = AA, AC, BB, BD, CC, CA, DD, DB
        """
        shape = (16, 16)
        z = pb.Signal(
            np.random.default_rng().standard_normal(shape),
            sample_rate=1 * u.Hz,
            start_time=Time.now(),
        )

        A, B, C, D = z[:8, :8], z[8:, :8], z[:8, 8:], z[8:, 8:]

        y = pb.concatenate(
            [pb.concatenate([A, B], axis=0), pb.concatenate([C, D], axis=0)], axis=1
        )
        assert_equal_signals(z, y)

        y = pb.concatenate(
            [pb.concatenate([A, C], axis=1), pb.concatenate([B, D], axis=1)], axis=0
        )
        assert_equal_signals(z, y)

        for X, Y in itertools.product([A, B, C, D], repeat=2):
            time_pairs = [(A, B), (A, D), (C, B), (C, D)]

            if any(X is M and Y is N for M, N in time_pairs):
                _ = pb.concatenate([X, Y], axis=0)
            else:
                with pytest.raises(ValueError):
                    _ = pb.concatenate([X, Y], axis=0)

            freq_pairs = [
                (A, A),
                (A, C),
                (B, B),
                (B, D),
                (C, C),
                (C, A),
                (D, D),
                (D, B),
            ]

            if any(X is M and Y is N for M, N in freq_pairs):
                _ = pb.concatenate([X, Y], axis=1)
            else:
                with pytest.raises(ValueError):
                    _ = pb.concatenate([X, Y], axis=1)

    def test_valid_radiosignal(self):
        """
              → t
           ┏━━━┳━━━┓
           ┃ A ┃ B ┃
         ↓ ┣━━━╋━━━┫
         f ┃ C ┃ D ┃
           ┗━━━┻━━━┛

        valid along t = AB, CD
        valid along f = AC, BD
        """
        shape = (16, 16)
        z = pb.RadioSignal(
            np.random.default_rng().standard_normal(shape),
            sample_rate=1 * u.Hz,
            start_time=Time.now(),
            chan_bw=1 * u.MHz,
            center_freq=1 * u.GHz,
        )

        A, B, C, D = z[:8, :8], z[8:, :8], z[:8, 8:], z[8:, 8:]

        y = pb.concatenate(
            [pb.concatenate([A, B], axis="time"), pb.concatenate([C, D], axis="time")],
            axis="freq",
        )
        assert_equal_radiosignals(z, y)

        y = pb.concatenate(
            [pb.concatenate([A, C], axis="freq"), pb.concatenate([B, D], axis="freq")],
            axis="time",
        )
        assert_equal_radiosignals(z, y)

        for X, Y in itertools.product([A, B, C, D], repeat=2):
            time_pairs = [(A, B), (C, D)]

            if any(X is M and Y is N for M, N in time_pairs):
                _ = pb.concatenate([X, Y], axis=0)
            else:
                with pytest.raises(ValueError):
                    _ = pb.concatenate([X, Y], axis=0)

            freq_pairs = [(A, C), (B, D)]

            if any(X is M and Y is N for M, N in freq_pairs):
                _ = pb.concatenate([X, Y], axis=1)
            else:
                with pytest.raises(ValueError):
                    _ = pb.concatenate([X, Y], axis=1)

    def test_start_time_none(self):
        shape = (32, 32)
        k = (0, 13, 19, 19, 23, 32)

        z = pb.Signal(
            np.random.default_rng().standard_normal(shape),
            sample_rate=1 * u.Hz,
            start_time=Time.now(),
        )

        for c in range(len(k) - 1):
            zs = [z[i:j] for i, j in zip(k, k[1:])]

            for j, x in enumerate(zs):
                if j != c:
                    x.start_time = None

            y = pb.concatenate(zs, axis="time")
            assert_equal_signals(z, y)

        zs = [z[i:j] for i, j in zip(k, k[1:])]
        for x in zs:
            x.start_time = None
        y = pb.concatenate(zs, axis="time")
        assert y.start_time is None

    def test_edge_cases(self):
        shape = (16, 16)

        z = pb.Signal(
            np.random.default_rng().standard_normal(shape),
            sample_rate=1 * u.Hz,
            start_time=Time.now(),
        )

        with pytest.raises(ValueError):
            _ = pb.concatenate([])

        with pytest.raises(TypeError):
            _ = pb.concatenate([z.data])

        with pytest.raises(TypeError):
            _ = pb.concatenate([z[:8], z[8:].data])


class TestSnippet:
    @pytest.mark.parametrize("use_dask", [True, False])
    def test_basic(self, use_dask):
        for t0 in [1.2, 10.5, 15.9]:
            z = pb.Signal(impulse(1024, t0), sample_rate=1 * u.Hz)
            if use_dask:
                z = z.to_dask_array()

            for n in [8, 16, 100]:
                x = pb.snippet(z, t0, 8)
                assert isinstance(x.data, da.Array if use_dask else np.ndarray)
                y = np.zeros_like(x.data)
                y[0] = 1

                assert np.allclose(x.data, y, atol=1e-6)
                assert x.sample_rate == z.sample_rate

    @pytest.mark.parametrize("t0", [Time.now(), None])
    def test_time(self, t0):
        for t in [15.2, 25.6, 11.1]:
            z = pb.Signal(impulse(1024, 512), sample_rate=1 * u.Hz, start_time=t0)

            y = pb.snippet(z, t, 8)

            if t0 is None:
                assert y.start_time is None
            else:
                assert Time.isclose(y.start_time, z.start_time + t * z.dt)

    def test_errors(self):
        z = pb.Signal(impulse(1024, 512), sample_rate=1 * u.Hz)

        oob_tn = [
            (-100, 100),
            (-100, -100),
            (1000, 50),
            (1000, -50),
            (2000, 20),
            (2000, -20),
        ]

        # Out of bounds
        for t, n in oob_tn:
            with pytest.raises(ValueError):
                _ = pb.snippet(z, t, n)

        # Bad arguments
        with pytest.raises(TypeError):
            _ = pb.snippet(z, t, np.arange(4))

        with pytest.raises(ValueError):
            _ = pb.snippet(z, np.arange(10), 10)


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
            x = (f(shape) + 1j * f(shape)).astype(np.complex128)
        else:
            x = f(shape).astype(np.float64)

        z = pb.Signal(x, sample_rate=1 * u.kHz, start_time=Time.now())

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
            x = pb.Signal(imp1, sample_rate=1 * u.kHz, start_time=Time.now())
            y = pb.time_shift(x, -shift)
            assert np.allclose(np.array(y), imp2)
            z = pb.time_shift(x, -shift * u.ms)
            assert np.allclose(np.array(z), imp2)


class TestFreqShift:
    @pytest.mark.parametrize("nband", [1, 4, 8])
    @pytest.mark.parametrize("npol", [1, 2])
    def test_int_roll(self, nband, npol):
        N = 1024
        sig = np.array([[sinusoid(N, 0)] * nband] * npol).T
        x = pb.BasebandSignal(
            sig, sample_rate=1 * u.Hz, center_freq=1 * u.kHz, start_time=Time.now()
        )
        for _ in range(20):
            shifts = np.random.choice(np.arange(-N // 2, N // 2), x.sample_shape)
            y = pb.freq_shift(x, shifts * x.sample_rate / len(x))
            a = abs(np.fft.fft(y, axis=0))
            assert (
                np.nonzero(~np.isclose(a, 0))[0] == sorted(shifts.flatten() % N)
            ).all()

    @pytest.mark.parametrize("nband", [1, 4, 8])
    @pytest.mark.parametrize("npol", [1, 2])
    def test_subsample_roll(self, nband, npol):
        N = 1024
        for _ in range(20):
            shifts = np.random.uniform(-N // 2, N // 2 - 1, (nband, npol))
            sig_1 = np.zeros((N, nband, npol), dtype=complex)
            for band in range(nband):
                for pol in range(npol):
                    sig_1[:, band, pol] = sinusoid(N, shifts[band, pol])
            sig_2 = np.array([[sinusoid(N, 0)] * nband] * npol).T
            x = pb.BasebandSignal(
                sig_1,
                sample_rate=1 * u.Hz,
                center_freq=1 * u.kHz,
                start_time=Time.now(),
            )
            y = pb.freq_shift(x, -shifts * x.sample_rate / len(x))
            print(shifts)
            assert np.allclose(np.array(y), sig_2)

    @pytest.mark.parametrize("nband", [1, 4, 8])
    @pytest.mark.parametrize("npol", [1, 2])
    @pytest.mark.parametrize("use_dask", [True, False])
    @pytest.mark.parametrize("use_complex", [True, False])
    def test_zeroing(self, use_dask, use_complex, nband, npol):
        N = 1024
        if use_dask:
            f = da.random.standard_normal
        else:
            f = np.random.standard_normal

        shape = (N, nband, npol)

        if use_complex:
            x = (f(shape) + 1j * f(shape)).astype(np.complex128)
        else:
            x = f(shape).astype(np.float64)

        z = pb.BasebandSignal(
            x,
            sample_rate=1 * u.kHz,
            center_freq=1 * u.kHz,
            start_time=Time.now(),
        )
        for _ in range(20):
            shifts = np.random.uniform(-N * 2, N * 2, (nband, npol))
            y = pb.freq_shift(z, shifts * z.sample_rate / len(z))
            a = abs(np.fft.fft(y, axis=0))
            shifts[abs(shifts) > N] = N
            assert np.count_nonzero(np.isclose(a, 0)) == np.ceil(abs(shifts)).sum()


class TestFastLen:
    def test_fast(self):
        for N in [4096, 4100, 4111]:
            x = np.arange(N, dtype=np.float64)
            z = pb.Signal(x, sample_rate=1 * u.Hz)
            y = pb.fast_len(z)
            assert len(y) == 4096
            assert np.allclose(y.data, np.arange(4096))


class TestSignalTransform:
    @pytest.mark.parametrize("arange", [np.arange, da.arange])
    def test_median_filter(self, arange):
        from scipy.ndimage import median_filter

        data = arange(9).reshape(-1, 3)
        res = median_filter(arange(9).reshape(-1, 3), size=3, mode="constant")

        sig_med_filt = pb.signal_transform(median_filter)

        t0 = Time.now()
        z = pb.Signal(data, sample_rate=1 * u.Hz, start_time=t0)

        x = type(z).like(z, res)
        y = sig_med_filt(z, size=3, mode="constant")

        assert isinstance(y.data, type(z.data))
        assert_equal_signals(x, y)

        kw = dict(center_freq=1 * u.GHz, chan_bw=10 * u.MHz)
        y = sig_med_filt(
            z, signal_type=pb.IntensitySignal, signal_kwargs=kw, size=3, mode="constant"
        )

        assert isinstance(y, pb.IntensitySignal)
        assert y.center_freq == kw["center_freq"]
        assert y.chan_bw == kw["chan_bw"]

        with pytest.raises(TypeError):
            _ = sig_med_filt(z, signal_type=np.ndarray, size=3, mode="constant")
