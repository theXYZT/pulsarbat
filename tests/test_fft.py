"""Tests for `pulsarbat.utils`"""

import pytest
import types
import numpy as np
import dask.array as da
import scipy.fft
import pulsarbat as pb

FFT_FUNCS = [
    "fft",
    "fft2",
    "fftn",
    "ifft",
    "ifft2",
    "ifftn",
    "hfft",
    "ihfft",
    "rfft",
    "rfft2",
    "rfftn",
    "irfft",
    "irfft2",
    "irfftn",
]


class TestFFT:
    def test_dir_funcs(self):
        actual_dir = dir(pb.fft)
        expected_dir = sorted(FFT_FUNCS)

        for a, b in zip(actual_dir, expected_dir):
            assert a == b
            f = getattr(pb.fft, a)
            assert isinstance(f, types.FunctionType)

        with pytest.raises(AttributeError):
            _ = pb.fft.fish

    @pytest.mark.parametrize(
        "fft_func, N", [("fft", 8), ("ifft", 8), ("irfft", 9), ("hfft", 9)]
    )
    def test_complex_input_fft(self, fft_func, N):
        pb_fft_func = getattr(pb.fft, fft_func)
        sp_fft_func = getattr(scipy.fft, fft_func)

        for d in [np.float32, np.float64]:
            a = np.arange(N, dtype=d) + 1j * np.arange(N, dtype=d)
            x = sp_fft_func(a)

            y = pb_fft_func(a)
            assert type(a) == type(y)
            assert x.dtype == y.dtype
            assert np.allclose(x, y)

            b = da.arange(N, dtype=d) + 1j * da.arange(N, dtype=d)

            z = pb_fft_func(b)
            assert type(b) == type(z)
            assert x.dtype == z.dtype
            assert np.allclose(x, np.array(z))

    @pytest.mark.parametrize("fft_func, N", [("rfft", 16), ("ihfft", 16)])
    def test_real_input_fft(self, fft_func, N):
        pb_fft_func = getattr(pb.fft, fft_func)
        sp_fft_func = getattr(scipy.fft, fft_func)

        for d in [np.float32, np.float64]:
            a = np.arange(N, dtype=d)
            x = sp_fft_func(a)

            y = pb_fft_func(a)
            assert type(a) == type(y)
            assert x.dtype == y.dtype
            assert np.allclose(x, y)

            b = da.arange(N, dtype=d)

            z = pb_fft_func(b)
            assert type(b) == type(z)
            assert x.dtype == z.dtype
            assert np.allclose(x, np.array(z))
