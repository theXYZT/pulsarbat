"""Tests for `pulsarbat.utils`"""

import pytest
import bisect
import numpy as np
from pulsarbat.utils import real_to_complex, next_fast_len, prev_fast_len

FAST = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 24, 25, 27, 28, 30, 32,
    35, 36, 40, 42, 45, 48, 49, 50, 54, 56, 60, 63, 64, 70, 72, 75, 80, 81, 84, 90, 96,
    98, 100, 105, 108, 112, 120, 125, 126, 128, 135, 140, 144, 147, 150, 160, 162, 168,
    175, 180, 189, 192, 196, 200, 210, 216, 224, 225, 240, 243, 245, 250, 252, 256, 270,
    280, 288, 294, 300, 315, 320, 324, 336, 343, 350, 360, 375, 378, 384, 392, 400, 405,
    420, 432, 441, 448, 450, 480, 486, 490, 500, 504, 512, 525, 540, 560, 567, 576, 588,
    600, 625, 630, 640, 648, 672, 675, 686, 700, 720, 729, 735, 750, 756, 768, 784, 800,
    810, 840, 864, 875, 882, 896, 900, 945, 960, 972, 980, 1000,
]


class TestRealToComplex:
    def gen_input(self, t, w, p):
        return np.cos(w * t + p)

    def prediction(self, t, w, p):
        return np.exp(1j * ((w - (len(t) / 4)) * t[::2] + p))

    def test_theoretical(self):
        for N in [511, 512]:
            t = np.linspace(0, 2 * np.pi, N, endpoint=False)
            for w in [1, 2, 127, 128, 129, 254, 255]:
                for p in [-np.pi, -np.pi / 2, 0, np.pi / 2]:
                    x = self.gen_input(t, w, p)
                    y = self.prediction(t, w, p)
                    z = real_to_complex(x)
                    assert np.allclose(y, z)

    def test_empty(self):
        for sample_shape in [(), (2,), (4, 4)]:
            x = np.zeros((0,) + sample_shape)
            y = real_to_complex(x, axis=0)
            assert np.array_equal(x, y)

    def test_axis(self):
        N = 128
        t = np.linspace(0, 2 * np.pi, N, endpoint=False)
        ws = [1, 2, 3, 4]

        x = np.stack([self.gen_input(t, w, 0) for w in ws], axis=0)
        y = np.stack([self.prediction(t, w, 0) for w in ws], axis=0)
        z = real_to_complex(x, axis=1)
        assert np.allclose(y, z)

        x = np.stack([self.gen_input(t, w, 0) for w in ws], axis=1)
        y = np.stack([self.prediction(t, w, 0) for w in ws], axis=1)
        z = real_to_complex(x, axis=0)
        assert np.allclose(y, z)

    def test_bad_args(self):
        x = np.ones((128, 4), dtype=complex)
        with pytest.raises(ValueError):
            _ = real_to_complex(x, axis=0)

    def test_dtype(self):
        d = [(np.float32, np.complex64), (np.float64, np.complex128)]
        for in_type, out_type in d:
            x = np.random.default_rng().standard_normal(32)
            y = real_to_complex(x.astype(in_type))
            assert y.dtype == out_type


class TestFastLengths:
    def test_next_fast_len(self):
        assert next_fast_len(0) == 0

        for i in range(1, 1000):
            assert next_fast_len(i) == FAST[bisect.bisect_left(FAST, i)]

    def test_prev_fast_len(self):
        assert prev_fast_len(0) == 0

        for i in range(1, 1000):
            assert prev_fast_len(i) == FAST[bisect.bisect(FAST, i) - 1]
