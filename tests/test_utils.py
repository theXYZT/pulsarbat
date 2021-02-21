"""Tests for `pulsarbat.utils`"""

import pytest
import numpy as np
from pulsarbat.utils import real_to_complex


class TestRealToComplex:
    def gen_input(self, t, w, p):
        return np.cos(w*t + p)

    def prediction(self, t, w, p):
        return np.exp(1j*((w - (len(t)/4))*t[::2] + p))

    def test_theoretical(self):
        for N in [511, 512]:
            t = np.linspace(0, 2*np.pi, N, endpoint=False)
            for w in [1, 2, 127, 128, 129, 254, 255]:
                for p in [-np.pi, -np.pi/2, 0, np.pi/2]:
                    x = self.gen_input(t, w, p)
                    y = self.prediction(t, w, p)
                    z = real_to_complex(x)
                    assert np.allclose(y, z)

    def test_axis(self):
        N = 128
        t = np.linspace(0, 2*np.pi, N, endpoint=False)
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
        x = np.ones((0, 4, 4))
        with pytest.raises(ValueError):
            _ = real_to_complex(x, axis=0)

        x = np.ones((128, 4), dtype=complex)
        with pytest.raises(ValueError):
            _ = real_to_complex(x, axis=0)

    def test_dtype(self):
        d = [(np.float32, np.complex64), (np.float64, np.complex128)]
        for in_type, out_type in d:
            x = np.random.default_rng().standard_normal(32)
            y = real_to_complex(x.astype(in_type))
            assert y.dtype == out_type
