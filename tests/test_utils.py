"""Tests for `pulsarbat.RadioSignal` and subclasses."""

import pytest
import numpy as np
import pulsarbat as pb


@pytest.mark.parametrize("in_type, out_type", [(np.float32, np.complex64),
                                               (np.float64, np.complex128),
                                               (np.complex64, np.complex64),
                                               (np.complex128, np.complex128)])
@pytest.mark.parametrize("N", [128, 127])
def test_real_to_complex(in_type, out_type, N):
    x = np.random.normal(0, 1, N).astype(in_type)
    y = pb.utils.real_to_complex(x, axis=0)
    assert y.dtype == out_type
    assert y.shape[0] == (N + 1) // 2
