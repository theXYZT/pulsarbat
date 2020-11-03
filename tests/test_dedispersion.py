"""Tests for `pulsarbat.RadioSignal` and subclasses."""

# flake8: noqa

import pytest
import numpy as np
import dask.array as da
import astropy.units as u
import pulsarbat as pb
from astropy.time import Time
from scipy.signal import sosfilt, butter


@pytest.mark.parametrize("ref_freq", [3, 5, 7] * u.GHz)
@pytest.mark.parametrize("dm", [100, -100])
def test_incoherent_dedispersion_reversibility(ref_freq, dm):
    kw = {'sample_rate': 10 * u.kHz,
          'center_freq': 5 * u.GHz,
          'bandwidth': 2 * u.GHz,
          'start_time': Time.now()}

    x = pb.RadioSignal(np.random.standard_normal((4096, 32)), **kw)
    DM = pb.DispersionMeasure(dm)

    y = pb.incoherent_dedispersion(x, DM, ref_freq=ref_freq)
    y = pb.incoherent_dedispersion(y, -DM, ref_freq=ref_freq)

    i = int(((y.start_time - x.start_time) / x.dt).to_value(u.one).round())
    assert np.allclose(x.data[i:i + len(y)], y)
