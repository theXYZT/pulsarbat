"""Tests for `pulsarbat.RadioSignal` and subclasses."""

import pytest
import numpy as np
import astropy.units as u
import pulsarbat as pb
from astropy.time import Time


@pytest.mark.parametrize("ref_freq", [3.95, 4, 4.05] * u.GHz)
@pytest.mark.parametrize("dm", [1, 10])
def test_coherent_dedispersion_reversibility(ref_freq, dm):
    shape = (8192, 4)
    x = np.random.normal(0, 1, shape) + np.random.normal(0, 1, shape) * 1j
    DM = pb.DispersionMeasure(dm)
    z = pb.BasebandSignal(x, sample_rate=16*u.MHz, center_freq=4*u.GHz)

    y = pb.coherent_dedispersion(z, DM, ref_freq=ref_freq)
    y = pb.coherent_dedispersion(y, -DM, ref_freq=ref_freq)


@pytest.mark.parametrize("ref_freq", [3, 5, 7] * u.GHz)
@pytest.mark.parametrize("dm", [100, -100])
def test_incoherent_dedispersion_reversibility(ref_freq, dm):
    kw = {'sample_rate': 10 * u.kHz,
          'center_freq': 5 * u.GHz,
          'chan_bw': 10 * u.MHz,
          'start_time': Time.now()}

    x = pb.RadioSignal(np.random.standard_normal((4096, 32)), **kw)
    DM = pb.DispersionMeasure(dm)

    y = pb.incoherent_dedispersion(x, DM, ref_freq=ref_freq)
    y = pb.incoherent_dedispersion(y, -DM, ref_freq=ref_freq)

    i = int(((y.start_time - x.start_time) / x.dt).to_value(u.one).round())
    assert np.allclose(x.data[i:i + len(y)], y)
