"""Tests for `Polyco` and `Phase`."""

import pytest
from pathlib import Path
import numpy as np
import astropy.units as u
from astropy.time import Time
import pulsarbat as pb
from pulsarbat.pulsar import Phase

TEST_POLYCO = Path(__file__).parent.absolute() / 'data' / 'timing.dat'


def test_polyco_simple():
    p = pb.pulsar.Polyco(TEST_POLYCO)
    assert np.all(pb.pulsar.Polyco(p) == p)

    t = Time('2018-05-06T12:00:00.000', format='isot', precision=9)
    ph = p(t)
    assert type(ph) is Phase
    assert ph.int.to_value(u.cycle).astype(np.int64) == 146726403657
    assert np.isclose(ph.frac.to_value(u.cycle), -0.151982394)
    assert u.isclose(p(t, deriv=1), 641.97319469 * u.cycle / u.s, rtol=1E-8)

    t = Time('2018-05-06T2:00:00.000', format='isot', precision=9)
    ph = p(t)
    assert ph.int.to_value(u.cycle).astype(np.int64) == 146703292586
    assert np.isclose(ph.frac.to_value(u.cycle), +0.49273169)
    assert u.isclose(p(t, deriv=1), 641.97469967 * u.cycle / u.s, rtol=1E-8)

    with pytest.raises(ValueError):
        t = Time('2017-05-06T2:00:00.000', format='isot', precision=9)
        _ = p(t)


def test_phase_simple():
    assert 4 * Phase(1.5) == Phase(6.0) == 6
    assert Phase(4.5) + 1.5 == Phase(6.0) == 6
    assert u.isclose(Phase(100/3) + Phase(10/3), Phase(110/3))
    assert u.isclose(np.exp(1j * Phase(0.5)), -1)
    assert u.isclose(np.exp(1j * Phase(1.0)), +1)
