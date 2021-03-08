"""Tests for `Polyco` and `Phase`."""

import pytest
from collections import Counter
from pathlib import Path
import numpy as np
import astropy.units as u
from astropy.time import Time
import pulsarbat as pb
from pulsarbat.pulsar import Phase

TEST_POLYCO = Path(__file__).parent.absolute() / 'data' / 'timing.dat'


class TestPolyco:
    def test_basic(self):
        p = pb.pulsar.Polyco(TEST_POLYCO)
        assert np.all(pb.pulsar.Polyco(p) == p)

        t = Time('2018-05-06T12:00:00.000', format='isot', precision=9)
        ph = p(t)
        assert type(ph) is Phase
        assert ph.int.to_value(u.cycle).astype(np.int64) == 146726403657
        assert np.isclose(ph.frac.to_value(u.cycle), -0.151982394)
        assert u.isclose(p(t, deriv=1), 641.97319469 * u.cycle/u.s, rtol=1E-8)

        t = Time('2018-05-06T2:00:00.000', format='isot', precision=9)
        ph = p(t)
        assert ph.int.to_value(u.cycle).astype(np.int64) == 146703292586
        assert np.isclose(ph.frac.to_value(u.cycle), +0.49273169)
        assert u.isclose(p(t, deriv=1), 641.97469967 * u.cycle/u.s, rtol=1E-8)

        with pytest.raises(ValueError):
            t = Time('2017-05-06T2:00:00.000', format='isot', precision=9)
            _ = p(t)

    def test_array(self):
        sr = 1 * u.MHz
        t0 = Time('2018-05-06T12:00:00.000', format='isot', precision=9)
        ts = t0 + np.arange(4096) / sr

        p = pb.pulsar.Polyco(TEST_POLYCO)
        ph = p(ts)
        assert ph.shape == (4096,)

        c = Counter(np.int64(ph.int.value))
        assert c[146726403657] == 1016
        assert c[146726403658] == 1558
        assert c[146726403659] == 1522


class TestPhase:
    def test_basic(self):
        a = Phase(1.2)
        repr(a)
        str(a)
        assert a.isscalar
        assert u.allclose(a.int, 1 * u.cycle)
        assert u.allclose(a.frac, 0.2 * u.cycle)
        assert u.allclose(a.cycle, 1.2 * u.cycle)
        assert u.allclose(a.to(u.deg), 432 * u.deg)

        a = Phase([1.2, 1.8, 2.6])
        repr(a)
        str(a)
        assert u.allclose(a.int, [1, 2, 3] * u.cycle)
        assert u.allclose(a.frac, [0.2, -0.2, -0.4] * u.cycle)
        assert u.allclose(a.cycle, [1.2, 1.8, 2.6] * u.cycle)
        assert u.allclose(a.to(u.deg), [432, 648, 936] * u.deg)

        assert np.array_equal(np.int64(a.int.value), np.array([1, 2, 3]))
        assert u.allclose(Phase(['49.64', '12.34']), Phase([49.64, 12.34]))

    def test_math_operations(self):
        a = Phase([1.2, 1.8, 2.6]) + Phase([3.4, 5.6, 2.3])
        b = Phase([4.6, 7.4, 4.9])
        assert u.allclose(a, b)

        a = Phase(1.2)
        assert u.isclose(4 * a, Phase(4 * 1.2))
        assert u.isclose(a + 0.8, Phase(1.2 + 0.8))

        b = Phase(0.3, 0.5)
        assert u.isclose(a + b, Phase(2.0))

        assert u.isclose(Phase(100/3) + Phase(10/3), Phase(110/3))
        assert u.isclose(np.exp(1j * Phase(0.5)), -1)
        assert u.isclose(np.exp(1j * Phase(1.0)), +1)
