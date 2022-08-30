"""Tests for pulsar phase prediction."""

import pytest
from io import StringIO
from pathlib import Path
import numpy as np
import astropy.units as u
from astropy.time import Time
import pulsarbat as pb
import cloudpickle

TEST_POLYCO = Path(__file__).parent.absolute() / "data" / "timing.dat"

ENTRY_TEXT = """B1937+21    7-May-18   0.00   58245.00000000000   71.020168
 146754136477.666475  641.928232294317   ao   90   12   327.000
 -3.17034199847385061e-07  2.76360291261698521e+00  8.05424212611731503e-05
 -1.14853014406135967e-07 -1.39769248548540950e-10  6.39552923641417649e-13
  3.19619782475082226e-15 -5.35166928586675360e-16 -4.58065943719761444e-19
  2.26855569952374124e-19 -4.63024751309515689e-23 -3.42478559749583972e-23
"""

INVALID = """B1937+21    7-May-18   93600.00   58245.40000000000   71.020167
 146776323107.982696  641.928232294317   ao  288   12   327.000
  2.53531626087091441e-08  2.71634485209303600e+00 -1.18579224383567124e-04
  2.13006534220737831e-08  1.82945349045995736e-10 -2.17093444265104196e-14
 -1.19143951096548092e-16  8.75742271090786040e-20  6.90038715651136990e-23
 -3.13982943744315582e-24 -4.22839364952768232e-28  5.14827898395789200e-29
B1937+21    7-May-18  163000.00   58245.68750000000            71.020167
 146792269715.610996  641.928232294317   ao   90   12   327.000
 -4.76040685223260378e-07  2.67040349305271674e+00  3.73410078415526979e-05
  1.57043805946084180e-07 -5.08960654800334953e-11 -3.36505250137997982e-13
 -1.89629091176250827e-14  1.96470551312769751e-16  9.40225741785451917e-18
 -9.15695855709336387e-20 -1.59591049042759423e-21  1.52097772879173001e-23
"""

phase_tols = dict(rtol=0, atol=1E-8 * u.cycle)


class TestPredictor:
    def test_basic(self):
        p = pb.PhasePredictor.from_polyco(TEST_POLYCO)
        assert np.all(pb.PhasePredictor(p) == p)
        assert len(p.intervals) == 1

        q = cloudpickle.loads(cloudpickle.dumps(p))
        assert np.all(q == p)

        t = Time("58245.375", format="mjd", precision=9)
        ph, f0, f1 = p(t), p.f0(t), p.f0(t, n=1)

        assert type(ph) is pb.Phase
        assert int(ph.int.cycle) == 146774936445
        assert np.isclose(ph.frac.cycle, 0.058161699852649296)
        assert u.isclose(f0, 641.973647812571 * u.cycle / u.s, rtol=1e-8)
        assert u.isclose(f1, -6.635997412662843e-08 * u.cycle / u.s ** 2, rtol=1e-8)

        ts = t + np.arange(10000) * u.us
        ph = p(ts)
        assert len(ph) == len(ts)
        assert int(ph[-1].int.cycle) == 146774936451
        assert np.isclose(ph[-1].frac.cycle, 0.4772562027766636)

        ts = Time("58244.91", format="mjd") + np.arange(1400) * u.min
        ph, f0 = p(ts), p.f0(ts)
        assert int(ph[-1].int.cycle) - int(ph[0].int.cycle) == 53887265
        assert np.all(f0 > 641.97269 * u.cycle / u.s)
        assert np.all(f0 < 641.97455 * u.cycle / u.s)
        assert f0.ptp() < (0.001858 * u.cycle / u.s)

        q = p[[0, 1, 2, 4, 5, 6, 8, 9]]
        assert len(q.intervals) == 3

        with pytest.raises(ValueError):
            t = Time("60000", format="mjd")
            _ = p(t)

    def test_stringio(self):
        p = pb.PhasePredictor.from_polyco(StringIO(ENTRY_TEXT))
        ref = pb.Phase(146754136477, 0.666475) + (-3.17034199847385061e-07)
        assert u.isclose(ref, p(p["tmid"][0]), **phase_tols)

        text = "this is not a polyco"
        with pytest.raises(ValueError):
            _ = pb.PhasePredictor.from_polyco(StringIO(text))

        with pytest.raises(ValueError):
            _ = pb.PhasePredictor.from_polyco(StringIO(INVALID))

    def test_phasepol(self):
        p = pb.PhasePredictor.from_polyco(StringIO(ENTRY_TEXT))
        t = p[0]["tmid"]
        pol, ref = p.phasepol(t)
        assert u.isclose(p(t + 1 * u.s), ref + pol(1), **phase_tols)
        assert u.isclose(p(t + 8 * u.s), ref + pol(8), **phase_tols)
        assert u.isclose(p(t + 1 * u.ms), ref + pol(0.001), **phase_tols)

        with pytest.raises(ValueError):
            _ = p.phasepol(t + np.arange(10) * u.ms)

    def test_polyco_entry(self):
        t = Time("60000", format="mjd")

        entry = pb.PolycoEntry(
            psr="Fake",
            obs="@",
            freq=1 * u.GHz,
            tmid=t,
            span=1 * u.day,
            rphase=0,
            poly=np.polynomial.Polynomial([0, 1]),
        )

        p = pb.PhasePredictor([entry])
        assert u.isclose(p(t), pb.Phase(0), **phase_tols)
        assert u.isclose(p(t + 1 * u.s), pb.Phase(1), **phase_tols)
        assert u.isclose(p(t + 1 * u.ms), pb.Phase(0.001), **phase_tols)
        assert u.isclose(p(t + 1 * u.hr), pb.Phase(3600), **phase_tols)

        assert u.isclose(p.f0(t), 1 * u.cycle / u.s)
        assert u.isclose(p.f0(t + 1 * u.hr), 1 * u.cycle / u.s)

        assert u.isclose(p.f0(t, n=1), 0 * u.cycle / u.s**2)
        assert u.isclose(p.f0(t + 1 * u.hr, n=1), 0 * u.cycle / u.s**2)


class TestPhase:
    def test_basic(self):
        a = pb.Phase(1.2)
        repr(a)
        str(a)
        assert a.isscalar
        assert u.allclose(a.int, 1 * u.cycle)
        assert u.allclose(a.frac, 0.2 * u.cycle)
        assert u.allclose(pb.FractionalPhase(a), 0.2 * u.cycle)
        assert u.allclose(a.cycle, 1.2 * u.cycle)
        assert u.allclose(a.to(u.deg), 432 * u.deg)

        a = pb.Phase([1.2, 1.8, 2.6])
        repr(a)
        str(a)
        assert u.allclose(a.int, [1, 2, 3] * u.cycle)
        assert u.allclose(a.frac, [0.2, -0.2, -0.4] * u.cycle)
        assert u.allclose(pb.FractionalPhase(a), [0.2, -0.2, -0.4] * u.cycle)
        assert u.allclose(a.cycle, [1.2, 1.8, 2.6] * u.cycle)
        assert u.allclose(a.to(u.deg), [432, 648, 936] * u.deg)

        assert np.array_equal(np.int64(a.int.value), np.array([1, 2, 3]))

        a = pb.Phase(np.arange(10))
        assert u.allclose(a.astype(np.int64), np.arange(10) * u.cycle)
        assert u.allclose(a.astype(np.float64), np.arange(10) * u.cycle)

    def test_from_string(self):
        def check_phase(ph, i, f):
            assert np.int64(ph.int.value) == i
            assert np.float64(ph.frac.value) == f

        assert u.allclose(pb.Phase(["49.64", "12.34"]), pb.Phase([49.64, 12.34]))

        s = "23424249845.323429437454"
        sx = [s, f"{s}E2", f"{s}e2", f"{s}E+2", f"{s}E-2"]
        ix = np.int64(
            [23424249845, 2342424984532, 2342424984532, 2342424984532, 234242498]
        )
        fx = np.float64(
            [0.323429437454, 0.3429437454, 0.3429437454, 0.3429437454, 0.45323429437454]
        )

        for s, i, f in zip(sx, ix, fx):
            check_phase(pb.Phase.from_string(s), i, f)

        ph = pb.Phase.from_string(sx)
        assert np.array_equal(np.int64(ph.int.value), ix)
        assert np.array_equal(np.float64(ph.frac.value), fx)

    def test_math_operations(self):
        a = pb.Phase([1.2, 1.8, 2.6]) + pb.Phase([3.4, 5.6, 2.3])
        b = pb.Phase([4.6, 7.4, 4.9])
        assert u.allclose(a, b)

        a = pb.Phase(1.2)
        assert u.isclose(4 * a, pb.Phase(4 * 1.2))
        assert u.isclose(a + 0.8, pb.Phase(1.2 + 0.8))

        b = pb.Phase(0.3, 0.5)
        assert u.isclose(a + b, pb.Phase(2.0))

        b = pb.Phase(pb.Phase(0.3), pb.Phase(0.5))
        assert u.isclose(a + b, pb.Phase(2.0))

        assert u.isclose(pb.Phase(100 / 3) + pb.Phase(10 / 3), pb.Phase(110 / 3))
        assert u.isclose(np.exp(1j * pb.Phase(0.5)), -1)
        assert u.isclose(np.exp(1j * pb.Phase(1.0)), +1)

    def test_common_operations(self):
        ph = pb.Phase([1.3, 2.3, 12.7, 0.1, 5.6, 6.8])
        assert ph.min() == pb.Phase(0.1)
        assert ph.max() == pb.Phase(12.7)
        ph2 = ph.sort()
        assert ph2[0] == pb.Phase(0.1)
        assert ph2[-1] == pb.Phase(12.7)
