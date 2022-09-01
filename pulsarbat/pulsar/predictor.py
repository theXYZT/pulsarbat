"""Pulsar phase prediction."""

import operator
import functools
from dataclasses import dataclass, asdict
import numpy as np
from numpy.polynomial import Polynomial
from astropy import units as u
from astropy.time import Time
from astropy.table import QTable
import pulsarbat as pb

__all__ = ["PolycoEntry", "PhasePredictor"]


@dataclass
class PolycoEntry:
    """Entry for polynomial-based pulsar phase prediction.

    Parameters
    ----------
    psr : str
        Pulsar name.
    obs : str
        Observatory code.
    freq : Quantity
        Observing frequency.
    tmid : Time
        Timestamp at midpoint of span.
    span : Quantity
        Length of span.
    rphase : int
        Integer part of reference phase
    poly : numpy.polynomial.Polynomial
        Phase prediction polynomial (domain in units of seconds).
    """

    psr: str
    obs: str
    freq: u.Quantity
    tmid: Time
    span: u.Quantity
    rphase: int
    poly: Polynomial


class PhasePredictor(QTable):
    """Pulsar phase predictor.

    Parameters
    ----------
    data : sequence of PhasePredictorEntry
        A list of phase prediction entries. All entries must have the
        same 'psr', 'obs', 'freq', and 'span' values.
    """

    _descriptions = {
        "psr": "Pulsar name",
        "obs": "Observatory code",
        "freq": "Observing frequency",
        "tmid": "Time at midpoint of span",
        "span": "Length of span",
        "rphase": "Integer part of reference phase",
        "poly": "Phase prediction polynomial (domain in seconds)",
    }

    def __init__(self, data=None, *args, **kwargs):
        try:
            data = [asdict(x) for x in data]
            data = sorted(data, key=operator.itemgetter("tmid"))
        except (TypeError, KeyError):
            pass
        else:
            kwargs.setdefault("descriptions", self._descriptions)

        super().__init__(data, *args, **kwargs)

        for k in ["psr", "obs", "freq", "span"]:
            if k in self.colnames and len(np.unique(self[k])) > 1:
                d = self._descriptions[k]
                raise ValueError(f"All entries must have the same '{k}' ({d}).")

        self._intervals = None

    @property
    def intervals(self):
        """Intervals where the predictor is valid and should be used."""
        if self._intervals is None:
            tstart = self["tmid"] - self["span"] / 2
            tstop = self["tmid"] + self["span"] / 2
            intervals = sorted(zip(tstart, tstop), key=lambda x: x[1])

            merged = []
            start, end = intervals.pop()
            while intervals:
                next_start, next_end = intervals.pop()
                if next_end >= start or start.isclose(next_end, 1 * u.ms):
                    start = min(start, next_start)
                else:
                    merged.append([start, end])
                    start, end = next_start, next_end
            merged.append([start, end])
            self._intervals = tuple(reversed(merged))

        return self._intervals

    def _get_index_and_dt(self, times):
        """Check if timestamps are within predictor range."""
        check = ((a <= times) & (times <= b) for a, b in self.intervals)
        check = functools.reduce(operator.or_, check)

        if not np.all(check):
            raise ValueError("Some timestamps outside predictor range!")

        span_ends = self["tmid"] + self["span"] / 2
        index = np.searchsorted(span_ends.mjd, times.mjd)
        dt = (times - self["tmid"][index]).to_value(u.s)
        return index, dt

    def __call__(self, times):
        """Predict pulse phase at given times.

        Parameters
        ----------
        times : Time
            Timestamp(s) for which phases are to be predicted.

        Returns
        -------
        phase : pulsarbat.Phase
            Predicted phase for given timestamps.
        """
        index, dt = self._get_index_and_dt(times)

        if times.isscalar:
            ph1 = self["rphase"][index]
            ph2 = self["poly"][index](dt)
        else:
            ph1 = np.zeros(times.shape, dtype=np.int64)
            ph2 = np.zeros(times.shape, dtype=np.float64)
            for i in np.unique(index):
                s = index == i
                ph1[s] = self["rphase"][i]
                ph2[s] = self["poly"][i](dt[s])

        return pb.Phase(ph1, ph2)

    def phasepol(self, t0):
        """Phase prediction polynomial centered at given timestamp."""
        if not t0.isscalar:
            raise ValueError("Timestamp must be a scalar.")

        index, dt = self._get_index_and_dt(t0)
        rphase = self["rphase"][index]
        polynomial = self["poly"][index].copy()
        polynomial.domain -= dt
        a = int(polynomial(0) // 1)

        return (polynomial - a).convert(), pb.Phase(rphase + a)

    def f0(self, times, n=0):
        """Predict rotation frequency or its derivatives for given times."""
        index, dt = self._get_index_and_dt(times)
        unit = u.cycle / u.s ** (n + 1)

        if times.isscalar:
            f = self["poly"][index].deriv(n + 1)(dt)
        else:
            f = np.zeros(times.shape, dtype=np.float64)
            for i in np.unique(index):
                s = index == i
                f[s] = self["poly"][i].deriv(n + 1)(dt[s])
        return f * unit

    @classmethod
    def from_polyco(cls, path):
        """Create a PhasePredictor instance by reading tempo1-style polycos.

        Parameters
        ----------
        path : path-like or file-like
            Either a path to a polyco file or a file-like object that
            supports the ``readline()`` method.

        Examples
        --------
        From reading polyco files,

        >>> predictor = pb.PhasePredictor.from_polyco("polyco.dat")

        Alternatively, a text stream can be used (such as ``io.StringIO``),

        >>> import io
        >>> str_io = io.StringIO(polyco_text)
        >>> predictor = pb.PhasePredictor.from_polyco(str_io)

        Notes
        -----
        The format of a polyco entry (a file may have more than one) is

        .. code-block:: text

            ====  =======   ============================================
            Line  Columns   Item
            ====  =======   ============================================
            1        1-10   Pulsar Name
                    12-20   Date (dd-mmm-yy)
                    21-31   UTC (hhmmss.ss)
                    32-51   TMID (MJD)
                    52-72   Dispersion Measure (pc / cm^3)
                    74-79   Doppler shift due to earth motion (10^-4)
                    80-86   Log_10 of fit rms residual in periods
            2        1-20   Reference Phase (RPHASE)
                    22-38   Reference rotation frequency (F0)
                    39-43   Observatory number
                    44-49   Data span (minutes)
                    50-54   Number of coefficients (NCOEFF)
                    55-64   Observing frequency (MHz)
                    65-71   (Optional) Binary orbit phase
                    72-80   (Optional) Orbital frequency (1/day)
            3*       1-25   Coefficient 1 (COEFF(1))
                    26-50   Coefficient 2 (COEFF(2))
                    51-75   Coefficient 3 (COEFF(3))
            ====  =======   ============================================
            * Subsequent lines have three coefficients each, up to NCOEFF

        The pulse phase and frequency at time T (in MJD) are then calculated as::

            DT = (T-TMID)*1440
            PHASE = RPHASE + DT*60*F0 + COEFF(1) + DT*COEFF(2) + DT^2*COEFF(3) + ....
            FREQ(Hz) = F0 + (1/60)*(COEFF(2) + 2*DT*COEFF(3) + 3*DT^2*COEFF(4) + ....)

        Example tempo2 call to produce one:

        .. code-block:: text

            tempo2 -tempo1 -f pulsar.par
                -polyco "56499 56500 90 12 12 ao 1400"
                         |-- MJD start
                               |-- MJD end
                                     |-- length of span (in minutes)
                                        |-- Number of polynomial coefficients
                                           |-- Max Hour Angle (12 is continuous)
                                              |-- Observatory code
                                                 |-- Frequency in MHz

        References
        ----------
        http://tempo.sourceforge.net/ref_man_sections/tz-polyco.txt
        https://bitbucket.org/psrsoft/tempo2/
        """

        f = path if hasattr(path, "readline") else open(path, "r")
        d2e = str.maketrans("Dd", "ee")

        table = []
        with f:
            while (line := f.readline()) :
                psr, _, _, mjd_mid, dm, *_ = line.split()
                rphase, f0, obs, span, ncoeff, freq, *_ = f.readline().split()
                r_int, _, r_frac = rphase.partition(".")

                coeffs = []
                for _ in range(-(int(ncoeff) // -3)):
                    coeffs += f.readline().translate(d2e).split()

                coeffs = np.array(coeffs, dtype=np.float64)
                coeffs[0] += float("0." + r_frac)
                coeffs[1] += float(f0) * 60

                entry = PolycoEntry(
                    psr=psr,
                    obs=obs,
                    freq=float(freq) * u.MHz,
                    tmid=Time(mjd_mid, format="mjd", precision=9),
                    span=int(span) * u.min,
                    rphase=np.int64("0" + r_int),
                    poly=Polynomial(coeffs, domain=[-60, +60]).convert(),
                )

                table.append(entry)

        return cls(table)
