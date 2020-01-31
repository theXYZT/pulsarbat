# This code is written by Marten H. van Kerkwijk and licensed under
# GNU GPL v3.0. It's copied directly from the mhvk/pulsar repository.
"""Read in and use tempo1 polyco files (tempo2 predict to come).

Examples
--------
>>> psr_polyco = predictor.Polyco('polyco_new.dat')
>>> predicted_phase = psr_polyco(time)

>>> phasepol = psr_polyco.phasepol(Timeindex, rphase='fraction')

For use with folding codes with times since some start time t0 in seconds:

>>> psr_polyco.phasepol(t0, 'fraction', t0=t0, time_unit=u.s, convert=True)

Notes
-----
The format of the polyco files is (from
http://tempo.sourceforge.net/ref_man_sections/tz-polyco.txt)
Line  Columns     Item
----  -------   -----------------------------------
1      1-10   Pulsar Name
      11-19   Date (dd-mmm-yy)
      20-31   UTC (hhmmss.ss)
      32-51   TMID (MJD)
      52-72   DM
      74-79   Doppler shift due to earth motion (10^-4)
      80-86   Log_10 of fit rms residual in periods
2      1-20   Reference Phase (RPHASE)
      21-38   Reference rotation frequency (F0)
      39-43   Observatory number
      44-49   Data span (minutes)
      50-54   Number of coefficients
      55-75   Observing frequency (MHz)
      76-80   Binary phase
3-     1-25   Coefficient 1 (COEFF(1))
      26-50   Coefficient 2 (COEFF(2))
      51-75   Coefficient 3 (COEFF(3))

The pulse phase and frequency at time T are then calculated as:
DT = (T-TMID)*1440
PHASE = RPHASE + DT*60*F0 + COEFF(1) + DT*COEFF(2) + DT^2*COEFF(3) + ....
FREQ(Hz) = F0 + (1/60)*(COEFF(2) + 2*DT*COEFF(3) + 3*DT^2*COEFF(4) + ....)

Example tempo2 call to produce one:

tempo2 -tempo1 \
    -f ~/packages/scintellometry/scintellometry/ephemerides/psrb1957+20.par \
    -polyco "56499 56500 300 12 12 aro 150.0"

#            MJDstart
#                  MJDend
#                        #min for which polynomial is fit
#                            #deg of polynomial
#                               #max HA (12 is continuous)
#                                  Observatory
#                                     Frequency
"""

from collections import OrderedDict
import numpy as np
from numpy.polynomial import Polynomial
from astropy.table import Table
import astropy.units as u
from astropy.time import Time


class Polyco(Table):
    def __init__(self, name):
        """Read in polyco file as Table, and set up class."""
        super(Polyco, self).__init__(polyco2table(name))

    def __call__(self, time, index=None, rphase=None, deriv=0, time_unit=None):
        """Predict phase or frequency (derivatives) for given mjd (array)

        Parameters
        ----------
        mjd_in : Time or `float` (array)
            Time instances of MJD's for which phases are to be generated.
            If `float`, assumed to be MJD (NOTE: less precise!)
        index : int (array), None, or float/Time
            indices into Table for corresponding polyco's; if None, it will be
            deterined from `mjd_in` (giving an explicit index can help speed up
            the evaluation).  If not an index or `None`, it will be used to
            find the index. Hence if one has a large array if closely spaced
            times, one can pass in a single element to speed matters up.
        rphase : None or 'fraction' or float (array)
            phase zero points for relevant polyco's; if None, use those
            stored in polyco.  (Those are typically large, so one looses
            some precision.)  Can also set 'fraction' or give the zero point.
        deriv : int
            Derivative to return (Default=0=phase, 1=frequency, etc.)
        time_unit : Unit
            Unit of time in which derivatives are expressed (Default: second)

        Returns
        -------
        phase / time**deriv
            In units of
        """
        time_unit = time_unit or u.s
        if not hasattr(time, 'mjd'):
            time = Time(time, format='mjd', scale='utc')
        try:  # This also catches index=None
            index = index.__index__()
        except (AttributeError, TypeError):
            index = self.searchclosest(time)

        mjd_mid = self['mjd_mid'][index]

        if np.any(np.abs(time.mjd - mjd_mid) * 1440 > self['span'][index] / 2):
            raise ValueError('(some) MJD outside of polyco range')

        if time.isscalar:
            polynomial = self.polynomial(index, rphase, deriv)
            dt = (time -
                  Time(self['mjd_mid'][index], format='mjd', scale='utc'))
            return (polynomial(dt.to(u.min).value) * u.cycle /
                    u.min**deriv).to(u.cycle / time_unit**deriv)
        else:
            results = np.zeros(len(time)) * u.cycle / time_unit**deriv
            for j in set(index):
                in_set = index == j
                polynomial = self.polynomial(j, rphase, deriv)
                dt = (time[in_set] -
                      Time(self['mjd_mid'][j], format='mjd', scale='utc'))
                results[in_set] = (polynomial(dt.to(u.min).value) * u.cycle /
                                   u.min**deriv)
            return results

    def polynomial(self,
                   index,
                   rphase=None,
                   deriv=0,
                   t0=None,
                   time_unit=u.min,
                   out_unit=None,
                   convert=False):
        """Prediction polynomial set up for times in MJD

        Parameters
        ----------
        index : int or float
            index into the polyco table (or MJD for finding closest)
        rphase : None or 'fraction' or float
            phase zero point; if None, use the one stored in polyco.
            (Those are typically large, so one looses some precision.)
            Can also set 'fraction' to use the stored one modulo 1, which is
            fine for folding, but breaks cycle count continuity between sets.
        deriv : int
            derivative of phase to take (1=frequency, 2=fdot, etc.); default 0

        Returns
        -------
        polynomial : Polynomial
            set up for MJDs between mjd_mid ± span

        Notes
        -----
        Units for the polynomial are cycles/second**deriv.  Taking a derivative
        outside will be per day (e.g., self.polynomial(1).deriv() gives
        frequencies in cycles/day)
        """

        out_unit = out_unit or time_unit

        try:
            index = index.__index__()
        except (AttributeError, TypeError):
            index = self.searchclosest(index)
        window = np.array([-1, 1]) * self['span'][index] / 2 * u.min

        polynomial = Polynomial(self['coeff'][index], window.value,
                                window.value)
        polynomial.coef[1] += self['f0'][index] * 60.

        if deriv == 0:
            if rphase is None:
                polynomial.coef[0] += self['rphase'][index]
            elif rphase == 'fraction':
                polynomial.coef[0] += self['rphase'][index] % 1
            else:
                polynomial.coef[0] = rphase
        else:
            polynomial = polynomial.deriv(deriv)
            polynomial.coef /= u.min.to(out_unit)**deriv

        if t0 is None:
            dt = 0. * time_unit
        elif not hasattr(t0, 'jd1') and t0 == 0:
            dt = (-self['mjd_mid'][index] * u.day).to(time_unit)
        else:
            dt = ((t0 - Time(self['mjd_mid'][index], format='mjd',
                             scale='utc')).jd * u.day).to(time_unit)

        polynomial.domain = (window.to(time_unit) - dt).value

        if convert:
            return polynomial.convert()
        else:
            return polynomial

    def phasepol(self,
                 index,
                 rphase=None,
                 t0=0.,
                 time_unit=u.day,
                 convert=False):
        """Phase prediction polynomial set up for times in MJD

        Parameters
        ----------
        index : int or float
            index into the polyco table (or MJD for finding closest)
        rphase : None or 'fraction' or float
            phase zero point; if None, use the one stored in polyco.
            (Those are typically large, so one looses some precision.)
            Can also set 'fraction' to use the stored one modulo 1, which is
            fine for folding, but breaks phase continuity between sets.

        Returns
        -------
        phasepol : Polynomial
            set up for MJDs between mjd_mid ± span
        """
        return self.polynomial(index,
                               rphase,
                               t0=t0,
                               time_unit=time_unit,
                               convert=convert)

    def fpol(self, index, t0=0., time_unit=u.day, convert=False):
        """Frequency prediction polynomial set up for times in MJD

        Parameters
        ----------
        index : int
            index into the polyco table

        Returns
        -------
        freqpol : Polynomial
            set up for MJDs between mjd_mid ± span
        """
        return self.polynomial(index,
                               deriv=1.,
                               t0=t0,
                               time_unit=time_unit,
                               out_unit=u.s,
                               convert=convert)

    def searchclosest(self, mjd):
        """Find index to polyco that is closest in time to (set of) Time/MJD"""
        mjd = getattr(mjd, 'mjd', mjd)
        i = np.clip(np.searchsorted(self['mjd_mid'], mjd), 1, len(self) - 1)
        i -= mjd - self['mjd_mid'][i - 1] < self['mjd_mid'][i] - mjd
        return i


def polyco2table(name):
    """Read in a tempo1,2 polyco file and convert it to a Table

    Parameters
    ----------
    name : string
        file name holding polyco data

    Returns
    -------
    t : Table
        each entry in the polyco file corresponds to one row, with columns
        psr, date, utc_mid, mjd_mid, dm, vbyc_earth, lgrms,
        rphase, f0, obs, span, ncoeff, freq, binphase, coeff[ncoeff]
    """

    with open(name, 'r') as polyco:
        line = polyco.readline()
        t = None
        while line != '':
            d = OrderedDict(
                zip([
                    'psr', 'date', 'utc_mid', 'mjd_mid', 'dm', 'vbyc_earth',
                    'lgrms'
                ], line.split()))
            d.update(
                dict(
                    zip([
                        'rphase', 'f0', 'obs', 'span', 'ncoeff', 'freq',
                        'binphase'
                    ],
                        polyco.readline().split()[:7])))
            for key in d:
                try:
                    d[key] = int(d[key])
                except ValueError:
                    try:
                        d[key] = float(d[key])
                    except ValueError:
                        pass
            d['coeff'] = []
            while len(d['coeff']) < d['ncoeff']:
                d['coeff'] += polyco.readline().split()

            d['coeff'] = np.array([float(item) for item in d['coeff']])

            if t is None:
                t = Table([[v] for v in d.values()], names=d.keys())
            else:
                t.add_row(d.values())

            line = polyco.readline()

    return t
