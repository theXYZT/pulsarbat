# This code is written by Marten H. van Kerkwijk and licensed under
# GNU GPL v3.0. It is copied from the 'scintillometry' package
# repository at: https://github.com/mhvk/scintillometry

# This is done to avoid forcing the user to install scintillometry also.

r"""Read in and use tempo1 polyco files.

Examples
--------
>>> psr_polyco = predictor.Polyco('polyco_new.dat')
>>> predicted_phase = psr_polyco(time)

>>> phasepol = psr_polyco.phasepol(Timeindex, rphase='fraction')

For use with folding codes with times since some start time t0 in seconds:

>>> psr_polyco.phasepol(t0, 'fraction', t0=t0, time_unit=u.second, convert=True)

Notes
-----
The format of the polyco files is (from
http://tempo.sourceforge.net/ref_man_sections/tz-polyco.txt)

.. code-block:: text

    Line  Columns Item
    ----  ------- -----------------------------------
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

The pulse phase and frequency at time T are then calculated as::

    DT = (T-TMID)*1440
    PHASE = RPHASE + DT*60*F0 + COEFF(1) + DT*COEFF(2) + DT^2*COEFF(3) + ....
    FREQ(Hz) = F0 + (1/60)*(COEFF(2) + 2*DT*COEFF(3) + 3*DT^2*COEFF(4) + ....)

Example tempo2 call to produce one:

.. code-block:: text

    tempo2 -tempo1 \
        -f ~/packages/scintellometry/scintellometry/ephemerides/psrb1957+20.par \
        -polyco "56499 56500 300 12 12 aro 150.0"
                 |-- MJD start
                       |-- MJD end
                             |-- number of minutes for which polynomial is fit
                                 |-- degree of the polynomial
                                    |-- maxium Hour Angle (12 is continuous)
                                       |-- Observatory
                                           |-- Frequency in MHz
"""

from __future__ import division, print_function

from collections import OrderedDict

import numpy as np
from numpy.polynomial import Polynomial
from astropy import units as u
from astropy.table import QTable
from astropy.coordinates import Angle
from astropy.time import Time

from ..dm import DispersionMeasure
from .phase import Phase


__doctest_skip__ = ['*']
__all__ = ['Polyco']


class Polyco(QTable):
    def __init__(self, *args, **kwargs):
        """Read in polyco file as Table, and set up class."""
        if len(args):
            data = args[0]
            args = args[1:]
        else:
            data = kwargs.pop('data', None)

        if isinstance(data, str):
            data = polyco2table(data)

        super().__init__(data, *args, **kwargs)

    def to_polyco(self, name='polyco.dat', style='tempo2'):
        """Write the polyco table to a polyco file.

        Parameters
        ----------
        name : str
            Filename
        style : {'tempo1'|'tempo2'}, optional
            Package which the writer should emulate.  Default: 'tempo2'
        """
        header_fmt = ''.join(
            [('{' + key + converter['fmt'] + ('}\n' if key == 'lgrms' else '}'))
             for key, converter in converters.items()
             if key in self.keys() or key in ('date', 'utc_mid')])

        coeff_fmt = fortran_fmt if style == 'tempo1' else '{:24.17e}'.format

        with open(name, 'w') as fh:
            for row in self:
                items = {k: row[k] for k in converters if k in self.keys()}
                # Special treatment for mjd_mid, date, and utc_mid.
                mjd_mid = items['mjd_mid']
                # Hack: unlike Time, Phase can format its int/frac as {:..f}.
                items['mjd_mid'] = Phase(mjd_mid.jd1-2400000.5, mjd_mid.jd2)
                item = mjd_mid.datetime.strftime('%d-%b-%y')
                if style == 'tempo1':
                    item = item.upper()
                items['date'] = item if item[0] != '0' else ' '+item[1:]
                mjd_mid.precision = 2
                items['utc_mid'] = float(mjd_mid.isot.split('T')[1]
                                         .replace(':', ''))

                fh.write(header_fmt.format(**items) + '\n')

                coeff = row['coeff']
                for i in range(0, len(coeff), 3):
                    fh.write(' ' + ' '.join([coeff_fmt(c)
                                             for c in coeff[i:i+3]]) + '\n')

    def __call__(self, time, index=None, rphase=None, deriv=0, time_unit=None):
        """Predict phase or frequency (derivatives) for given mjd (array)

        Parameters
        ----------
        mjd_in : `~astropy.time.Time` or float (array)
            Time instances of MJD's for which phases are to be generated.
            If float, assumed to be MJD (NOTE: less precise!)
        index : int (array), None, float, or `~astropy.time.Time`
            indices into Table for corresponding polyco's; if None, it will be
            deterined from ``mjd_in`` (giving an explicit index can help speed
            up the evaluation).  If not an index or `None`, it will be used to
            find the index. Hence if one has a large array if closely spaced
            times, one can pass in a single element to speed matters up.
        rphase : None, 'fraction', 'ignore', or float (array)
            Phase zero point; if None, use the one stored in polyco
            (those are typically large, so we ensure we preserve precision by
            using the `~scintillometry.phases.Phase` class for the result.)
            Can also set 'fraction' to use the stored one modulo 1, which is
            fine for folding, but breaks cycle count continuity between sets,
            'ignore' for just keeping the value stored in the coefficients,
            or a value that should replace the zero point.
        deriv : int
            Derivative to return (Default=0=phase, 1=frequency, etc.)
        time_unit : Unit
            Unit of time in which derivatives are expressed (Default: second)

        Returns
        -------
        result : `~scintillometry.phases.Phase` or `~astropy.units.Quantity`
            A phase if ``deriv=0`` and ``rphase=None`` to preserve precision;
            otherwise, a quantity with units of ``cycle / time_unit**deriv``.
        """
        time_unit = time_unit or u.s
        if not hasattr(time, 'mjd'):
            time = Time(time, format='mjd', scale='utc')
        try:  # This also catches index=None
            index = index.__index__()
        except (AttributeError, TypeError):
            index = self.searchclosest(time)

        # Convert offsets to minutes for later use in polynomial evaluation.
        dt = (time - self['mjd_mid'][index]).to(u.min)
        if np.any(dt > self['span'][index]/2):
            raise ValueError('(some) MJD outside of polyco range')

        # Check whether we need to add the reference phase at the end.
        do_phase = (deriv == 0 and rphase is None)
        if do_phase:
            # If so, do not add it inside the polynomials.
            rphase = 'ignore'

        if time.isscalar:
            result = self.polynomial(index, rphase, deriv)(dt.value)
        else:
            result = np.zeros(time.shape)
            for j in set(index):
                sel = index == j
                result[sel] = self.polynomial(j, rphase, deriv)(dt[sel].value)

        # Apply units from the polynomials.
        result = result << u.cycle/u.min**deriv
        # Convert to requested unit in-place.
        result <<= u.cycle/time_unit**deriv
        # Add reference phase to it if needed.
        return result + self['rphase'][index] if do_phase else result

    def polynomial(self, index, rphase=None, deriv=0,
                   t0=None, time_unit=u.min, out_unit=None,
                   convert=False):
        """Prediction polynomial set up for times in MJD

        Parameters
        ----------
        index : int or float
            index into the polyco table (or MJD for finding closest)
        rphase : None or 'fraction' or 'ignore' or float
            Phase zero point; if None, use the one stored in polyco.
            (Those are typically large, so one looses some precision.)
            Can also set 'fraction' to use the stored one modulo 1, which is
            fine for folding, but breaks cycle count continuity between sets,
            'ignore' for just keeping the value stored in the coefficients,
            or a value that should replace the zero point.
        deriv : int
            derivative of phase to take (1=frequency, 2=fdot, etc.); default 0

        Returns
        -------
        polynomial : Polynomial
            set up for MJDs between mjd_mid +/- span

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
        window = np.array([-1, 1]) * self['span'][index]/2

        polynomial = Polynomial(self['coeff'][index],
                                window.value, window.value)
        polynomial.coef[1] += self['f0'][index].to_value(u.cycle/u.minute)

        if deriv == 0:
            if rphase is None:
                polynomial.coef[0] += self['rphase'][index].value
            elif rphase == 'fraction':
                polynomial.coef[0] += self['rphase']['frac'][index].value % 1
            elif rphase != 'ignore':
                polynomial.coef[0] = rphase
        else:
            polynomial = polynomial.deriv(deriv)
            polynomial.coef /= u.min.to(out_unit)**deriv

        if t0 is not None:
            dt = Time(t0, format='mjd') - self['mjd_mid'][index]
            polynomial.domain = (window - dt).to(time_unit).value

        if convert:
            return polynomial.convert()
        else:
            return polynomial

    def phasepol(self, index, rphase=None, t0=0., time_unit=u.day,
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
            set up for MJDs between mjd_mid +/- span
        """
        return self.polynomial(index, rphase, t0=t0, time_unit=time_unit,
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
            set up for MJDs between mjd_mid +/- span
        """
        return self.polynomial(index, deriv=1,
                               t0=t0, time_unit=time_unit, out_unit=u.s,
                               convert=convert)

    def searchclosest(self, mjd):
        """Find index to polyco that is closest in time to (set of) Time/MJD"""
        mjd = getattr(mjd, 'mjd', mjd)
        mjd_mid = self['mjd_mid'].mjd
        i = np.clip(np.searchsorted(mjd_mid, mjd), 1, len(self)-1)
        i -= mjd-mjd_mid[i-1] < mjd_mid[i]-mjd
        return i


def int_frac(s):
    mjd_int, _, frac = s.strip().partition('.')
    return np.array((int('0' + mjd_int), float('0.' + frac)),
                    dtype=[('int', int), ('frac', float)])


def change_type(cls, **kwargs):
    def convert(x):
        if x.dtype.names:
            args = [x[k] for k in x.dtype.names]
        else:
            args = [x]
        return cls(*args, **kwargs)

    return convert


converters = OrderedDict(
    (('psr', dict(parse=str, fmt=':<10s')),
     ('date', dict(fmt=':>10s')),  # inferred from mjd_mid
     ('utc_mid', dict(fmt=':11.2f')),  # inferred from mjd_mid
     ('mjd_mid', dict(parse=int_frac, fmt=':20.11f',
                      convert=change_type(Time, format='mjd'))),
     ('dm', dict(parse=float, fmt='.value:21.6f',
                 convert=change_type(DispersionMeasure))),
     ('vbyc_earth', dict(parse=float, fmt='.value:7.3f',
                         convert=change_type(u.Quantity, unit=1e-4))),
     ('lgrms', dict(parse=float, fmt=':7.3f')),
     ('rphase', dict(parse=int_frac, fmt=':20.6f',
                     convert=change_type(Phase))),
     ('f0', dict(parse=float, fmt='.value:18.12f',
                 convert=change_type(u.Quantity, unit=u.cycle/u.s))),
     ('obs', dict(parse=str, fmt=':>5s')),
     ('span', dict(parse=int, fmt='.value:5.0f',
                   convert=change_type(u.Quantity, unit=u.minute))),
     ('ncoeff', dict(parse=int, fmt=':5d')),
     ('freq', dict(parse=float, fmt='.value:10.3f',
                   convert=change_type(u.Quantity, unit=u.MHz))),
     ('binphase', dict(parse=float, fmt='.value:7.4f',
                       convert=change_type(Angle, unit=u.cy))),
     ('forb', dict(parse=float, fmt='.value:9.4f',
                   convert=change_type(u.Quantity, unit=u.cy/u.day)))))


def polyco2table(name):
    """Parse a tempo1,2 polyco file and convert it to a QTable.

    Parameters
    ----------
    name : string
        file name holding polyco data

    Returns
    -------
    t : `~astropy.table.QTable`
        Each entry in the polyco file corresponds to one row; columns
        hold psr, date, utc_mid, mjd_mid, dm, vbyc_earth, lgrms,
        rphase, f0, obs, span, ncoeff, freq, binphase & forb (optional),
        and coeff[ncoeff].
    """
    d2e = ''.maketrans('Dd', 'ee')

    t = []
    with open(name, 'r') as polyco:
        line = polyco.readline()
        while line != '':
            # Parse Header.
            pieces = line.split() + polyco.readline().split()
            d = OrderedDict(((key, converter['parse'](piece))
                             for (key, converter), piece in
                             zip(converters.items(), pieces)
                             if 'parse' in converter))
            # Parse coefficients.
            d['coeff'] = []
            while len(d['coeff']) < d['ncoeff']:
                d['coeff'] += polyco.readline().split()

            d['coeff'] = np.array([float(c.translate(d2e))
                                   for c in d['coeff']])

            t.append(d)
            line = polyco.readline()

    t = QTable(t)
    for key in t.colnames:
        try:
            t[key] = converters[key]['convert'](t[key])
        except KeyError:
            pass

    return t


def fortran_fmt(x, base_fmt='23.16e'):
    s = format(x, base_fmt)
    pre, _, post = s.partition('.')
    mant, _, exp = post.partition('e')
    return pre[:-1] + '0.' + pre[-1] + mant + 'D{:+03d}'.format(int(exp) + 1)
