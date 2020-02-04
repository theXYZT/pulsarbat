"""Pulsar processing."""

import numpy as np
from .core import (IntensitySignal, verify_scalar_quantity)
from .predictor import Polyco
from astropy.time import Time
import astropy.units as u

__all__ = ['fold']


class PulseProfile:
    """Class for pulse profiles."""
    pass


def get_pulse_phases(start_time: Time, num_samples: int,
                     sample_rate: u.Quantity, polyco: Polyco):
    """Returns pulse phases for given parameters."""
    if not isinstance(start_time, Time) or not start_time.isscalar:
        raise ValueError('Invalid start time provided.')

    num_samples = int(num_samples)
    verify_scalar_quantity(sample_rate, u.Hz)

    p = polyco.phasepol(start_time,
                        rphase='fraction',
                        t0=start_time,
                        time_unit=u.s,
                        convert=True)
    ph = p(np.arange(num_samples) * (1 / sample_rate).to(u.s).value)
    ph -= np.floor(ph[0])
    return ph


def fold(z: IntensitySignal, polyco: Polyco, ngate: int):
    """Fold a pulse profile."""
    def bincount1d(x, ph, ngate):
        return np.bincount(ph, x, minlength=ngate)

    ph = get_pulse_phases(z.start_time, len(z), z.sample_rate, polyco)
    ph = (np.remainder(ph, 1) * ngate).astype(np.int32)

    counts = np.bincount(ph, minlength=ngate)
    profile = np.apply_along_axis(bincount1d, 0, np.array(z), ph, ngate)

    raise NotImplementedError('whoops')
