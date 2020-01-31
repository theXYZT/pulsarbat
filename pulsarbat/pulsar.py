"""Pulsar processing."""

import numpy as np
from .core import (BasebandSignal, IntensitySignal, verify_scalar_quantity)
from .predictor import Polyco
from astropy.time import Time
import astropy.units as u

__all__ = ['fold']


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
    pulse_profile_shape = (ngate, ) + z.shape[1:]
    pulse_profile = np.zeros(pulse_profile_shape, dtype=np.float64)

    raise NotImplementedError('whoops')
