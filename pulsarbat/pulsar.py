"""Pulsar processing."""

import numpy as np
from .core import (IntensitySignal, verify_scalar_quantity)
from .predictor import Polyco
from astropy.time import Time
import astropy.units as u

__all__ = ['PulseProfile', 'get_pulse_phases', 'fold']


class PulseProfile(IntensitySignal):
    """Class for pulse profiles.

    Requires that the last axis of `data` refer to the pulse phase.
    Thus, `data` must have a minimum of 3 dimensions with a shape of
    `(nsamples, nchan, ..., ngate)` where `ngate` refers to the number
    of pulse phase bins. In the case of pulse profiles, the samples along the time axis are
    interpreted as sub-integrations.

    See the documentation for `~pulsarbat.RadioSignal` for other
    specifications.

    `counts` is the array of sample counts that make up each pulse phase
    bin and must have a shape of `(nsamples, ngate)`. These counts are
    saved separately in the object to ensure accurate normalization of
    pulse profiles when required.

    Parameters
    ----------
    data : `~numpy.ndarray`
        The pulse profile being stored as an array.
    sample_rate : `~astropy.units.Quantity`
        The number of samples per second. Must be in units of frequency.
        For pulse profiles, this is the rate of sub-integrations.
    start_time : `~astropy.time.Time`
        The start time of the signal (that is, the time at the first
        sample of the signal).
    center_freq : `~astropy.units.Quantity`
        The observing frequency at the center of the signal's band. Must
        be in units of frequency.
    bandwidth : `~astropy.units.Quantity`
        The total bandwidth of the signal. The channel bandwidth is this
        total bandwidth divided by the number of channels. Must be in
        units of frequency.
    counts: `~numpy.ndarray`
        The sample counts in each pulse phase bin of each sub-integration.
    """
    _min_ndim = 3

    def __init__(self, data: np.ndarray, sample_rate: u.Quantity,
                 start_time: Time, center_freq: u.Quantity,
                 bandwidth: u.Quantity, counts: np.ndarray):
        super().__init__(data, sample_rate, start_time, center_freq, bandwidth)

        if counts.shape == (len(self), self.ngate):
            self._counts = counts
        else:
            raise ValueError("Incorrect counts for pulse profile.")

    @property
    def ngate(self):
        """Number of pulse phase bins or gates."""
        return self.shape[-1]

    @property
    def counts(self):
        """Sample counts of the pulse profile."""
        return np.array(self._counts)

    @property
    def phase(self):
        ph = np.linspace(0, 1, self.ngate, endpoint=False)
        return ph + 0.5/self.ngate

    def pulse_profile(self):
        """Returns a normalized pulse profile averaged over time."""
        profile = self.data.sum(0)
        counts = self.counts.sum(0)
        return profile / counts

    def roll(self, ngate=None):
        """Rolls pulse profile by given number of gates."""
        self._counts = np.roll(self._counts, ngate, axis=-1)
        self._data = np.roll(self._data, ngate, axis=-1)

    def append(self, other):
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
    ph = p(np.arange(num_samples) * (1 / sample_rate).to_value(u.s))
    ph -= np.floor(ph[0])
    return ph


def fold(z: IntensitySignal, polyco: Polyco, ngate: int):
    """Fold a pulse profile."""
    def bincount1d(x, ph, ngate):
        return np.bincount(ph, x, minlength=ngate)

    ph = get_pulse_phases(z.start_time, len(z), z.sample_rate, polyco)
    ph = (np.remainder(ph, 1) * ngate).astype(np.int32)

    counts = np.bincount(ph, minlength=ngate)[np.newaxis]
    profile = np.apply_along_axis(bincount1d, 0, np.array(z), ph, ngate)
    profile = np.moveaxis(profile[np.newaxis], 1, -1)

    return PulseProfile.like(z, profile, counts=counts,
                             sample_rate=1 / z.time_length)
