"""Module for all things dispersion."""

import numpy as np
import astropy.units as u

__all__ = ['DispersionMeasure']


def taper_function(freqs, bandwidth):
    x = (freqs / bandwidth).to_value(u.one)
    taper = 1 + (x / 0.48)**80
    return taper


class DispersionMeasure(u.SpecificTypeQuantity):
    _equivalent_unit = _default_unit = u.pc / u.cm**3
    dispersion_constant = u.s * u.MHz**2 * u.cm**3 / u.pc / 2.41E-4

    def time_delay(self, f, ref_freq):
        """Time delay of frequencies relative to reference frequency."""
        coeff = self.dispersion_constant * self
        delay = coeff * (1 / f**2 - 1 / ref_freq**2)
        return delay.to(u.s)

    def sample_delay(self, f, ref_freq, sample_rate):
        """Sample delay of frequencies relative to reference frequency."""
        samples = self.time_delay(f, ref_freq) * sample_rate
        samples = samples.to_value(u.one)
        return int(np.copysign(np.ceil(abs(samples)), samples))

    def inverse_transfer_function(self, f, ref_freq):
        """Returns the inverse transfer function for coherent dedispersion."""
        coeff = self.dispersion_constant * self
        phase = coeff * f * u.cycle * (1 / ref_freq - 1 / f)**2
        inv_transfer = np.exp(-1j * phase.to_value(u.rad))
        return inv_transfer.astype(np.complex64)


def chirp_function(self, DM, f, ref_freq, taper=False):
    inv_transfer = DM.inverse_transfer_function(f, ref_freq)

