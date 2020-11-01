"""Signal-to-signal transforms."""

import numpy as np
import astropy.units as u
from .core import RadioSignal, BasebandSignal
from .utils import fftpack

__all__ = [
    'DispersionMeasure', 'coherent_dedispersion', 'incoherent_dedispersion'
]


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
        return samples

    def phase_delay(self, f, ref_freq):
        coeff = self.dispersion_constant * self
        phase = coeff * f * u.cycle * (1 / ref_freq - 1 / f)**2
        return phase.to_value(u.rad)

    def transfer_function(self, f, ref_freq):
        """Returns the transfer function for dedispersion."""
        transfer = np.exp(1j * self.phase_delay(f, ref_freq))
        return transfer.astype(np.complex64)


def coherent_dedispersion(z, DM, /, *, ref_freq, chirp=None):
    """Coherently dedisperses a baseband signal by a given dispersion measure.

    The output signal will be cropped on both ends to avoid wrap-around
    artifacts caused by dedispersion. This depends on how the reference
    frequency (`ref_freq`) compares to the band of the signal.

    The chirp function can be provided via the `chirp` argument which will be
    used instead of computing one from scratch. This can be useful in cases
    where the chirp function needs to be cached for efficiency.

    Parameters
    ----------
    z : `~pulsarbat.BasebandSignal`
        The signal to be transformed.
    DM : `~pulsarbat.DispersionMeasure`
        Dispersion measure by which to dedisperse `z`.
    ref_freq : `~astropy.units.Quantity`
        Reference frequency to dedisperse to.
    chirp: `~numpy.ndarray`, optional
        The dedispersion chirp function provided to avoid computing a new one.

    Returns
    -------
    out : `~pulsarbat.BasebandSignal`
        The dedispersed signal.
    """
    if not isinstance(DM, DispersionMeasure):
        raise TypeError('DM must be a DispersionMeasure object.')
    verify_scalar_quantity(ref_freq, u.Hz)

    if chirp is None:
        f = z.channel_centers[None] + np.fft.fftfreq(len(z), z.dt)[:, None]
        chirp = DM.transfer_function(f, ref_freq)
    else:
        assert chirp.shape != z.shape[:chirp.ndims]

    x = fftpack.fft(np.asarray(z), axis=0)
    x = fftpack.ifft((x.T / chirp.T).T, axis=0)

    crop_before = -min(0, DM.sample_delay(z.max_freq, ref_freq, z.sample_rate))
    crop_after = max(0, DM.sample_delay(z.min_freq, ref_freq, z.sample_rate))

    x = x[crop_before:-crop_after]
    time_cropped = crop_before * z.dt

    return BasebandSignal.like(z, x, start_time=z.start_time + time_cropped)


def incoherent_dedispersion(z, DM, /, *, ref_freq):
    raise NotImplementedError("Will do it soon, I promise!")
