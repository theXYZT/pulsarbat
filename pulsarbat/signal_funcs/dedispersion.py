"""Dedispersion routines."""

import math
import numpy as np
import astropy.units as u
from ..core import RadioSignal, BasebandSignal

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
        The dedispersion chirp function can be provided to avoid
        computing a new one. This is useful when the same dedispersion
        operation is being done on many blocks of the same size

    Returns
    -------
    out : `~pulsarbat.BasebandSignal`
        The dedispersed signal.
    """
    if not isinstance(z, BasebandSignal):
        raise TypeError("Signal must be a BasebandSignal object.")

    if chirp is None:
        f = z.channel_freqs[None] + np.fft.fftfreq(len(z), z.dt)[:, None]
        chirp = DM.transfer_function(f, ref_freq)

    x = np.fft.fft(z.data, axis=0)
    x = np.fft.ifft((x.T / chirp.T).T, axis=0)

    y = type(z).like(z, x)

    delay_top = DM.sample_delay(z.max_freq, ref_freq, z.sample_rate)
    delay_bot = DM.sample_delay(z.min_freq, ref_freq, z.sample_rate)

    start = math.ceil(-min(0, delay_top, delay_bot))
    stop = len(y) - math.ceil(+max(0, delay_top, delay_bot))

    return type(z).like(z, x)[start:stop]


def incoherent_dedispersion(z, DM, /, *, ref_freq):
    """Incoherently dedisperses a signal by a given dispersion measure.

    The output signal will be cropped on both ends to avoid wrap-around
    artifacts caused by dedispersion. This depends on how the reference
    frequency (`ref_freq`) compares to the band of the signal.

    Parameters
    ----------
    z : `~pulsarbat.RadioSignal`
        The signal to be transformed.
    DM : `~pulsarbat.DispersionMeasure`
        Dispersion measure by which to dedisperse `z`.
    ref_freq : `~astropy.units.Quantity`
        Reference frequency to dedisperse to.

    Returns
    -------
    out : `~pulsarbat.RadioSignal`
        The dedispersed signal.
    """
    if not isinstance(z, RadioSignal):
        raise TypeError("Signal must be a RadioSignal object.")

    delays = DM.sample_delay(z.channel_freqs, ref_freq, z.sample_rate)
    delays = delays.round().astype(int)

    crop_before = -min(0, delays[0], delays[-1])
    delays += crop_before
    N = len(z) - max(delays)

    x = np.stack([z.data[j:j+N, i] for i, j in enumerate(delays)], axis=1)

    if crop_before and z.start_time is not None:
        new_start = z.start_time + crop_before * z.dt
    else:
        new_start = z.start_time

    return type(z).like(z, x, start_time=new_start)
