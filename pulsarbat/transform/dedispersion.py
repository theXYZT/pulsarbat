"""Dedispersion routines."""

import math
import numpy as np
from numpy.core.overrides import set_module
import astropy.units as u
import pulsarbat as pb

__all__ = [
    'DispersionMeasure', 'coherent_dedispersion', 'incoherent_dedispersion'
]


@set_module("pulsarbat")
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


@set_module("pulsarbat")
def coherent_dedispersion(z, DM, /, *, ref_freq=None):
    """Coherently dedisperses a baseband signal by a given dispersion measure.

    The output signal will be cropped on both ends to avoid wrap-around
    artifacts caused by dedispersion. This depends on where the
    reference frequency (`ref_freq`) compared to the band of the signal.

    Parameters
    ----------
    z : `~pulsarbat.BasebandSignal`
        The signal to be transformed.
    DM : `~pulsarbat.DispersionMeasure`
        Dispersion measure by which to dedisperse `z`.
    ref_freq : `~astropy.units.Quantity`, optional
        Reference frequency for dedispersion. If None (default), uses
        the center frequency from signal.

    Returns
    -------
    out : `~pulsarbat.BasebandSignal`
        The dedispersed signal.
    """
    def _transfer_func(DM, center_freq, N, dt, ref_freq):
        f = center_freq + np.fft.fftfreq(N, dt)
        return np.exp(-1j * DM.phase_delay(f, ref_freq)).astype(np.complex64)

    def _delayed_transfer_func(DM, center_freq, N, dt, ref_freq):
        delayed_tf = dask.delayed(_transfer_func, pure=True)
        delayed_chirp = delayed_tf(DM, center_freq, N, dt, ref_freq)
        return da.from_delayed(delayed_chirp, dtype=np.complex64, shape=(N,))

    if not isinstance(z, pb.BasebandSignal):
        raise TypeError("Signal must be a BasebandSignal object.")

    if ref_freq is None:
        ref_freq = z.center_freq

    try:
        import dask
        import dask.array as da
    except ImportError:
        use_dask = False
    else:
        use_dask = isinstance(z.data, da.Array)

    if use_dask:
        transfer_func = _delayed_transfer_func
    else:
        transfer_func = _transfer_func

    ix = tuple(slice(None) if i < 2 else None for i in range(z.ndim))

    chirp = np.stack([transfer_func(DM, f, len(z), z.dt, ref_freq)
                      for f in z.channel_freqs], axis=1)[ix]

    x = pb.fft.ifft(pb.fft.fft(z.data, axis=0) * chirp, axis=0)

    delay_top = DM.sample_delay(z.max_freq, ref_freq, z.sample_rate)
    delay_bot = DM.sample_delay(z.min_freq, ref_freq, z.sample_rate)

    start = math.ceil(-min(0, delay_top, delay_bot))
    stop = x.shape[0] - math.ceil(+max(0, delay_top, delay_bot))

    return type(z).like(z, x)[start:stop]


@set_module("pulsarbat")
def incoherent_dedispersion(z, DM, /, *, ref_freq=None):
    """Incoherently dedisperses a signal by a given dispersion measure.

    The output signal will be cropped on both ends to avoid wrap-around
    artifacts caused by dedispersion. This depends on where the
    reference frequency (`ref_freq`) compared to the band of the signal.

    Parameters
    ----------
    z : `~pulsarbat.RadioSignal`
        The signal to be transformed.
    DM : `~pulsarbat.DispersionMeasure`
        Dispersion measure by which to dedisperse `z`.
    ref_freq : `~astropy.units.Quantity`, optional
        Reference frequency for dedispersion. If None (default), uses
        the center frequency from signal.

    Returns
    -------
    out : `~pulsarbat.RadioSignal`
        The dedispersed signal.
    """
    if not isinstance(z, pb.RadioSignal):
        raise TypeError("Signal must be a RadioSignal object.")

    if ref_freq is None:
        ref_freq = z.center_freq

    delays = DM.sample_delay(z.channel_freqs, ref_freq, z.sample_rate)
    delays = delays.round().astype(np.int64)

    crop_before = -min(0, delays[0], delays[-1])
    delays += crop_before
    N = len(z) - max(delays)

    x = np.stack([z.data[j:j+N, i] for i, j in enumerate(delays)], axis=1)

    if crop_before and z.start_time is not None:
        new_start = z.start_time + crop_before * z.dt
    else:
        new_start = z.start_time

    return type(z).like(z, x, start_time=new_start)
