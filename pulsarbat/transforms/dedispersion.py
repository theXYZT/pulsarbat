"""Dedispersion routines."""

import math
import numpy as np
import astropy.units as u
import pulsarbat as pb
import dask
import dask.array as da


__all__ = [
    "DispersionMeasure",
    "DM",
    "coherent_dedispersion",
    "incoherent_dedispersion",
]


def _transfer_function(coeff, N, dt, center_freq, ref_freq):
    f = center_freq.to(u.Hz) + np.fft.fftfreq(N, dt).to(u.Hz)
    phase = coeff * f * u.cycle * (1 / ref_freq - 1 / f) ** 2
    tf = np.exp(-1j * phase.to_value(u.rad))
    return tf.astype(np.complex64)


class DispersionMeasure(u.SpecificTypeQuantity):
    """Dispersion Measure class (with default units of pc / cm^3)."""

    _equivalent_unit = _default_unit = u.pc / u.cm ** 3
    dispersion_constant = u.s * u.MHz ** 2 * u.cm ** 3 / u.pc / 2.41e-4

    def time_delay(self, f, ref_freq):
        """Time delay of frequencies relative to reference frequency."""
        coeff = self.dispersion_constant * self
        delay = coeff * (1 / f ** 2 - 1 / ref_freq ** 2)
        return delay.to(u.s)

    def sample_delay(self, f, ref_freq, sample_rate):
        """Sample delay of frequencies relative to reference frequency."""
        samples = self.time_delay(f, ref_freq) * sample_rate
        samples = samples.to_value(u.one)
        return samples

    def chirp_function(self, N, dt, center_freq, ref_freq, use_dask=False):
        """Chirp function for coherent dedispersion."""
        coeff = self.dispersion_constant * self
        tf_args = (coeff, N, dt, center_freq, ref_freq)

        if use_dask:
            delayed_tf = dask.delayed(_transfer_function, pure=True)
            chirp = da.from_delayed(
                delayed_tf(*tf_args), dtype=np.complex64, shape=(N,)
            )
        else:
            chirp = _transfer_function(*tf_args)

        return chirp

    def chirp_from_signal(self, z, /, *, ref_freq=None):
        """Returns chirp function to dedisperse given baseband signal."""
        if not isinstance(z, pb.BasebandSignal):
            raise TypeError("Signal must be a BasebandSignal object.")

        ix = tuple(slice(None) if i < 2 else None for i in range(z.ndim))
        N, dt = len(z), z.dt

        if ref_freq is None:
            ref_freq = z.center_freq

        chirps = [
            self.chirp_function(N, dt, f, ref_freq, isinstance(z.data, da.Array))
            for f in z.channel_freqs
        ]

        return np.stack(chirps, axis=1)[ix]


DM = DispersionMeasure


def coherent_dedispersion(z, DM, /, *, ref_freq=None, chirp=None):
    """Coherently dedisperses a baseband signal.

    The given signal will be coherently dedispersed by a given dispersion
    measure (DM). If a reference frequency (``ref_freq``) is not given, the
    center frequency of the signal will be used as reference.

    Optionally, a pre-computed chirp function (``chirp``) can be provided
    as an array. If a chirp is provided, it will not be checked against
    the given DM and reference frequency for correctness.

    The output signal will be cropped on both ends to avoid wrap-around
    artifacts caused by dedispersion. This depends on where the
    reference frequency (``ref_freq``) is compared to the band of the
    signal.

    Parameters
    ----------
    z : BasebandSignal
        The signal to be transformed.
    DM : DispersionMeasure
        Dispersion measure by which to dedisperse ``z``.
    ref_freq : Quantity, optional
        Reference frequency for dedispersion. If None (default), uses
        the center frequency from signal.
    chirp : array-like, optional
        A pre-computed chirp function. Must be a 2-D array with shape
        ``z.shape[:2]``.

    Returns
    -------
    out : BasebandSignal
        The dedispersed signal.
    """
    if not isinstance(z, pb.BasebandSignal):
        raise TypeError("Signal must be a BasebandSignal object.")

    if ref_freq is None:
        ref_freq = z.center_freq

    if chirp is None:
        chirp = DM.chirp_from_signal(z, ref_freq=ref_freq)

    x = pb.fft.ifft(pb.fft.fft(z.data, axis=0) * chirp, axis=0)

    delay_top = DM.sample_delay(z.max_freq, ref_freq, z.sample_rate)
    delay_bot = DM.sample_delay(z.min_freq, ref_freq, z.sample_rate)

    start = math.ceil(-min(0, delay_top, delay_bot))
    stop = x.shape[0] - math.ceil(+max(0, delay_top, delay_bot))

    return type(z).like(z, x)[start:stop]


def incoherent_dedispersion(z, DM, /, *, ref_freq=None):
    """Incoherently dedisperses a signal by a given dispersion measure.

    The output signal will be cropped on both ends to avoid wrap-around
    artifacts caused by dedispersion. This depends on where the
    reference frequency (``ref_freq``) compared to the band of the signal.

    Parameters
    ----------
    z : RadioSignal
        The signal to be transformed.
    DM : DispersionMeasure
        Dispersion measure by which to dedisperse ``z``.
    ref_freq : Quantity, optional
        Reference frequency for dedispersion. If None (default), uses
        the center frequency from signal.

    Returns
    -------
    out : RadioSignal
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

    x = np.stack([z.data[j : j + N, i] for i, j in enumerate(delays)], axis=1)

    new_start = z.start_time
    if crop_before and z.start_time is not None:
        new_start += crop_before * z.dt

    return type(z).like(z, x, start_time=new_start)
