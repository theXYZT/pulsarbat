"""Signal-to-signal transforms."""

import functools
import numpy as np
import astropy.units as u
from .core import Signal, BasebandSignal, DispersionMeasure

__all__ = ['dedisperse', 'channelize', ]


def transform(func):
    """Decorator for all transforms."""
    @functools.wraps(func)
    def wrapper(z, *args, **kwargs):
        if not isinstance(z, BasebandSignal):
            raise TypeError('Input signal must be a BasebandSignal object.')
        return func(z, *args, **kwargs)

    return wrapper


@transform
def dedisperse(z: BasebandSignal, DM: DispersionMeasure, ref_freq: u.Quantity,
               chirp: np.ndarray = None) -> BasebandSignal:
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


@transform
def channelize(z: BasebandSignal, factor: int) -> BasebandSignal:
    """Channelizes a signal by a given factor.

    For example, if `factor` is 8, and the input signal has 4 channels,
    the output signal will have 32 channels. A factor less than 8 is not
    recommended due to artifacts caused from extremely small Fourier
    Transforms.

    The output signal will also be cropped at the end if `len(z)` is not
    divisible by `factor`.

    Parameters
    ----------
    z : `~pulsarbat.BasebandSignal`
        The signal to be transformed.
    factor : int
        Channelization factor.

    Returns
    -------
    out : `~pulsarbat.BasebandSignal`
        The channelized signal.
    """
    if not isinstance(factor, int):
        raise TypeError("factor must be an integer.")

    N = factor * (len(z) // factor)
    x = np.array(z)[:N]

    new_shape = (-1, factor) + x.shape[1:]
    x = np.swapaxes(x.reshape(new_shape), 1, 2)

    x = np.fft.fftshift(fftpack.fft(x, axis=2), axes=(2, ))

    new_shape = (len(x), -1) + x.shape[3:]
    x = x.reshape(new_shape)

    return BasebandSignal.like(z, x, sample_rate=z.sample_rate / factor)


@transform
def convolve(z: BasebandSignal, h: Signal):
    """Convolves a filter h with a signal z."""

    if not isinstance(h, Signal):
        raise TypeError('Filter must be a Signal object.')

    if h.sample_rate != z.sample_rate:
        err = 'Input signal and filter have different sample rates!'
        raise ValueError(err)

    raise NotImplementedError("I promise I'll do this later :)")
