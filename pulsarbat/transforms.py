"""Signal-to-signal transforms."""

import os
import functools
import numpy as np
import astropy.units as u
from .core import (Signal, BasebandSignal, DispersionMeasure,
                   verify_scalar_quantity, InvalidSignalError)

try:
    import pyfftw
    pyfftw.config.NUM_THREADS = int(os.environ.get('OMP_NUM_THREADS', 2))
    fftpack = pyfftw.interfaces.numpy_fft
except ImportError:
    fftpack = np.fft

__all__ = ['dedisperse', 'channelize', 'linear_to_circular',
           'circular_to_linear']


def transform(func):
    """Decorator for all transforms."""

    @functools.wraps(func)
    def wrapper(z, *args, **kwargs):
        if not isinstance(z, BasebandSignal):
            raise TypeError('Input signal must be a BasebandSignal object.')
        return func(z, *args, **kwargs)
    return wrapper


@transform
def dedisperse(z: BasebandSignal, DM: DispersionMeasure, ref_freq: u.Quantity):
    """Dedisperses a signal by a given dispersion measure.

    The output signal will be cropped on both ends to avoid wrap-around
    artifacts caused by dedispersion. This depends on how the reference
    frequency (`ref_freq`) compares to the band of the signal.

    Parameters
    ----------
    z : `~pulsarbat.BasebandSignal`
        The signal to be transformed.
    DM : `~pulsarbat.DispersionMeasure`
        Dispersion measure by which to dedisperse `z`.
    ref_freq : `~astropy.units.Quantity`
        Reference frequency to dedisperse to.

    Returns
    -------
    out : `~pulsarbat.BasebandSignal`
        The dedispersed signal.
    """
    if not isinstance(DM, DispersionMeasure):
        raise TypeError('DM must be a DispersionMeasure object.')
    verify_scalar_quantity(ref_freq, u.Hz)

    N = len(z)
    f = z.channel_centers[None] + np.fft.fftfreq(N, z.dt)[:, None]
    phase_factor = np.asfortranarray(DM.phase_factor(f, ref_freq))

    x = fftpack.fft(np.array(z), axis=0)
    x = fftpack.ifft((x.T * phase_factor.T).T, axis=0)

    crop_before = -min(0, DM.sample_delay(z.max_freq, ref_freq, z.sample_rate))
    crop_after = max(0, DM.sample_delay(z.min_freq, ref_freq, z.sample_rate))

    x = x[crop_before:-crop_after]
    time_cropped = crop_before * z.dt

    return z.copy(z=x, start_time=z.start_time + time_cropped)


@transform
def channelize(z: BasebandSignal, factor: int):
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

    x = np.fft.fftshift(fftpack.fft(x, axis=2), axes=(2,))

    new_shape = (len(x), -1) + x.shape[3:]
    x = x.reshape(new_shape)

    return z.copy(z=x, sample_rate=z.sample_rate/factor)


@transform
def convolve(z: BasebandSignal, h: Signal):
    """Convolves a filter h with a signal z."""

    if not isinstance(h, Signal):
        raise TypeError('Filter must be a Signal object.')

    if h.sample_rate != z.sample_rate:
        err = 'Input signal and filter have different sample rates!'
        raise InvalidSignalError(err)

    if h.ndim > z.ndim:
        raise InvalidSignalError('Filter has more dimensions than signal!')
    else:
        h.expand_dims(z.ndim)

    raise NotImplementedError(':)')


@transform
def linear_to_circular(z: BasebandSignal, axis: int):
    """Converts a baseband signal from linear basis to circular basis.

    The polarization components are expected to be located along the
    axis provided (`axis`). If `z.shape[axis] != 2`, an exception is
    raised since there must be exactly two polarization components.

    It is assumed that the linear components are ordered as (X, Y) and
    circular components are ordered as (R, L).

    Parameters
    ----------
    z : `~pulsarbat.BasebandSignal`
        The signal to be converted. Must be in linear basis for
        meaningful results.
    axis : int
        Polarization axis.

    Returns
    -------
    out : `~pulsarbat.BasebandSignal`
        The converted signal.
    """
    if axis in [0, 1]:
        raise ValueError('Invalid polarization axis!')

    if not z.shape[axis] == 2:
        err = 'Polarization axis does not have 2 components!'
        raise InvalidSignalError(err)

    X = np.expand_dims(np.take(z, 0, axis), axis)
    Y = np.expand_dims(np.take(z, 1, axis), axis)

    circular = np.append(X - 1j*Y, X + 1j*Y, axis=axis) / np.sqrt(2)
    return z.copy(z=circular)


@transform
def circular_to_linear(z: BasebandSignal, axis: int):
    """Converts a baseband signal from circular basis to linear basis.

    The polarization components are expected to be located along the
    axis provided (`axis`). If `z.shape[axis] != 2`, an exception is
    raised since there must be exactly two polarization components.

    It is assumed that the linear components are ordered as (X, Y) and
    circular components are ordered as (R, L).

    Parameters
    ----------
    z : `~pulsarbat.BasebandSignal`
        The signal to be converted. Must be in circular basis for
        meaningful results.
    axis : int
        Polarization axis.

    Returns
    -------
    out : `~pulsarbat.BasebandSignal`
        The converted signal.
    """
    if axis in [0, 1]:
        raise ValueError('Invalid polarization axis!')

    if not z.shape[axis] == 2:
        err = 'Polarization axis does not have 2 components!'
        raise InvalidSignalError(err)

    R = np.expand_dims(np.take(z, 0, axis), axis)
    L = np.expand_dims(np.take(z, 1, axis), axis)

    linear = np.append(R + L, 1j * (R - L), axis=axis) / np.sqrt(2)
    return z.copy(z=linear)
