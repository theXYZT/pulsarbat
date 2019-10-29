"""Signal-to-signal transforms."""

import numpy as np
import astropy.units as u
import os
from .core import BasebandSignal, DispersionMeasure

try:
    import pyfftw
    pyfftw.config.NUM_THREADS = int(os.environ.get('OMP_NUM_THREADS', 2))
    fftpack = pyfftw.interfaces.numpy_fft
except ImportError:
    fftpack = np.fft

__all__ = ['dedisperse', 'channelize']


def dedisperse(z: BasebandSignal, DM: DispersionMeasure, ref_freq: u.Quantity):
    """Dedisperses a signal by a given dispersion measure.

    The output signal will be cropped on both ends to avoid wrap-around
    artifacts caused by dedispersion.

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
    if not isinstance(z, BasebandSignal):
        raise TypeError('Input signal must be a BasebandSignal object.')

    if not isinstance(DM, DispersionMeasure):
        raise TypeError('DM must be a DispersionMeasure object.')

    N = len(z)
    f = z.channel_centers[None] + np.fft.fftfreq(N, z.dt)[:, None]
    phase_factor = DM.phase_factor(f, ref_freq)

    x = fftpack.fft(np.array(z), axis=0)
    x = fftpack.ifft((x.T * phase_factor.T).T, axis=0)

    crop_before = -min(0, DM.sample_delay(z.max_freq, ref_freq, z.sample_rate))
    crop_after = max(0, DM.sample_delay(z.min_freq, ref_freq, z.sample_rate))

    x = x[crop_before:N-crop_after]
    time_cropped = crop_before * z.dt

    return z.copy(z=x, start_time=z.start_time + time_cropped)


def channelize(z: BasebandSignal, factor: int):
    """Channelizes a signal by a given factor.

    For example, if `factor` is 8, and the input signal has 4 channels,
    the output signal will have 32 channels. A factor less than 8 is not
    recommended due to artifacts caused from extremely small Fourier
    Transforms.

    The output signal will also be cropped at the end to accomodate
    channelizing.

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
    if not isinstance(z, BasebandSignal):
        raise TypeError('Input signal must be a BasebandSignal object.')

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


def linear_to_circular(z: BasebandSignal, axis: int):
    """Converts a baseband signal from linear basis to circular basis.

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
    assert z.shape[axis] == 2

    X = np.take(z, 0, axis)
    Y = np.take(z, 1, axis)

    ind = [slice(None)] * len(z.shape)
    ind[axis] = None

    R = ((X + 1j*Y)/np.sqrt(2))[tuple(ind)]
    L = ((X - 1j*Y)/np.sqrt(2))[tuple(ind)]

    circular = np.append(R, L, axis=axis)
    return z.copy(z=circular)


def circular_to_linear(z: BasebandSignal, axis: int):
    """Converts a baseband signal from circular basis to linear basis.

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
    assert z.shape[axis] == 2

    R = np.take(z, 0, axis)
    L = np.take(z, 1, axis)

    ind = [slice(None)] * len(z.shape)
    ind[axis] = None

    X = ((L + R)/np.sqrt(2))[tuple(ind)]
    Y = (1j*(L - R)/np.sqrt(2))[tuple(ind)]

    linear = np.append(X, Y, axis=axis)
    return z.copy(z=linear)
