"""Signal reductions."""

import numpy as np
from .core import BasebandSignal, IntensitySignal

__all__ = ['to_stokes', 'to_intensity']


def to_intensity(z: BasebandSignal):
    """Converts a baseband signal to intensities via complex square."""
    data = np.array(z)
    intensity = data.real**2 + data.imag**2

    return IntensitySignal(z=intensity,
                           sample_rate=z.sample_rate,
                           start_time=z.start_time,
                           center_freq=z.center_freq,
                           bandwidth=z.bandwidth)


def to_stokes(z: BasebandSignal, pol_type: str, axis: int):
    """Converts a baseband signal to IQUV Stokes representation.

    `pol_type` is the polarization type of the input signal (either
    'linear' or 'circular'), and `axis` refers to the polarization axis.

    Parameters
    ----------
    z : `~pulsarbat.BasebandSignal`
        The signal to be reduced.
    pol_type : {'linear', 'circular'}
        Polarization type for the input signal.
    axis : int
        Polarization axis on input signal.

    Returns
    -------
    out : `~pulsarbat.IntensitySignal`
        Signal in Stokes IQUV representation.
    """

    if pol_type == 'linear':
        return linear_to_stokes(z, axis)
    elif pol_type == 'circular':
        return circular_to_stokes(z, axis)
    else:
        message = f"{pol_type} is not a supported pol_type."
        raise ValueError(message)


def linear_to_stokes(z: BasebandSignal, axis: int):
    """Converts a signal in linear basis to Stokes IQUV.

    `axis` refers to the polarization axis. The first index on `axis` is
    assumed to be the X component, and the second index is assumed to be
    the Y component.

    Parameters
    ----------
    z : `~pulsarbat.BasebandSignal`
        The signal to be reduced.
    axis : int
        Polarization axis on input signal.

    Returns
    -------
    out : `~pulsarbat.IntensitySignal`
        Signal in Stokes IQUV representation.
    """
    def get_id(i):
        ndim = len(z.shape)
        ind = [slice(None)] * ndim
        ind[axis] = i
        return tuple(ind)

    assert z.shape[axis] == 2
    stokes_shape = z.shape[:axis] + (4, ) + z.shape[axis + 1:]
    stokes = np.empty(stokes_shape, dtype=np.float32, order='F')

    X = np.take(z, 0, axis)
    Y = np.take(z, 1, axis)

    stokes[get_id(0)] = X.real**2 + X.imag**2 + Y.real**2 + Y.imag**2
    stokes[get_id(1)] = X.real**2 + X.imag**2 - Y.real**2 - Y.imag**2
    stokes[get_id(2)] = +2 * (X * Y.conj()).real
    stokes[get_id(3)] = -2 * (X * Y.conj()).imag

    return IntensitySignal(z=stokes,
                           sample_rate=z.sample_rate,
                           start_time=z.start_time,
                           center_freq=z.center_freq,
                           bandwidth=z.bandwidth)


def circular_to_stokes(z: BasebandSignal, axis: int):
    """Converts a signal in circular basis to Stokes IQUV.

    `axis` refers to the polarization axis. The first index on `axis` is
    assumed to be the R component, and the second index is assumed to be
    the L component.

    Parameters
    ----------
    z : `~pulsarbat.BasebandSignal`
        The signal to be reduced.
    axis : int
        Polarization axis on input signal.

    Returns
    -------
    out : `~pulsarbat.IntensitySignal`
        Signal in Stokes IQUV representation.
    """
    def get_id(i):
        ndim = len(z.shape)
        ind = [slice(None)] * ndim
        ind[axis] = i
        return tuple(ind)

    assert z.shape[axis] == 2
    stokes_shape = z.shape[:axis] + (4, ) + z.shape[axis + 1:]
    stokes = np.empty(stokes_shape, dtype=np.float32, order='F')

    R = np.take(z, 0, axis)
    L = np.take(z, 1, axis)

    stokes[get_id(0)] = R.real**2 + R.imag**2 + L.real**2 + L.imag**2
    stokes[get_id(1)] = +2 * (R * L.conj()).real
    stokes[get_id(2)] = -2 * (R * L.conj()).imag
    stokes[get_id(3)] = R.real**2 + R.imag**2 - L.real**2 - L.imag**2

    return IntensitySignal(z=stokes,
                           sample_rate=z.sample_rate,
                           start_time=z.start_time,
                           center_freq=z.center_freq,
                           bandwidth=z.bandwidth)
