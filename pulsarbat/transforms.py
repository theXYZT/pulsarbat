"""Signal-to-signal transforms."""

import numpy as np
import astropy.units as u
from .core import BasebandSignal, DispersionMeasure
from . import utils

try:
    import pyfftw
    fftpack = pyfftw.interfaces.numpy_fft
except ImportError:
    fftpack = np.fft

__all__ = []


def dedisperse(z: BasebandSignal, DM: DispersionMeasure):
    """Dedisperses a signal by a given dispersion measure.

    Parameters
    ----------
    z : `~pulsarbat.BasebandSignal`
        The signal to be transformed.
    DM : `~astropy.units.Quantity`
        Dispersion measure by which to dedisperse `z`. Must be in units
        equivalent to `1/cm**2`

    Returns
    -------
    out : `~pulsarbat.BasebandSignal`
        The dedispersed signal.
    """
    if not isinstance(DM, u.Quantity):
        raise TypeError('DM must be an astropy Quantity.')
    if not DM.unit.is_equivalent(1/u.cm**2):
        raise u.UnitTypeError('DM must have units equivalent to 1/cm**2.')

    x = np.array(z)

    phase_factor = utils.Dispersion.phase_factor(len)



    x = fftpack.fft(x, axis=0)
    x = fftpack.ifft(x * phase_factor, axis=0)
    return utils.baseband_signal_like(z, x)
