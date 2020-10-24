"""Collection of handy utilities."""

import os
import numpy as np
import astropy.units as u
from scipy.fft import next_fast_len

try:
    import pyfftw
    pyfftw.config.NUM_THREADS = int(os.environ.get('OMP_NUM_THREADS', 2))
    fftpack = pyfftw.interfaces.numpy_fft
except ImportError:
    fftpack = np.fft


__all__ = [
    'fftpack',
    'next_fast_len',
    'verify_scalar_quantity',
    'real_to_complex',
    'complex_noise',
]


def verify_scalar_quantity(a, unit):
    """Verify of given obj is a scale astropy Quantity."""

    if not isinstance(a, u.Quantity):
        raise TypeError(f'Expected astropy Quantity, got {type(a)}')

    if not a.isscalar:
        raise ValueError('Expected a scalar quantity.')

    if not a.unit.is_equivalent(unit):
        expected = f'Expected units of {unit.physical_type}'
        raise u.UnitTypeError(f'{expected}, got units of {a.unit}')

    return True


def complex_noise(shape, power):
    """Generates complex gaussian noise with given shape and power.

    Parameters
    ----------
    shape : int or tuple of ints
        Shape of returned array.
    power : float
        Noise power.

    Returns
    -------
    out : `~numpy.ndarray`
        Complex noise with given power.
    """
    rng = np.random.default_rng()
    re = rng.normal(0, 1 / np.sqrt(2), shape)
    im = rng.normal(0, 1 / np.sqrt(2), shape)
    return (re + 1j * im) * np.sqrt(power)


def real_to_complex(z, axis=0):
    """Convert a real baseband signal to a complex baseband signal.

    This function computes the analytic representation of the input
    signal via a Hilbert transform, throwing away negative frequency
    components. Then, the signal is shifted in frequency domain by -B/2
    where B is the bandwidth of the signal. Finally, the signal is
    decimated by a factor of 2, which results in the complex baseband
    representation of the input signal [1]_.

    If the input signal is complex-valued, only the real component is
    used.

    Parameters
    ----------
    z : `~numpy.ndarray`
        Input signal.
    axis : int, optional
        Axis over which to convert the signal. This will be the axis
        that represents time. Default is 0.

    Returns
    -------
    out : np.ndarray
        The complex baseband representation of the input signal.

    References
    ----------
    .. [1] https://dsp.stackexchange.com/q/43278/17721
    """
    if np.iscomplexobj(z):
        z = z.real
    z = z.astype(np.float32)

    # Pick the correct axis to work on
    if z.ndim > 1:
        ind = [np.newaxis] * z.ndim
        ind[axis] = slice(None)
    N = z.shape[axis]

    # Hilbert transform
    z = fftpack.fft(z, axis=axis)
    h = np.zeros(N)
    if N % 2 == 0:
        h[0] = h[N // 2] = 1
        h[1:N // 2] = 2
    else:
        h[0] = 1
        h[1:(N + 1) // 2] = 2
    if z.ndim > 1:
        h = h[tuple(ind)]
    z = fftpack.ifft(z * h, axis=axis)

    # Frequency shift signal by -B/2
    h = np.exp(-1j * np.pi / 2 * np.arange(N))
    if z.ndim > 1:
        h = h[tuple(ind)]
    z *= h

    # Decimate signal by factor of 2
    dec = [slice(None)] * z.ndim
    dec[axis] = slice(None, None, 2)
    z = z[tuple(dec)]
    return z.astype(np.complex64)


def taper_function(freqs, bandwidth):
    x = (freqs / bandwidth).to_value(u.one)
    taper = 1 + (x / 0.48)**80
    return taper
