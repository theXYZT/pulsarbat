"""Collection of handy utilities."""

import numpy as np
import astropy.units as u

__all__ = [
    'verify_scalar_quantity', 'complex_noise', 'real_to_complex'
]


def verify_scalar_quantity(a, unit):
    if not isinstance(a, u.Quantity):
        raise TypeError(f'Expected astropy Quantity, got {type(a)}')

    if not a.unit.is_equivalent(unit):
        expected = f'Expected units of {unit.physical_type}'
        raise u.UnitTypeError(f'{expected}, got units of {a.unit}')

    if not a.isscalar:
        raise ValueError(f'Expected a scalar quantity.')

    return True


def complex_noise(shape: tuple, power: float):
    """Generates complex gaussian noise with given shape and power."""
    r = np.random.normal(0, 1 / np.sqrt(2), shape)
    i = np.random.normal(0, 1 / np.sqrt(2), shape)
    return (r + 1j * i) * np.sqrt(power)


def abs2(x):
    """Returns the absolute square of an array."""
    return x.real**2 + x.imag**2


def taper_function(freqs, bandwidth):
    x = (freqs / bandwidth).to_value(u.one)
    taper = 1 + (x / 0.48)**80
    return taper


def real_to_complex(z, axis=0):
    """
    Convert a real baseband signal to a complex baseband signal.

    This function computes the analytic representation of the input
    signal via a Hilbert transform, throwing away negative frequency
    components. Then, the signal is shifted in frequency domain by -B/2
    where B is the bandwidth of the signal. Finally, the signal is
    decimated by a factor of 2, which results in the complex baseband
    representation of the input signal [1]_.

    Parameters
    ----------
    z : BasebandSignal
        Input signal, must be real.
    axis : int, optional
        Axis over which to convert the signal. This will be the axis
        that represents time. If not given, the last axis is used.

    Returns
    -------
    out : BasebandSignal
        The complex baseband representation of the input signal.

    Raises
    ------
    TypeError
        If input parameters are not of the right type.
    IndexError
        if `axes` is larger than the last axis of `z`.

    Notes
    -----
    This function assumes the input signal is a causal signal.

    References
    ----------
    .. [1] https://dsp.stackexchange.com/q/43278/17721
    """
    if np.iscomplexobj(z):
        raise TypeError('Signal is already complex.')

    # Pick the correct axis to work on
    if z.ndim > 1:
        ind = [np.newaxis] * z.ndim
        ind[axis] = slice(None)
    N = z.shape[axis]

    # Hilbert transform
    z = np.fft.fft(z, axis=axis)
    h = np.zeros(N)
    if N % 2 == 0:
        h[0] = h[N // 2] = 1
        h[1:N // 2] = 2
    else:
        h[0] = 1
        h[1:(N + 1) // 2] = 2
    if z.ndim > 1:
        h = h[tuple(ind)]
    z = np.fft.ifft(z * h, axis=axis)

    # Frequency shift signal by -B/2
    h = np.exp(-1j * np.pi / 2 * np.arange(N))
    if z.ndim > 1:
        h = h[tuple(ind)]
    z *= h

    # Decimate signal by factor of 2
    dec = [slice(None)] * z.ndim
    dec[axis] = slice(None, None, 2)
    z = z[tuple(dec)]

    z = z.astype(np.complex64)

    return z
