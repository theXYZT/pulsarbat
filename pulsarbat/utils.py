"""Collection of handy utilities."""

import numpy as np
from scipy.fft import next_fast_len
# import astropy.units as u


__all__ = [
    'real_to_complex',
    'next_fast_len',
]


def real_to_complex(z, axis=0):
    """Convert a real baseband signal to a complex baseband signal.

    This function computes the analytic representation of the input
    signal via a Hilbert transform, throwing away negative frequency
    components. Then, the signal is shifted in frequency domain by -B/2
    where B is the bandwidth of the signal. Finally, the signal is
    decimated by a factor of 2, which results in the complex baseband
    representation of the input signal.

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
    """
    if np.iscomplexobj(z):
        z = z.real

    out_dtype = np.complex128 if z.dtype == np.float64 else np.complex64

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
    return z.astype(out_dtype)


# def taper_function(freqs, bandwidth):
#     x = (freqs / bandwidth).to_value(u.one)
#     taper = 1 + (x / 0.48)**80
#     return taper
