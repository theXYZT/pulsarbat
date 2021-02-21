"""Collection of handy utilities."""

import numpy as np
from scipy.fft import next_fast_len


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

    Parameters
    ----------
    z : `~numpy.ndarray`
        Input signal. Must be real.
    axis : int, optional
        Axis over which to convert the signal. This will be the axis
        that represents time. Default is 0.

    Returns
    -------
    out : np.ndarray
        The complex baseband representation of the input signal.
    """
    z = np.asarray(z)
    if np.iscomplexobj(z):
        raise ValueError("Input must be real-valued.")

    out_dtype = np.complex128 if z.dtype == np.float64 else np.complex64
    N = z.shape[axis]

    if N < 1:
        raise ValueError(f"Invalid number of data points ({N}).")

    # Pick the correct axis to work on
    ind = [np.newaxis] * z.ndim
    ind[axis] = slice(None)

    # Get analytic signal via Hilbert transform and shift by -B/2
    h = np.zeros(N)
    h[0] = 1
    h[1:N//2] = 2
    h[N//2] = 2 if N % 2 else 1

    z = np.fft.ifft(np.fft.fft(z, axis=axis) * h[tuple(ind)], axis=axis)
    z *= np.exp(-1j * np.pi / 2 * np.arange(N))[tuple(ind)]

    # Decimate signal by factor of 2 (along axis)
    dec = [slice(None)] * z.ndim
    dec[axis] = slice(None, None, 2)
    return z[tuple(dec)].astype(out_dtype)
