"""Collection of handy utilities."""

import numpy as np
from functools import lru_cache
import scipy.fft


__all__ = [
    "real_to_complex",
    "next_fast_len",
    "prev_fast_len",
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

    out_dtype = np.complex64 if z.dtype == np.float32 else np.complex128
    N = z.shape[axis]

    if N == 0:
        return z.astype(out_dtype)

    # Pick the correct axis to work on
    ind = [np.newaxis] * z.ndim
    ind[axis] = slice(None)

    # Get analytic signal via Hilbert transform and shift by -B/2
    h = np.zeros(N, dtype=out_dtype)
    h[0] = 1
    h[1 : N // 2] = 2
    if N > 1:
        h[N // 2] = 2 if N % 2 else 1

    z = scipy.fft.ifft(scipy.fft.fft(z, axis=axis) * h[tuple(ind)], axis=axis)
    z *= np.exp(-1j * np.pi / 2 * np.arange(N))[tuple(ind)]

    # Decimate signal by factor of 2 (along axis)
    dec = [slice(None)] * z.ndim
    dec[axis] = slice(None, None, 2)
    return z[tuple(dec)].astype(out_dtype)


@lru_cache(maxsize=1024)
def next_fast_len(N):
    """Returns smallest 7-smooth number >= N."""
    if N <= 10:
        return N

    f7, guess = 1, 2 * N
    while f7 < guess:
        f75 = f7
        while f75 < guess:
            x = f75

            while x < N:
                x *= 2

            while 1:
                if x < N:
                    x *= 3
                elif x > N:
                    if x < guess:
                        guess = x
                    if x & 1:
                        break
                    x >>= 1
                else:
                    return N

            f75 *= 5
        f7 *= 7
    return guess


@lru_cache(maxsize=1024)
def prev_fast_len(N):
    """Returns largest 7-smooth number <= N."""
    if N <= 10:
        return N

    f7, guess = 1, 1
    while f7 <= N:
        f75 = f7
        while f75 <= N:
            x = f75

            while x <= N:
                x *= 2
            x >>= 1

            while 1:
                if x < N:
                    if x > guess:
                        guess = x
                    x *= 3
                elif x > N:
                    if x & 1:
                        break
                    x >>= 1
                else:
                    return N

            f75 *= 5
        f7 *= 7
    return guess
