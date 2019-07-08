"""Signal-to-signal transforms."""

import numpy as np
try:
    import pyfftw
    fftpack = pyfftw.interfaces.numpy_fft
except ImportError:
    fftpack = np.fft

__all__ = ['real_to_complex', 'complex_to_real']


def real_to_complex(z, axis=0):
    """Convert a real baseband signal to a complex baseband signal.

    This function computes the analytic representation of the input signal
    via a Hilbert transform, throwing away negative frequency components.
    Then, the signal is shifted in frequency domain by -B/2 where B is the
    bandwidth of the signal. Finally, the signal is decimated by a factor
    of 2, which results in the complex baseband representation of the input
    signal (See [1]_).

    Parameters
    ----------
    z : array_like
        Input array, only real part is used.
    axis : int, optional
        Axis over which to convert the signal. This will be the axis that
        represents time. Default is 0.

    Returns
    -------
    out : complex ndarray
        The complex baseband representation of the input signal, transformed
        along the `axis`.

    Raises
    ------
    TypeError
        If `z` is complex-valued.
    IndexError
        if `axes` is larger than the last axis of `z`.

    See Also
    --------
    complex_to_real : The inverse of `real_to_complex`.

    Notes
    -----
    This function assumes the input signal is a causal signal.

    References
    ----------
    .. [1] https://dsp.stackexchange.com/a/43281/17721

    """
    z = np.asarray(z).real

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

    return z


def complex_to_real(z, axis=0):
    """Convert a complex baseband signal to a real baseband signal."""
    pass
