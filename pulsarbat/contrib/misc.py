"""Experimental routines."""

# flake8: noqa

import numpy as np
import astropy.units as u
from astropy.time import Time
import pulsarbat as pb


__all__ = [
    "stft",
    "istft",
]


def stft(z, /, window="boxcar", nperseg=256, noverlap=0, nfft=None):
    """Performs a short-time Fourier transform on a baseband signal.

    Behaves the same as `scipy.signal.stft`. Currently, only supports
    `window='boxcar'`, `noverlap=0` and `nfft = None`. This results in
    the critically-sampled perfect reconstruction STFT - using a
    rectangular window function with no overlap.

    For now, users should only use the `z` and `nperseg` arguments.

    When fully implemented, should behave exactly as `scipy.signal.stft`
    with support for different window and overlap configurations, and
    lazy execution via dask.
    """
    if window != "boxcar" or noverlap != 0 or nfft != None:
        return NotImplemented

    if not isinstance(z, pb.BasebandSignal):
        raise ValueError("z must be a BasebandSignal.")

    if nfft is None:
        nfft = nperseg

    nperseg, noverlap, nfft = int(nperseg), int(noverlap), int(nfft)
    z = z[: len(z) - len(z) % nperseg, :]

    new_shape = (-1, nperseg) + z.sample_shape
    x = z.data.reshape(new_shape)
    x = x.swapaxes(1, 2)

    x = pb.fft.fft(x, axis=2, n=nfft)
    x = np.fft.fftshift(x, axes=(2,))

    out_shape = (x.shape[0], -1) + x.shape[3:]
    x = x.reshape(out_shape)
    x /= nperseg

    falign = "center" if nfft % 2 else "bottom"
    return type(z).like(z, x, sample_rate=z.sample_rate / nfft, freq_align=falign)


def istft(z, /, window="boxcar", nperseg=256, noverlap=0, nfft=None):
    """Performs an inverse short-time Fourier transform.

    Behaves the same as `scipy.signal.istft`. Currently, only supports
    `window='boxcar'`, `noverlap=0` and `nfft = None`.

    For now, users should only use the `z` and `nperseg` arguments.

    When fully implemented, should behave exactly as `scipy.signal.istft`
    with support for different window and overlap configurations, and
    lazy execution via dask.
    """
    if window != "boxcar" or noverlap != 0 or nfft != None:
        return NotImplemented

    if not isinstance(z, pb.BasebandSignal):
        raise ValueError("z must be a BasebandSignal.")

    if nfft is None:
        nfft = nperseg

    nperseg, noverlap, nfft = int(nperseg), int(noverlap), int(nfft)

    new_shape = (len(z), -1, nfft) + z.sample_shape[1:]
    x = z.data.reshape(new_shape)
    x *= nperseg

    x = x.swapaxes(1, 2)
    x = np.fft.ifftshift(x, axes=(1,))
    x = pb.fft.ifft(x, axis=1, n=nfft)
    x = x[:, :nperseg]

    out_shape = (-1,) + x.shape[2:]
    x = x.reshape(out_shape)

    return type(z).like(z, x, sample_rate=z.sample_rate * nfft, freq_align="center")
