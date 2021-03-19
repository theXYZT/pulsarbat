"""Experimental routines."""

# flake8: noqa

import numpy as np
import astropy.units as u
from astropy.time import Time
import pulsarbat as pb


__all__ = [
    'stft',
    'istft',
    'phase_deconvolution',
]


def stft(z, /, window='boxcar', nperseg=256, noverlap=0, nfft=None):
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
    if window != 'boxcar' or noverlap != 0 or nfft != None:
        return NotImplemented

    if not isinstance(z, pb.BasebandSignal):
        raise ValueError("z must be a BasebandSignal.")

    if nfft is None:
        nfft = nperseg

    nperseg, noverlap, nfft = int(nperseg), int(noverlap), int(nfft)
    z = z[:len(z) - len(z) % nperseg, :]

    x = z.data.reshape((-1, nperseg) + z.sample_shape).swapaxes(1, 2)
    x = np.fft.fftshift(np.fft.fft(x, axis=2, n=nfft), axes=(2,))
    x = x.reshape((x.shape[0], -1) + x.shape[3:]) / nperseg

    out_sr = z.sample_rate / nfft
    falign = 'center' if nfft % 2 else 'bottom'
    return type(z).like(z, x, sample_rate=z.sample_rate / nfft,
                        freq_align=falign)


def istft(z, /, window='boxcar', nperseg=256, noverlap=0, nfft=None):
    """Performs an inverse short-time Fourier transform.

    Behaves the same as `scipy.signal.istft`. Currently, only supports
    `window='boxcar'`, `noverlap=0` and `nfft = None`.

    For now, users should only use the `z` and `nperseg` arguments.

    When fully implemented, should behave exactly as `scipy.signal.istft`
    with support for different window and overlap configurations, and
    lazy execution via dask.
    """
    if window != 'boxcar' or noverlap != 0 or nfft != None:
        return NotImplemented

    if not isinstance(z, pb.BasebandSignal):
        raise ValueError("z must be a BasebandSignal.")

    if nfft is None:
        nfft = nperseg

    nperseg, noverlap, nfft = int(nperseg), int(noverlap), int(nfft)

    x = z.data.reshape((len(z), -1, nfft) + z.sample_shape[1:]) * nperseg
    x = np.fft.ifftshift(x.swapaxes(1, 2), axes=(1,))
    x = np.fft.ifft(x, axis=1, n=nfft)[:, :nperseg]
    x = x.reshape((-1,) + x.shape[2:])

    out_sr = z.sample_rate * nfft
    return type(z).like(z, x, sample_rate=z.sample_rate * nfft,
                        freq_align='center')


def phase_deconvolution(z, h):
    """Performs a deconvolution using only the phases of an inverse filter."""
    if not isinstance(z, pb.BasebandSignal):
        raise ValueError("z must be a BasebandSignal.")

    if not isinstance(h, pb.Signal):
        raise ValueError("h must be a BasebandSignal.")

    N, Nh = len(z), len(h)
    if N < Nh:
        raise ValueError("h can not be longer than z.")

    if h.ndim > z.ndim:
        raise ValueError("h can not have more dimensions than z.")

    h = h[(slice(None),) * h.ndim + (None,) * (z.ndim - h.ndim)]

    sig = np.fft.fft(z.data, axis=0)
    filt = np.fft.fft(h.data, axis=0, n=N)
    filt = np.where(np.abs(filt) > 1E-20, filt / np.abs(filt), 1)
    x = np.fft.ifft(sig / filt, axis=0)
    return type(z).like(z, x)[:N-Nh+1]
