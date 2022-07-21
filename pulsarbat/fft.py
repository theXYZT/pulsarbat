"""FFT Functions with proper dispatching for Dask Arrays."""

import scipy.fft
from functools import singledispatch
import dask.array as da


_FFT_FUNCS = [
    "fft",
    "fft2",
    "fftn",
    "ifft",
    "ifft2",
    "ifftn",
    "rfft",
    "rfft2",
    "rfftn",
    "irfft",
    "irfft2",
    "irfftn",
    "hfft",
    "ihfft",
]


def __dir__():
    return sorted(_FFT_FUNCS)


def __getattr__(name):
    if name not in _FFT_FUNCS:
        raise AttributeError(f"module {__name__} has no attribute {name}")

    _fft_func = getattr(scipy.fft, name)

    @singledispatch
    def func(*args, **kwargs):
        return _fft_func(*args, **kwargs)

    @func.register(da.Array)
    def _(*args, **kwargs):
        wrapped_func = da.fft.fft_wrap(_fft_func)
        return wrapped_func(*args, **kwargs)

    func.__qualname__ = _fft_func.__qualname__
    func.__name__ = _fft_func.__name__
    func.__doc__ = _fft_func.__doc__
    return func
