"""Dask-enabled Baseband reader classes."""

import dask
import dask.array as da
from .baseband_readers import BasebandRawReader, GUPPIRawReader

__all__ = ['DaskBasebandRawReader', 'DaskGUPPIRawReader', ]


class DaskBasebandRawReader(BasebandRawReader):
    """Dask-enabled baseband reader for raw voltage data."""
    def read_data(self, N, offset=None):
        _shape = (N,) + self._fh.sample_shape
        _dtype = self._fh.dtype
        _chunk = (N,) + (1,) * len(self._fh.sample_shape)

        _delayed_func = dask.delayed(self._read_array, pure=True)
        z = da.from_delayed(_delayed_func(N, offset), shape=_shape,
                            dtype=_dtype)
        return da.rechunk(z, chunks=_chunk)


class DaskGUPPIRawReader(DaskBasebandRawReader, GUPPIRawReader):
    """Dask-enabled baseband reader for GUPPI raw voltage data format."""
    pass
