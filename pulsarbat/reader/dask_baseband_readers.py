"""Dask-enabled Baseband reader classes."""

import dask
import dask.array as da
from .baseband_readers import BasebandReader, GUPPIRawReader, DADAStokesReader

__all__ = [
    'DaskBasebandReader', 'DaskGUPPIRawReader', 'DaskDADAStokesReader'
]


class DaskBasebandReader(BasebandReader):
    """Dask-enabled baseband reader for raw voltage data."""
    def read_data(self, N, offset):
        _shape = (N, ) + self._fh.sample_shape
        _dtype = self._fh.dtype
        _chunk = (-1, ) + ('auto', ) * len(self._fh.sample_shape)

        _delayed_func = dask.delayed(self._read_array, pure=True)
        z = da.from_delayed(_delayed_func(N, offset),
                            shape=_shape,
                            dtype=_dtype)
        return da.rechunk(z, chunks=_chunk)


class DaskGUPPIRawReader(DaskBasebandReader, GUPPIRawReader):
    """Dask-enabled baseband reader for GUPPI raw voltage data format."""
    pass


class DaskDADAStokesReader(DaskBasebandReader, DADAStokesReader):
    """Dask-enabled baseband reader for GUPPI raw voltage data format."""
    pass
