"""Dask-enabled Baseband reader classes."""

import contextlib
import dask
import dask.array as da
from .baseband_readers import (BasebandReader, BasebandRawReader,
                               GUPPIRawReader, DADAStokesReader)

__all__ = [
    'DaskBasebandReader', 'DaskBasebandRawReader', 'DaskGUPPIRawReader',
    'DaskDADAStokesReader'
]


class DaskBasebandReader(BasebandReader):
    """Dask-enabled wrapper around StreamReader from the `~baseband` package.

    Parameters
    ----------
    name, **kwargs
        Arguments to pass on to `~baseband.open` to create a StreamReader
        object via `baseband.open(name, 'rs', **kwargs)`.
    """
    @dask.delayed(pure=False)
    def _read_delayed(self, n, offset, /, lock=contextlib.nullcontext()):
        with lock:
            return self._read_array(n, offset)

    def _read_stream(self, n, /, **kwargs):
        """Read N samples from current stream position into array-like."""
        z = da.from_delayed(self._read_delayed(n, self.offset, **kwargs),
                            dtype=self.dtype, shape=(n,) + self.sample_shape)

        _chunk = (-1, ) + ('auto', ) * len(self.sample_shape)
        return da.rechunk(z, chunks=_chunk)


class DaskBasebandRawReader(DaskBasebandReader, BasebandRawReader):
    """Dask-enabled baseband reader for raw voltage data."""
    pass


class DaskGUPPIRawReader(DaskBasebandReader, GUPPIRawReader):
    """Dask-enabled baseband reader for GUPPI raw voltage data format."""
    pass


class DaskDADAStokesReader(DaskBasebandReader, DADAStokesReader):
    """Dask-enabled baseband reader for GUPPI raw voltage data format."""
    pass
