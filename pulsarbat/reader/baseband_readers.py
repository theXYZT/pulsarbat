"""Baseband reader classes."""

import baseband

if tuple(map(int, baseband.version.version.split('.'))) < (4,):
    raise ImportError('Require baseband >= 4.0')

import numpy as np
import astropy.units as u
from astropy.time import Time

from .base import AbstractReader
from ..core import Signal, DualPolarizationSignal
from ..utils import verify_scalar_quantity, real_to_complex

__all__ = ['BasebandReader', 'GUPPIRawReader']


class BasebandReader(AbstractReader):
    """Base class for data readable by the `~baseband` package.

    Parameters
    ----------
    fh : `~baseband.base.base.StreamReaderBase`
        A baseband stream reader handle that will read the baseband
        data.
    """
    def __init__(self, fh):
        if not isinstance(fh, baseband.base.base.StreamReaderBase):
            raise ValueError('fh must be a Baseband StreamReaderBase object.')

        self._fh = fh
        self._complex = self._fh.complex_data
        verify_scalar_quantity(self._fh.sample_rate, u.Hz)

    @property
    def sample_rate(self):
        """Sample rate of the complex baseband representation of the data."""
        if not self._fh.complex_data:
            sr = (self._fh.sample_rate / 2).to(u.MHz)
        else:
            sr = self._fh.sample_rate.to(u.MHz)
        return sr

    @property
    def start_time(self):
        """Time at first sample of data."""
        return Time(self._fh.start_time, format='isot', precision=9)

    @property
    def stop_time(self):
        """Time at the end of data (time after the last sample)."""
        return Time(self._fh.stop_time, format='isot', precision=9)

    def seek(self, offset, whence=0):
        """Seek to a specific read position.

        This works like a normal filehandle seek, but the offset is in samples
        (or a relative or absolute time).

        Parameters
        ----------
        offset : int, `~astropy.units.Quantity`, or `~astropy.time.Time`
            Offset to move to.  Can be an (integer) number of samples,
            an offset in time units, or an absolute time.
        whence : {0, 1, 2, 'start', 'current', or 'end'}, optional
            Like regular seek, the offset is taken to be from the start if
            ``whence=0`` (default), from the current position if 1,
            and from the end if 2.  One can alternativey use 'start',
            'current', or 'end' for 0, 1, or 2, respectively.  Ignored if
            ``offset`` is a time.
        """
        self._fh.seek(offset, whence)

    def tell(self, unit=None):
        """Current read position (relative to the start position).

        Parameters
        ----------
        unit : `~astropy.units.Unit` or str, optional
            Time unit the offset should be returned in.  By default, no unit
            is used, i.e., an integer enumerating samples is returned. For the
            special string 'time', the absolute time is calculated.

        Returns
        -------
        offset : int, `~astropy.units.Quantity`, or `~astropy.time.Time`
        """
        return self._fh.tell(unit)

    @property
    def time(self):
        """Timestamp of current read position."""
        return Time(self.tell(unit='time'), format='isot', precision=9)

    def _read_to_array(self, N):
        """Read N time samples of complex baseband data into a Numpy array."""
        if self._complex:
            shape = (N, ) + self._fh.sample_shape
            z = np.empty(shape, dtype=np.complex64, order='F')
            self._fh.read(out=z)
        else:
            shape = (2*N, ) + self._fh.sample_shape
            z = np.empty(shape, dtype=np.float32, order='F')
            self._fh.read(out=z)
            z = real_to_complex(z)
        return z

    def read(self, N, start=None):
        """Read N time samples of complex baseband data."""
        if start is not None:
            self.seek(start)
        read_time = self.time
        z = self._read_to_array(N)
        return Signal(z, sample_rate=self.sample_rate, start_time=read_time)


class GUPPIRawReader(BasebandReader):
    """Baseband reader for GUPPI raw voltage data format."""

    def __init__(self, fh):
        if not isinstance(fh, baseband.guppi.base.GUPPIStreamReader):
            raise ValueError('fh must be a GUPPIStreamReader object.')

        if fh.ndim != 3:
            raise ValueError('GUPPIStreamReader must have 3 dimensions.')

        super().__init__(fh)

        check_chan_bw = u.isclose(self.sample_rate,
                                  self.header['CHAN_BW'] * u.MHz)

        check_tbin = u.isclose(self.sample_rate,
                               1 / (self.header['TBIN'] * u.s))

        if not (check_tbin and check_chan_bw):
            err = 'StreamReader sample rate does not agree with GUPPI header!'
            raise ValueError(err)

    @property
    def header(self):
        return self._fh.header0

    @property
    def center_freq(self):
        return self.header['OBSFREQ'] * u.MHz

    @property
    def bandwidth(self):
        return self.header['OBSBW'] * u.MHz

    @property
    def pol_type(self):
        return {'LIN': 'linear', 'CIRC': 'circular'}[self.header['FD_POLN']]

    def read(self, N: int):
        _kwargs = {'sample_rate': self.sample_rate,
                   'center_freq': self.center_freq,
                   'bandwidth': self.bandwidth,
                   'pol_type': self.pol_type,
                   'start_time': self.time}

        z = self._read_to_array(N).transpose(0, 2, 1)
        return DualPolarizationSignal(z, **_kwargs)
