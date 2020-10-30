"""Baseband reader classes."""

import baseband

if tuple(map(int, baseband.version.version.split('.'))) < (4,):
    raise ImportError('Require baseband >= 4.0')

import numpy as np
import astropy.units as u
from astropy.time import Time

from .base import AbstractReader
from ..core import Signal, DualPolarizationSignal
from ..utils import verify_scalar_quantity, times_are_close, real_to_complex

__all__ = ['BasebandReader', 'GUPPIRawReader']


class BasebandRawReader(AbstractReader):
    """Base class for raw voltage data readable by the `~baseband` package.

    Parameters
    ----------
    fh : `~baseband.base.base.StreamReaderBase`
        A baseband stream reader handle that will read the baseband
        data.
    """
    def __init__(self, fh):
        if not isinstance(fh, baseband.base.base.StreamReaderBase):
            raise ValueError('fh must be a Baseband StreamReaderBase object.')

        expected_stop_time = fh.start_time + (fh.shape[0] / fh.sample_rate)
        if not times_are_close(fh.stop_time, expected_stop_time):
            err = 'StreamReader stop time does not match number of samples.'
            raise ValueError(err)

        verify_scalar_quantity(fh.sample_rate, u.Hz)
        self._fh = fh

    @property
    def sample_rate(self):
        """Sample rate of the complex baseband representation of the data."""
        if self._fh.complex_data:
            return self._fh.sample_rate.to(u.MHz)
        else:
            return self._fh.sample_rate.to(u.MHz) / 2

    @property
    def start_time(self):
        """Time at first sample of data."""
        return Time(self._fh.start_time, format='isot', precision=9)

    @property
    def stop_time(self):
        """Time at the end of data (time after the last sample)."""
        end = self.start_time + (len(self) / self.sample_rate)
        return Time(end, format='isot', precision=9)

    def __len__(self):
        if self._fh.complex_data:
            return self._fh.shape[0]
        else:
            return self._fh.shape[0] // 2

    def seek(self, offset, whence=0):
        """Seek to a specific read position.

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
        if self._fh.complex_data:
            self._fh.seek(offset, whence)
        elif isinstance(offset, int):
            self._fh.seek(offset * 2, whence)
        else:
            self._fh.seek(offset, whence)
            self._fh.seek(self.tell() * 2)

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
        if unit is None and not self._fh.complex_data:
            return self._fh.tell() // 2
        else:
            return self._fh.tell(unit)

    @property
    def time(self):
        """Timestamp at current read position."""
        return Time(self.tell(unit='time'), format='isot', precision=9)

    def _read_array(self, N, offset=None):
        """Read N samples at given offset into a Numpy array."""
        if offset is not None:
            self.seek(offset)

        if self._fh.complex_data:
            shape = (N, ) + self._fh.sample_shape
            z = np.empty(shape, dtype=np.complex64, order='F')
            self._fh.read(out=z)
        else:
            shape = (2*N, ) + self._fh.sample_shape
            z = np.empty(shape, dtype=np.float32, order='F')
            self._fh.read(out=z)
            z = real_to_complex(z)
        return z

    def read_data(self, N, offset=None):
        return self._read_array(N, offset)

    @property
    def signal_kwargs(self):
        kw = {'sample_rate': self.sample_rate, }
        return kw

    def read(self, N, offset=None):
        """Read N samples at given offset into a Signal object."""
        self.seek(offset)
        start_time = self.time

        z = self.read_data(N, offset)
        return Signal(z, start_time=start_time, **self.signal_kwargs)


class GUPPIRawReader(BasebandRawReader):
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

    @property
    def signal_kwargs(self):
        kw = {'sample_rate': self.sample_rate,
              'center_freq': self.center_freq,
              'bandwidth': self.bandwidth,
              'pol_type': self.pol_type}
        return kw

    def read(self, N, offset=None):
        """Read N samples at given offset into a Signal object."""
        self.seek(offset)
        start_time = self.time

        z = self.read_data(N, offset).transpose(0, 2, 1)
        return DualPolarizationSignal(z, start_time=start_time,
                                      **self.signal_kwargs)
