"""Baseband reader classes."""

import baseband

if tuple(map(int, baseband.version.version.split('.'))) < (4,):
    raise ImportError('Require baseband >= 4.0')

import numpy as np
import astropy.units as u
from astropy.time import Time

from .base import AbstractReader
from ..core import Signal, BasebandSignal, DualPolarizationSignal
from ..utils import verify_scalar_quantity, times_are_close, real_to_complex

__all__ = ['BasebandReader', 'BasebandRawReader', 'GUPPIRawReader']


class BasebandReader(AbstractReader):
    """Base class for data readable by the `~baseband` package.

    Parameters
    ----------
    fh : `~baseband.base.base.StreamReaderBase`
        A baseband stream reader handle that will read the baseband
        data.
    """
    def __init__(self, fh, /):
        if not isinstance(fh, baseband.base.base.StreamReaderBase):
            raise ValueError('fh must be a Baseband StreamReaderBase object.')

        self._fh = fh

        if not times_are_close(fh.stop_time, self.stop_time):
            err = 'StreamReader stop time does not match number of samples.'
            raise ValueError(err)

    @property
    def start_time(self):
        """Time at first sample of data."""
        return Time(self._fh.start_time, format='isot', precision=9)

    @property
    def stop_time(self):
        """Time at the end of data (time after the last sample)."""
        end = self.start_time + (len(self) / self.sample_rate)
        return Time(end, format='isot', precision=9)

    @property
    def sample_rate(self):
        """Sample rate of the data."""
        return self._fh.sample_rate.to(u.MHz)

    def __len__(self):
        return self._fh.shape[0]

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
        """Timestamp at current read position."""
        return Time(self.tell(unit='time'), format='isot', precision=9)

    def _read_array(self, N, offset=None):
        """Read N samples at given offset into a Numpy array."""
        if offset is not None:
            self.seek(offset)
        return self._fh.read(N)

    def read_data(self, N, offset=None):
        return self._read_array(N, offset)

    def read(self, N, offset=None):
        """Read N samples at given offset into a Signal object."""
        self.seek(offset)
        start_time = self.time
        return Signal(self.read_data(N, offset), start_time=start_time,
                      sample_rate=self.sample_rate)


class BasebandRawReader(BasebandReader):
    """Base class for raw voltage data readable by the `~baseband` package.

    Parameters
    ----------
    fh : `~baseband.base.base.StreamReaderBase`
        A baseband stream reader handle that will read the baseband
        data.
    """
    def __init__(self, fh, /, center_freq, bandwidth, sideband):
        super().__init__(fh)

        if verify_scalar_quantity(center_freq, u.Hz):
            self.center_freq = center_freq.to(u.MHz)

        if verify_scalar_quantity(bandwidth, u.Hz):
            self.bandwidth = bandwidth.to(u.MHz)

        if type(sideband) is bool:
            self.sideband = sideband
        else:
            self.sideband = np.array(sideband)
            if len(fh.sample_shape) != self.sideband.shape:
                err = "StreamReader sample shape != sideband shape"
                raise ValueError(err)

    @property
    def sample_rate(self):
        """Sample rate of the complex baseband representation of the data."""
        if self._fh.complex_data:
            return self._fh.sample_rate.to(u.MHz)
        else:
            return self._fh.sample_rate.to(u.MHz) / 2

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
            z = np.empty(shape, dtype=np.float64, order='F')
            self._fh.read(out=z)
            z = real_to_complex(z)

        if self.sideband is False:
            z = z.conj()
        elif self.sideband is not True:
            slc = np.logical_not(self.sideband)
            z[:, slc] = z[:, slc].conj()

        return z

    def read_data(self, N, offset=None):
        return self._read_array(N, offset)

    def read(self, N, offset=None):
        """Read N samples at given offset into a Signal object."""
        kwargs = {'sample_rate': self.sample_rate,
                  'center_freq': self.center_freq,
                  'bandwidth': self.bandwidth}

        self.seek(offset)
        kwargs['start_time'] = self.time

        return BasebandSignal(self.read_data(N, offset), **kwargs)


class GUPPIRawReader(BasebandRawReader):
    """Baseband reader for GUPPI raw voltage data format."""

    def __init__(self, fh, /):
        if not isinstance(fh, baseband.guppi.base.GUPPIStreamReader):
            raise ValueError('fh must be a GUPPIStreamReader object.')

        if fh.ndim != 3:
            raise ValueError('GUPPIStreamReader must have 3 dimensions.')

        if not fh.header0['OBS_MODE'] == 'RAW':
            err = 'GUPPI data is not raw voltage data according to header.'
            raise ValueError(err)

        _header_bw = abs(fh.header0['OBSBW']) * u.MHz
        _fh_bw = fh.sample_rate * fh.sample_shape.nchan

        if not u.isclose(_fh_bw, _header_bw):
            err = "StreamReader sample rate and nchan don't match bandwidth"
            raise ValueError(err)

        if not fh.header0.sideband == (fh.header0['CHAN_BW'] > 0):
            err = 'StreamReader sideband not consistent with GUPPI header!'
            raise ValueError(err)

        self._fh = fh

        if not times_are_close(fh.stop_time, self.stop_time):
            err = 'StreamReader stop time does not match number of samples.'
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
    def sideband(self):
        return self.header.sideband

    def read(self, N, offset=None):
        """Read N samples at given offset into a Signal object."""
        kwargs = {'sample_rate': self.sample_rate,
                  'center_freq': self.center_freq,
                  'bandwidth': self.bandwidth,
                  'pol_type': self.pol_type}

        self.seek(offset)
        kwargs['start_time'] = self.time

        z = self.read_data(N, offset).transpose(0, 2, 1)
        return DualPolarizationSignal(z, **kwargs)
