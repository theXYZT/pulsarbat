"""Baseband reader classes."""

import baseband

if tuple(map(int, baseband.version.version.split('.'))) < (4,):
    raise ImportError('Require baseband >= 4.0')

import operator
import numpy as np
import astropy.units as u
from astropy.time import Time

from .base import AbstractReader
from ..core import (Signal, BasebandSignal, DualPolarizationSignal,
                    FullStokesSignal)
from ..utils import real_to_complex

__all__ = ['BasebandReader', 'BasebandRawReader', 'GUPPIRawReader',
           'DADAStokesReader']


def _times_are_close(t1, t2):
    return np.abs(t1 - t2) < 0.1 * u.ns


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

        if not _times_are_close(fh.stop_time, self.stop_time):
            err = 'StreamReader stop time does not match number of samples.'
            raise ValueError(err)

    @property
    def header(self):
        return self._fh.header0

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
        return self._fh.seek(offset, whence)

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

    def _read_array(self, N, offset):
        """Read N samples at given offset into a Numpy array."""
        self.seek(offset)
        return self._fh.read(N)

    def read_data(self, N, offset):
        return self._read_array(N, offset)

    def read(self, N, offset=None):
        """Read N samples at given offset into a Signal object."""
        offset = self.tell() if offset is None else self.seek(offset)
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
    center_freq : `~astropy.units.Quantity`
        The observing frequency at the center of the signal's band. Must
        be in units of frequency.
    bandwidth : `~astropy.units.Quantity`
        The total bandwidth of the signal. The channel bandwidth is this
        total bandwidth divided by the number of channels. Must be in
        units of frequency.
    sideband : bool or array-like
        False indicates spectral flip (lower sideband representation),
        True indicates no spectral flip (upper sideband representation).
        If channels have different sidebands, an array-like object of
        boolean elements can be passed (must have the same shape as the
        StreamReader's sample shape).
    """
    def __init__(self, fh, /, center_freq, bandwidth, sideband=True):
        super().__init__(fh)

        try:
            self._center_freq = center_freq.to(u.MHz)
            assert self._center_freq.isscalar
        except Exception:
            raise ValueError("Invalid value for center_freq!")

        try:
            self._bandwidth = bandwidth.to(u.MHz)
            assert self._bandwidth.isscalar
        except Exception:
            raise ValueError("Invalid value for bandwidth!")

        if type(sideband) is bool:
            self._sideband = sideband
        else:
            self._sideband = np.array(sideband).astype(bool)
            if fh.sample_shape != self.sideband.shape:
                err = "StreamReader sample shape != sideband shape"
                raise ValueError(err)

    @property
    def sideband(self):
        """True if upper sideband, False if lower sideband."""
        return self._sideband

    @property
    def sample_rate(self):
        """Sample rate of the complex baseband representation of the data."""
        if self._fh.complex_data:
            return self._fh.sample_rate.to(u.MHz)
        else:
            return self._fh.sample_rate.to(u.MHz) / 2

    @property
    def center_freq(self):
        """Center observing frequency of the signal."""
        return self._center_freq

    @property
    def bandwidth(self):
        """Total bandwidth of signal."""
        return self._bandwidth

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
            return self._fh.seek(offset, whence)

        try:
            offset = operator.index(offset)
        except Exception:
            try:
                offset = offset - self.start_time
            except Exception:
                pass
            else:
                whence = 0

            offset = int((offset * self.sample_rate).to(u.one).round())

        return self._fh.seek(offset * 2, whence) // 2

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

    def _read_array(self, N, offset):
        """Read N samples at given offset into a Numpy array."""
        self.seek(offset)

        if self._fh.complex_data:
            shape = (N, ) + self._fh.sample_shape
            z = np.empty(shape, dtype=np.complex64, order='F')
            self._fh.read(out=z)
        else:
            shape = (2*N, ) + self._fh.sample_shape
            z = np.empty(shape, dtype=np.float64, order='F')
            self._fh.read(out=z)
            z = real_to_complex(z, axis=0)

        if self.sideband is False:
            z = z.conj()
        elif self.sideband is not True:
            slc = np.logical_not(self.sideband)
            z[:, slc] = z[:, slc].conj()

        return z

    def read(self, N, offset=None):
        """Read N samples at given offset into a Signal object."""
        kwargs = {'sample_rate': self.sample_rate,
                  'center_freq': self.center_freq,
                  'bandwidth': self.bandwidth}

        offset = self.tell() if offset is None else self.seek(offset)
        kwargs['start_time'] = self.time

        return BasebandSignal(self.read_data(N, offset), **kwargs)


class GUPPIRawReader(BasebandRawReader):
    """Baseband reader for GUPPI raw voltage data format.

    Parameters
    ----------
    fh : `~baseband.guppi.base.GUPPIStreamReader`
        A GUPPI Stream Reader handle that will provide access to raw
        voltage data.
    """

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

        if not _times_are_close(fh.stop_time, self.stop_time):
            err = 'StreamReader stop time does not match number of samples.'
            raise ValueError(err)

    @property
    def sideband(self):
        return self.header.sideband

    @property
    def center_freq(self):
        return self.header['OBSFREQ'] * u.MHz

    @property
    def bandwidth(self):
        return abs(self.header['OBSBW']) * u.MHz

    @property
    def pol_type(self):
        return {'LIN': 'linear', 'CIRC': 'circular'}[self.header['FD_POLN']]

    def read(self, N, offset=None):
        """Read N samples at given offset into a Signal object."""
        kwargs = {'sample_rate': self.sample_rate,
                  'center_freq': self.center_freq,
                  'bandwidth': self.bandwidth,
                  'pol_type': self.pol_type}

        offset = self.tell() if offset is None else self.seek(offset)
        kwargs['start_time'] = self.time

        z = self.read_data(N, offset).transpose(0, 2, 1)
        return DualPolarizationSignal(z, **kwargs)


class DADAStokesReader(BasebandReader):
    """Reader for full Stokes data in DADA format.

    Parameters
    ----------
    fh : `~baseband.base.base.StreamReaderBase`
        A baseband stream reader handle that will read the baseband
        data.
    """
    def __init__(self, fh, /):
        if not isinstance(fh, baseband.dada.base.DADAStreamReader):
            raise ValueError('fh must be a Baseband DADAStreamReader object.')

        if fh.ndim != 3:
            raise ValueError('DADAStreamReader must have 3 dimensions.')

        if not (fh.header0['NPOL'] == 4 and fh.header0['NDIM'] == 1):
            raise ValueError('Does not look like Stokes data')

        self._fh = fh

        if not _times_are_close(fh.stop_time, self.stop_time):
            err = 'StreamReader stop time does not match number of samples.'
            raise ValueError(err)

    @property
    def sideband(self):
        return self.header.sideband

    @property
    def center_freq(self):
        return self.header['FREQ'] * u.MHz

    @property
    def bandwidth(self):
        return abs(self.header['BW']) * u.MHz

    def read(self, N, offset=None):
        """Read N samples at given offset into a Signal object."""
        kwargs = {'sample_rate': self.sample_rate,
                  'center_freq': self.center_freq,
                  'bandwidth': self.bandwidth}

        offset = self.tell() if offset is None else self.seek(offset)
        kwargs['start_time'] = self.time

        z = self.read_data(N, offset).transpose(0, 2, 1)
        if not self.sideband:
            z = np.flip(z, axis=1)

        return FullStokesSignal(z, **kwargs)
