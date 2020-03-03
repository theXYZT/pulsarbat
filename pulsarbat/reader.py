"""Observation management."""

from abc import ABC, abstractmethod
import numpy as np
import astropy.units as u
from astropy.time import Time
from .core import BasebandSignal
from .utils import verify_scalar_quantity, real_to_complex

__all__ = ['AbstractReader', 'BasebandReader']


class AbstractReader(ABC):
    """Abstract base class for data readers.

    Readers are used to read in data into standardized `~pulsarbat.Signal`
    objects. All Reader classes must define their own `read()`, `seek()`,
    and `tell()` methods.
    """
    @abstractmethod
    def read(self, n):
        """Reads n time samples of data."""
        pass

    @abstractmethod
    def seek(self, n, whence=0):
        """Change read position to the given sample offset."""
        pass

    @abstractmethod
    def tell(self):
        """Return current read position."""
        pass


class BasebandReader(AbstractReader):
    """Base class for data readable by the `~baseband` package."""
    def __init__(self, file_handle):
        self._fh = file_handle

        if verify_scalar_quantity(self._fh.sample_rate, u.Hz):
            self._sample_rate = self._fh.sample_rate.to(u.Hz)

        self._start_time = Time(self._fh.start_time, format='isot',
                                precision=9)
        self._stop_time = Time(self._fh.stop_time, format='isot',
                               precision=9)

    @property
    def sample_rate(self):
        """Sample rate of the data."""
        return self._sample_rate.copy()

    @property
    def start_time(self):
        """Time at first sample of data."""
        return self._start_time.copy()

    @property
    def stop_time(self):
        """Time at the end of data (time just after the last sample)."""
        return self._stop_time.copy()

    def _convert_to_signal(z, sample_rate, start_time):
        raise NotImplementedError(":)")

    def read(self, N: int):
        """Read N time samples of complex baseband data."""
        if type(N) is not int or N < 1:
            raise ValueError("N must be a positive integer.")

        start_time = self.tell(unit='time')

        if self._fh.complex_data:
            shape = (N, ) + self._fh.sample_shape
            z = np.empty(shape, dtype=np.complex64, order='F')
            self._fh.read(out=z)
            sample_rate = self.sample_rate
        else:
            shape = (2 * N, ) + self._fh.sample_shape
            z = np.empty(shape, dtype=np.float32, order='F')
            self._fh.read(out=z)
            z = real_to_complex(z)
            sample_rate = self.sample_rate / 2

        return self._convert_to_signal(z, sample_rate, start_time)

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


class PUPPIRawReader(BasebandReader):
    def __init__(self, file_handle):
        super().__init__(file_handle)
        self._header = self.fh.header0
        assert self.sample_rate == self._header['CHAN_BW'] * u.MHz

        self.center_freq = self._header['OBSFREQ'] * u.MHz
        self.bandwidth = self.sample_rate * self._header['OBSNCHAN']

    def read(self, num_samples, timestamp=None):
        if timestamp is not None:
            assert self.start_time <= timestamp
            assert timestamp + num_samples / self.sample_rate < self.stop_time
            self.fh.seek(timestamp)

        read_start_time = self.fh.tell('time')

        z = self.fh.read(num_samples)
        return BasebandSignal(z.transpose(0, 2, 1),
                              sample_rate=self.sample_rate,
                              start_time=read_start_time,
                              center_freq=self.center_freq,
                              bandwidth=self.bandwidth)
