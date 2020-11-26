"""Base module for reader classes."""

import operator
from abc import ABCMeta, abstractmethod
import astropy.units as u
from astropy.time import Time
from ..core import Signal

__all__ = ['AbstractReader', 'OutOfBoundsError']


class OutOfBoundsError(EOFError):
    """Raised when stream position is out of bounds."""
    pass


class AbstractReader(metaclass=ABCMeta):
    """Abstract base class for readers.

    Readers are used to read data into `~Signal` objects. Subclasses
    must necessarily define:
      * `__len__()` -- Returns length of stream in number of samples.
      * `sample_rate` -- property, number of samples per unit time.
      * `start_time` -- property, timestamp at start of stream.
      * `_read_stream(n)` -- Reads `n` samples and returns an array-like.
    """
    def __init__(self):
        self.offset = 0

    @abstractmethod
    def __len__(self):
        """Length of stream in number of samples."""
        pass  # pragma: no cover

    @property
    @abstractmethod
    def sample_rate(self):
        """Sample rate (number of samples per unit time)."""
        pass  # pragma: no cover

    @property
    @abstractmethod
    def start_time(self):
        """Timestamp at first sample."""
        pass  # pragma: no cover

    @property
    def stop_time(self):
        """Timestamp after last sample."""
        stop = self.start_time + (len(self) / self.sample_rate)
        return Time(stop, format='isot', precision=9)

    @property
    def time(self):
        """Timestamp at current read position."""
        curr = self.start_time + (self.tell() / self.sample_rate)
        return Time(curr, format='isot', precision=9)

    def seek(self, offset, whence=0):
        """Change read position to the given offset.

        Offset is interpreted relative to position indicated by `whence`:
          * `0` or `'start'` -- Start of the stream (default). `offset`
            must be positive.
          * `1` or `'current'` -- Current stream position. `offset` must
            be positive or negative.
          * `2` or `'end'` -- End of the stream. `offset` must be negative.

        Parameters
        ----------
        offset : int, `~astropy.units.Quantity`, or `~astropy.time.Time`
            Offset to move to. Can be an `int` (number of samples),
            an astropy Quantity in time units, or an absolute time as an
            astropy Time object (`whence` has no effect in this case).
            For the latter two, the seek position is moved to the
            nearest integer sample.
        whence : {0, 1, 2, 'start', 'current', or 'end'}, optional
            Position that `offset` is taken relative to. Default is `0`,
            which is the start of the stream.

        Returns
        -------
        offset : int
            The new absolute position (in number of samples).
        """
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

        if whence in {0, 'start'}:
            self.offset = offset
        elif whence in {1, 'current'}:
            self.offset = self.offset + offset
        elif whence in {2, 'end'}:
            self.offset = len(self) + offset
        else:
            raise ValueError("Invalid 'whence'. Should be 0 or 'start', 1 or "
                             "'current', or 2 or 'end'.")

        if self.offset < 0 or self.offset > len(self):
            raise OutOfBoundsError("Cannot seek beyond bounds of stream!")
        return self.offset

    def tell(self, unit=None):
        """Return current stream position.

        Parameters
        ----------
        unit : `~astropy.units.Unit`, optional
            Time unit the offset should be returned in. By default, the
            current stream position is returned as an integer.

        Returns
        -------
        offset : int or `~astropy.units.Quantity`
            Current stream position.
        """
        if unit is None:
            return self.offset

        return (self.offset / self.sample_rate).to(unit)

    @abstractmethod
    def _read_stream(self, n, /, **kwargs):
        """Read N samples from current stream position into array-like."""
        pass  # pragma: no cover

    def _to_signal(self, z, /, start_time):
        """Return Signal containing given data."""
        return Signal(z, sample_rate=self.sample_rate, start_time=start_time)

    def read(self, n, /, **kwargs):
        """Read N samples from current stream position into Signal."""
        curr_time = self.time

        if self.tell() + n > len(self):
            raise OutOfBoundsError("Cannot read beyond end of stream")

        z = self._read_stream(n, **kwargs)
        _ = self.seek(n, whence=1)
        return self._to_signal(z, curr_time)
