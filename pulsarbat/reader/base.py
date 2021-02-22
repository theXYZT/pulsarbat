"""Base module for readers.

Readers are used to read data into `~Signal` objects. All readers
should expose the following properties:
  * `dtype` -- Data-type of signal data.
  * `shape` -- Shape of data.
  * `ndim` -- Number of dimensions in data.
  * `sample_shape` -- Shape of a sample.
  * `sample_rate` -- Number of samples per unit time.
  * `dt` -- Sample spacing in time units.
  * `time_length` -- Length of signal data in time units.
  * `start_time` -- Timestamp at first sample (or None).
  * `stop_time` -- Timestamp after last sample (or None).
  * `time` -- Timestamp at current read position (or None).

All readers should also expose the following methods:
  * `__len__()` -- Returns length of stream in number of samples.
  * `seek(n)` -- Change read position to the given offset.
  * `tell()` -- Returns current stream position.
  * `read(n, **kwargs)` -- Read `n` samples from current read position.
  * `dask_read(n, **kwargs)` -- Alias for `read(n, use_dask=True, **kwargs)`.

The following keyword arguments are assumed to be universally accepted
by the `read()` for all readers:
  * `use_dask` -- boolean, whether to use Dask arrays.
"""

import operator
import numpy as np
import astropy.units as u
from astropy.time import Time
import pulsarbat as pb

__all__ = [
    'BaseReader',
    'ConcatenatedReader',
]


class OutOfBoundsError(EOFError):
    """Raised when stream position is out of bounds."""
    pass


class BaseReader:
    """Base class for readers.

    Subclasses must either implement the `_read_array()` method if using
    the default `read()` implementation and structure.

    Parameters
    ----------
    shape : tuple of ints
        Signal shape.
    dtype : data-type
        Data-type of signal data.
    signal_type : class, optional
        Type of signal that will be returned by `read()`. Default is
        `Signal`. Accepted values are subclasses of `Signal`.
    sample_rate : :py:class:`astropy.units.Quantity`
        The number of samples per second. Must be in units of frequency.
    start_time : :py:class:`astropy.time.Time`, optional
        Timestamp at first sample of signal data. Default is None.
    **signal_kwargs
        Additional `kwargs` to pass on to `signal_type` when creating a
        Signal object.
    """
    def __init__(self, /, *, shape, dtype, signal_type=pb.Signal,
                 sample_rate, start_time=None, **signal_kwargs):

        if not issubclass(signal_type, pb.Signal):
            raise ValueError("Bad signal_type. Must be Signal or subclass.")

        self._signal_type = signal_type
        self._signal_kwargs = signal_kwargs

        self._dtype = np.dtype(dtype)
        self._shape = tuple(operator.index(a) for a in shape)

        if not self.ndim:
            raise ValueError("Invalid shape.")

        self.sample_rate = sample_rate
        self.start_time = start_time
        self.offset = 0

        # Read 0 samples now to catch potential errors earlier
        z = self.read(0)

        # Make sure shape and dtype are consistent
        if z.shape != (0,) + self.sample_shape:
            raise ValueError("Provided shape does not match output shape!")
        if z.dtype != self.dtype:
            raise ValueError("Provided dtype does not match output dtype!")

    def _attr_repr(self):
        st = 'N/A' if self.start_time is None else self.start_time.isot
        return (f"Start time: {st}\n"
                f"Sample rate: {self.sample_rate}\n"
                f"Time length: {self.time_length}\n"
                f"Offset: {self.offset}\n")

    def __str__(self):
        signature = f"{self.__class__.__name__} @ {hex(id(self))}"
        info = f"shape={self.shape}, dtype={self.dtype}"
        s = f"{signature}\n{'-' * len(signature)}\n"
        s += f"Data Container: {self._signal_type.__name__}<{info}>\n"
        s += self._attr_repr()
        return s.strip()

    def __repr__(self):
        info = f"shape={self.shape}, dtype={self.dtype}"
        sig = f"{self._signal_type.__name__}({info})"
        s = f"{self.__class__.__name__}<{sig}> @ {hex(id(self))}"
        return s

    def __len__(self):
        """Length of signal data in number of samples."""
        return self.shape[0]

    @property
    def shape(self):
        """Shape of data."""
        return self._shape

    @property
    def sample_shape(self):
        """Shape of a sample."""
        return self.shape[1:]

    @property
    def ndim(self):
        """Number of dimensions in data."""
        return len(self.shape)

    @property
    def dtype(self):
        """Data-type of data."""
        return self._dtype

    @property
    def sample_rate(self):
        """Sample rate of the signal data."""
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, sample_rate):
        try:
            temp = sample_rate.to(u.MHz)
            assert temp.isscalar and temp > 0
        except Exception:
            raise ValueError("Invalid sample_rate. Must be a positive scalar "
                             "Quantity with units of Hz or equivalent.")
        else:
            self._sample_rate = sample_rate

    @property
    def start_time(self):
        """Timestamp at first sample."""
        return self._start_time

    @start_time.setter
    def start_time(self, start_time):
        try:
            temp = None
            if start_time is not None:
                temp = Time(start_time, format='isot', precision=9)
                assert temp.isscalar
        except Exception:
            raise ValueError("Invalid start_time. Must be a scalar astropy "
                             "Time object.")
        else:
            self._start_time = temp

    @property
    def stop_time(self):
        """Timestamp after last sample."""
        if self.start_time is None:
            return None
        return self.start_time + (len(self) / self.sample_rate)

    @property
    def dt(self):
        """Sample spacing (1 / sample_rate)."""
        return (1 / self.sample_rate).to(u.s)

    @property
    def time_length(self):
        """Length of signal in time units."""
        return (len(self) / self.sample_rate).to(u.s)

    @property
    def time(self):
        """Timestamp at current read position."""
        if self.start_time is None:
            return None
        return self.start_time + (self.tell() / self.sample_rate)

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
            new_offset = offset
        elif whence in {1, 'current'}:
            new_offset = self.offset + offset
        elif whence in {2, 'end'}:
            new_offset = len(self) + offset
        else:
            raise ValueError("Invalid 'whence'. Should be 0 or 'start', 1 or "
                             "'current', or 2 or 'end'.")

        if new_offset < 0 or new_offset > len(self):
            raise OutOfBoundsError("Cannot seek beyond bounds of stream!")

        self.offset = new_offset
        return self.offset

    def tell(self, unit=None):
        """Returns current stream position.

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

    def _read_array(self, n, offset, /):
        """Read n samples from given offset into numpy array."""
        raise NotImplementedError("Must be implemented by subclasses "
                                  "if using the default read methods.")

    def _read_data(self, n, /, use_dask=False, **kwargs):
        """Read n samples from current read position into array-like."""
        if use_dask:
            import dask
            import dask.array as da

            delayed_read = dask.delayed(self._read_array, pure=True)

            _out_shape = (n,) + self.sample_shape
            z = da.from_delayed(delayed_read(n, self.offset, **kwargs),
                                dtype=self.dtype, shape=_out_shape)

            chunks = (-1, ) + ('auto', ) * len(self.sample_shape)
            z = z.rechunk(chunks)
        else:
            z = self._read_array(n, self.offset, **kwargs)

        return z

    def _verify_n(self, n):
        """Verify that n given to .read() is meaningful."""
        if (n := operator.index(n)) < 0:
            raise ValueError("n must be non-negative.")

        if self.tell() + n > len(self):
            raise OutOfBoundsError("Cannot read beyond end of stream")

        return n

    def read(self, n, /, **kwargs):
        """Read `n` samples from current read position.

        Parameters
        ----------
        n : int
            Number of samples to read. Must be non-negative.
        **kwargs
            Additional keyword arguments. Currently supported are:
              * `use_dask` -- Whether to use dask arrays.

        Returns
        -------
        out : `~Signal` object or subclass
            Signal of length `n` containing data that was read.
        """
        n = self._verify_n(n)

        z = self._signal_type(self._read_data(n, **kwargs),
                              sample_rate=self.sample_rate,
                              start_time=self.time,
                              **self._signal_kwargs)

        self.seek(n, whence=1)
        return z

    def dask_read(self, n, /, **kwargs):
        """Read `n` samples from current read position using dask arrays.

        A convenience method equivalent to `.read(n, use_dask=True, **kwargs).

        Parameters
        ----------
        n : int
            Number of samples to read.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        out : `~Signal` object or subclass
            Signal of length `n` containing data that was read.
        """
        return self.read(n, use_dask=True, **kwargs)


class ConcatenatedReader(BaseReader):
    def __init__(self, readers, axis):
        if not all(isinstance(r, BaseReader) for r in readers):
            raise TypeError("Not all objects in readers are readers!")

        if axis in {0, 'time'}:
            raise ValueError("Can't concatenate readers in time!")
        self._axis = axis

        for r in readers:
            r.seek(0)
        self._readers = tuple(readers)

        z = pb.concatenate([r.read(0) for r in self._readers],
                           axis=self._axis)

        min_length = min(len(r) for r in readers)
        super().__init__(shape=(min_length,) + z.sample_shape, dtype=z.dtype,
                         signal_type=type(z), sample_rate=z.sample_rate,
                         start_time=z.start_time)

    def read(self, n, /, **kwargs):
        """Read `n` samples from current read position.

        Parameters
        ----------
        n : int
            Number of samples to read.
        **kwargs
            Additional keyword arguments. Currently supported are:
              * `use_dask` -- Whether to use dask arrays.

        Returns
        -------
        out : `~Signal` object or subclass
            Signal of length `n` containing data that was read.
        """
        n = self._verify_n(n)

        for r in self._readers:
            r.seek(self.offset)

        z = pb.concatenate([r.read(n, **kwargs) for r in self._readers],
                           axis=self._axis)

        self.seek(n, whence=1)
        return z
