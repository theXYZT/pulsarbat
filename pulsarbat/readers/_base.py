"""Base module for readers.

Readers are used to read data into `~Signal` objects. A reader `r`
should expose the following user-facing attributes/methods:

  * `len(r)` -- Length of data in number of samples.
  * `r.shape` -- Shape of data.
  * `r.sample_shape` -- Shape of a sample.
  * `r.ndim` -- Number of dimensions.
  * `r.dtype` -- Data-type of signal data.
  * `r.sample_rate` -- Number of samples per unit time.
  * `r.dt` -- Sample spacing in time units.
  * `r.time_length` -- Length of signal data in time units.
  * `r.start_time` -- Timestamp at first sample (or None).
  * `r.stop_time` -- Timestamp after last sample (or None).
  * `r.read(offset, n, **kwargs)` -- Read `n` samples from given offset.
  * `r.offset_at(t)` -- Offset at given timestamp or Quantity.
  * `r.time_at(offset)` -- Timestamp at given offset (or None).
  * `r.contains(t)` -- Whether time(s) are within the bounds of the data.
  * `t in r` -- Whether times is within the bounds of the data.

The following keyword arguments should be accepted the `read()` method:
  * `use_dask` -- boolean, whether to use Dask arrays.
  * `chunks` -- Chunk sizes if using dask arrays.
"""

import operator
import numpy as np
import astropy.units as u
from astropy.time import Time
import pulsarbat as pb

__all__ = [
    "BaseReader",
]


class OutOfBoundsError(EOFError):
    """Raised when stream position is out of bounds."""

    pass


class BaseReader:
    """Base class for readers.

    Subclasses must either implement the ``_read_array()`` method if using
    the default ``.read()`` implementation, or implement their own
    ``.read()`` method.

    Parameters
    ----------
    shape : tuple of int
        Signal shape.
    dtype : dtype
        Data-type of signal data.
    signal_type : subclass of .Signal, default: .Signal
        Type of signal that will be returned by ``read()``.
    sample_rate : Quantity
        The number of samples per second. Must be in units of frequency.
    start_time : Time, optional
        Timestamp at first sample of signal data. Default is None.
    **signal_kwargs
        Additional kwargs to pass on to ``signal_type`` when creating a
        :py:class:`.Signal` object.
    """

    def __init__(
        self,
        /,
        *,
        shape,
        dtype,
        signal_type=pb.Signal,
        sample_rate,
        start_time=None,
        **signal_kwargs,
    ):

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

        # Read 0 samples now to catch potential errors earlier
        z = self.read(0, 0)

        # Make sure shape and dtype are consistent
        if z.shape != (0,) + self.sample_shape:
            raise ValueError("Provided shape does not match output shape!")

        if z.dtype != self.dtype:
            raise ValueError("Provided dtype does not match output dtype!")

    def _attr_repr(self):
        st = "N/A" if self.start_time is None else self.start_time.isot
        return (
            f"Start time: {st}\n"
            f"Sample rate: {self.sample_rate}\n"
            f"Time length: {self.time_length}\n"
        )

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

    def __getattr__(self, name):
        if name in self._signal_kwargs:
            return self._signal_kwargs[name]
        return super().__getattribute__(name)

    def __dir__(self):
        members = set(object.__dir__(self))
        members.update(self._signal_kwargs)
        return sorted(members)

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
            temp = sample_rate.to(u.Hz)
            assert temp.isscalar and temp > 0
        except Exception:
            raise ValueError(
                "Invalid sample_rate. Must be a positive scalar "
                "Quantity with units of Hz or equivalent."
            )
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
                temp = Time(start_time, format="isot", precision=9)
                assert temp.isscalar
        except Exception:
            raise ValueError(
                "Invalid start_time. Must be a scalar astropy " "Time object."
            )
        else:
            self._start_time = temp

    @property
    def stop_time(self):
        """Timestamp after last sample."""
        return self.time_at(len(self))

    @property
    def dt(self):
        """Sample spacing (1 / sample_rate)."""
        return (1 / self.sample_rate).to(u.s)

    @property
    def time_length(self):
        """Length of signal in time units."""
        return (len(self) / self.sample_rate).to(u.s)

    def contains(self, t, /):
        """Whether time(s) are within the bounds of the signal."""
        if self.start_time is None:
            return np.zeros(t.shape, bool) if t.shape else False

        t0, t1 = self.start_time, self.stop_time
        edge = ~Time.isclose(t, t1) | Time.isclose(t, t0)
        return edge & (t0 <= t) & (t < t1)

    def __contains__(self, t):
        """Whether time is within the bounds of the signal."""
        return self.contains(t)

    def offset_at(self, t, /):
        """Returns nearest integer offset at given time.

        Parameters
        ----------
        t : Quantity or Time
            Can be an absolute time as an astropy Time object, or an
            astropy Quantity in time units relative to the start.

        Returns
        -------
        offset : int
            The nearest integer position (in number of samples).
        """
        try:
            t = t - self.start_time
        except Exception:
            pass

        offset = int((t * self.sample_rate).to(u.one).round())

        if offset < 0 or offset > len(self):
            raise OutOfBoundsError("Given time is out of bounds!")

        return offset

    def time_at(self, offset, /, unit=None):
        """Returns time at given offset.

        Parameters
        ----------
        offset : int
            Position in number of samples.
        unit : Unit, optional
            Desired unit of returned value (as an astropy unit).
            By default (None), the absolute timestamp is returned.

        Returns
        -------
        t : Quantity or Time
            Time relative to the start as an astropy Quantity if `unit`
            is provided, otherwise an absolute time as an astropy Time
            object.
        """
        if unit is not None:
            return (offset / self.sample_rate).to(unit)

        if self.start_time is None:
            return None

        return self.start_time + (offset / self.sample_rate)

    def _read_array(self, offset, n, /):
        """Read n samples from given offset into numpy array."""
        return NotImplemented

    def _read_data(self, offset, n, /, use_dask=False, **kwargs):
        """Read n samples from current read position into array-like."""
        if use_dask:
            import dask
            import dask.array as da

            delayed_read = dask.delayed(self._read_array, pure=True)

            _out_shape = (n,) + self.sample_shape
            z = da.from_delayed(delayed_read(offset, n, **kwargs),
                                dtype=self.dtype, shape=_out_shape)

            chunks = kwargs.get("chunks")
            if chunks is None:
                chunks = (-1,) + ("auto",) * len(self.sample_shape)
            z = z.rechunk(chunks)

        else:
            z = self._read_array(offset, n, **kwargs)

        return z

    def read(self, offset, n, /, **kwargs):
        """Read `n` samples from given offset.

        Parameters
        ----------
        offset : int
            Position to read from. Must be non-negative.
        n : int
            Number of samples to read. Must be non-negative.
        **kwargs
            Currently supported keyword arguments:

              * ``use_dask`` -- Whether to use dask arrays.
              * ``chunks`` -- Chunk sizes if using dask arrays. By default,
                there is no chunking along the zeroth dimension.

        Returns
        -------
        out : Signal
            Signal of length ``n`` containing data that was read.
        """
        if (offset := operator.index(offset)) < 0:
            raise ValueError("offset must be a non-negative int.")

        if (n := operator.index(n)) < 0:
            raise ValueError("n must be a non-negative int.")

        if offset + n > len(self):
            raise OutOfBoundsError("Cannot read beyond end of stream")

        return self._signal_type(
            self._read_data(offset, n, **kwargs),
            sample_rate=self.sample_rate,
            start_time=self.time_at(offset),
            **self._signal_kwargs,
        )

    def dask_read(self, offset, n, /, **kwargs):
        """Read `n` samples from given offset using Dask arrays.

        A convenience method equivalent to the :py:meth:`.read()` method with
        ``use_dask=True``.

        Parameters
        ----------
        offset : int
            Position to read from. Must be non-negative.
        n : int
            Number of samples to read. Must be non-negative.
        **kwargs
            Additional keyword arguments. Currently supported:
              * ``chunks`` -- Chunk sizes if using dask arrays.
                              By default, there is no chunking along
                              the zeroth dimension.

        Returns
        -------
        out : Signal
            Signal of length ``n`` containing data that was read as a
            Dask array.
        """
        return self.read(offset, n, use_dask=True, **kwargs)
