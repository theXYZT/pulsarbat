"""Baseband reader classes."""

import numpy as np
import astropy.units as u
from astropy.time import Time
from contextlib import nullcontext
import baseband
import pulsarbat as pb
from pulsarbat.readers import BaseReader

__all__ = [
    "BasebandReader",
    "GUPPIRawReader",
    "DADAStokesReader",
]


class BasebandReader(BaseReader):
    """Wrapper around StreamReader from the `~baseband` package.

    Parameters
    ----------
    name, **kwargs
        Arguments to pass on to `~baseband.open` to create a StreamReader
        object via `baseband.open(name, 'rs', **kwargs)`.
    signal_type : class, optional
        Type of signal that will be returned by `read()`. Default is
        `Signal`. Accepted values are subclasses of `Signal`.
    signal_kwargs : dict, optional
        Additional `kwargs` to pass on to `signal_type` when creating a
        Signal object. Must not include `sample_rate` or `start_time` as
        dictionary fields.
    intensity : bool, optional
        Whether the data is intensity data. If `signal_type` is a
        subclass of `pb.IntensitySignal`, assumed to be True. If
        `signal_type` is a subclass of `pb.BasebandSignal`, assumed to be
        False. Default is False.
    lower_sideband : bool or array-like, optional
        Whether the data is lower-sideband (LSB) data. Default is False.
        If not a boolean, must be an array-like of booleans with the
        same shape as `sample_shape` of original data as read by
        `StreamReader` in the `baseband` package.
    """

    def __init__(
        self,
        name,
        /,
        *,
        signal_type=pb.Signal,
        signal_kwargs=dict(),
        intensity=None,
        lower_sideband=False,
        **kwargs,
    ):

        self._name = name
        self._kwargs = kwargs

        if intensity is None:
            self._intensity = issubclass(signal_type, pb.IntensitySignal)
        else:
            self._intensity = bool(intensity)

            if issubclass(signal_type, pb.BasebandSignal) and self.intensity:
                raise ValueError("intensity must be False when using pb.BasebandSignal")

            if issubclass(signal_type, pb.IntensitySignal) and not self.intensity:
                raise ValueError("intensity must be True when using pb.IntensitySignal")

        with self._get_fh() as fh:
            self._complex_data = bool(fh.complex_data)
            self._in_sample_shape = fh.shape[1:]

            if self.intensity and self.complex_data:
                raise ValueError("Intensity data cannot be complex-valued!")

            if self.real_baseband:
                _sr = (fh.sample_rate / 2).to(u.MHz)
                _length = fh.shape[0] // 2
                _dtype = np.complex64
            else:
                _sr = fh.sample_rate.to(u.MHz)
                _length = fh.shape[0]
                _dtype = np.complex64 if self.complex_data else np.float32

            _t0 = Time(fh.start_time, format="isot", precision=9)

        self.lower_sideband = lower_sideband
        self._dtype = np.dtype(_dtype)

        # Determine sample shape by reading a dummy array
        _shape = (_length,) + self._read_array(0, 0).shape[1:]

        super().__init__(
            shape=_shape,
            dtype=_dtype,
            signal_type=signal_type,
            sample_rate=_sr,
            start_time=_t0,
            **signal_kwargs,
        )

    def _get_fh(self):
        return baseband.open(self._name, "rs", **self._kwargs)

    @property
    def complex_data(self):
        """Whether the data is complex-valued."""
        return self._complex_data

    @property
    def intensity(self):
        """Where the data is intensity data (as opposed to raw baseband)."""
        return self._intensity

    @property
    def real_baseband(self):
        """Whether the data is real-valued baseband data."""
        return not (self.intensity or self.complex_data)

    @property
    def lower_sideband(self):
        """Whether data is lower sideband (LSB) data."""
        return self._lower_sideband

    @lower_sideband.setter
    def lower_sideband(self, s):
        if type(s) != bool:
            s = np.array(s).astype(bool)
            if s.shape != self._in_sample_shape:
                err = f"Got {s.shape}, expected {self._in_sample_shape}"
                raise ValueError(f"Invalid lower_sideband shape. {err}")

        self._lower_sideband = s

    def _read_baseband(self, offset, n, /, lock=nullcontext(), **kwargs):
        """Read n samples from given offset using baseband."""
        with lock:
            with self._get_fh() as fh:
                if self.real_baseband:
                    fh.seek(2 * offset)
                    z = pb.utils.real_to_complex(fh.read(2 * n), axis=0)
                else:
                    fh.seek(offset)
                    z = fh.read(n)

        if not self.intensity:
            if self.lower_sideband is True:
                z = z.conj()
            elif self.lower_sideband is not False:
                z[:, self.lower_sideband] = z[:, self.lower_sideband].conj()

        return z.astype(self.dtype, copy=False)

    def _read_array(self, offset, n, /, **kwargs):
        """Read n samples from given offset into numpy array.

        Post-processing for specific formats (such as correcting the
        order of data dimensions) should be done in this method by
        subclasses.
        """
        return self._read_baseband(offset, n, **kwargs)

    def read(self, offset, n, /, **kwargs):
        """Read `n` samples from given offset.

        Parameters
        ----------
        offset : int
            Position to read from. Must be non-negative.
        n : int
            Number of samples to read. Must be non-negative.
        **kwargs
            Additional keyword arguments. Currently supported are:
              * `use_dask` -- Whether to use dask arrays.
              * `lock` -- A lock object to prevent concurrent reads.
                          Must be a context manager.

        Returns
        -------
        out : `~Signal` object or subclass
            Signal of length `n` containing data that was read.
        """
        return super().read(offset, n, **kwargs)


class GUPPIRawReader(BasebandReader):
    """Baseband reader for GUPPI raw voltage data format.

    Parameters
    ----------
    name
        File name, filehandle, or sequence of file names to pass on to
        `~baseband.open` to create a GUPPIStreamReader object via
        `baseband.open(name, 'rs', format='guppi', squeeze=False)`.
    """

    def __init__(self, name, /):
        kwargs = {"format": "guppi", "squeeze": False}

        with baseband.open(name, "rs", **kwargs) as fh:
            header = fh.header0
            obsfreq = header["OBSFREQ"] * u.MHz
            pol = {"LIN": "linear", "CIRC": "circular"}[header["FD_POLN"]]

        signal_kwargs = {
            "center_freq": obsfreq,
            "freq_align": "center",
            "pol_type": pol,
        }

        super().__init__(
            name,
            signal_type=pb.DualPolarizationSignal,
            signal_kwargs=signal_kwargs,
            lower_sideband=not header.sideband,
            **kwargs,
        )

    def _read_array(self, offset, n, /, **kwargs):
        """Read n samples from current read position into numpy array."""
        z = self._read_baseband(offset, n, **kwargs)
        return z.transpose(0, 2, 1)


class DADAStokesReader(BasebandReader):
    """Reader for full Stokes data in DADA format.

    Parameters
    ----------
    name
        File name, filehandle, or sequence of file names to pass on to
        `~baseband.open` to create a DADAStreamReader object via
        `baseband.open(name, 'rs', format='dada')`.
    """

    def __init__(self, name, /):
        kwargs = {"format": "dada", "squeeze": False}

        with baseband.open(name, "rs", **kwargs) as fh:
            header = fh.header0

            if not (header["NPOL"] == 4 and header["NDIM"] == 1):
                raise ValueError("Does not look like Full Stokes data")

            lsb = header["BW"] < 0
            freq = header["FREQ"] * u.MHz
            chan_bw = abs(header["BW"] / header["NCHAN"]) * u.MHz
            freq_align = "top" if lsb else "bottom"

        signal_kwargs = {
            "center_freq": freq,
            "chan_bw": chan_bw,
            "freq_align": freq_align,
        }

        super().__init__(
            name,
            signal_type=pb.FullStokesSignal,
            signal_kwargs=signal_kwargs,
            lower_sideband=lsb,
            **kwargs,
        )

    def _read_array(self, offset, n, /, **kwargs):
        """Read n samples from current read position into numpy array."""
        z = self._read_baseband(offset, n, **kwargs)

        if self.lower_sideband:
            z = np.flip(z, axis=-1)

        return z.transpose(0, 2, 1)
