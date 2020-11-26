"""Baseband reader classes."""

import baseband
import numpy as np
import astropy.units as u
from astropy.time import Time

from .base import AbstractReader
from ..core import (BasebandSignal, DualPolarizationSignal,
                    FullStokesSignal)
from ..utils import real_to_complex

__all__ = ['BasebandReader', 'BasebandRawReader', 'GUPPIRawReader',
           'DADAStokesReader']


class BasebandReader(AbstractReader):
    """Wrapper around StreamReader from the `~baseband` package.

    Parameters
    ----------
    name, **kwargs
        Arguments to pass on to `~baseband.open` to create a StreamReader
        object via `baseband.open(name, 'rs', **kwargs)`.
    """
    def __init__(self, name, /, **kwargs):
        super().__init__()
        self._name = name
        self._kwargs = kwargs
        with self._get_fh() as fh:
            self._info = fh.info
            self._dtype = fh.dtype

    def _get_fh(self):
        return baseband.open(self._name, 'rs', **self._kwargs)

    def __len__(self):
        return self._info.shape[0]

    @property
    def sample_shape(self):
        return self._info.shape[1:]

    @property
    def dtype(self):
        return self._dtype

    @property
    def sample_rate(self):
        """Sample rate of the data."""
        return self._info.sample_rate

    @property
    def start_time(self):
        """Time at first sample of data."""
        return Time(self._info.start_time, format='isot', precision=9)

    def _read_array(self, n, offset):
        """Read n samples from current stream position into numpy array."""
        with self._get_fh() as fh:
            fh.seek(offset)
            z = fh.read(n)
        return z.astype(self.dtype, copy=False)

    def _read_stream(self, n, /, **kwargs):
        """Read N samples from current stream position into array-like."""
        return self._read_array(n, self.offset)


class BasebandRawReader(BasebandReader):
    """Base class for raw voltage data readable by the `~baseband` package.

    Parameters
    ----------
    name, **kwargs
        Arguments to pass on to `~baseband.open` to create a StreamReader
        object via `baseband.open(name, 'rs', **kwargs)`.
    center_freq : `~astropy.units.Quantity`
        The frequency at the center of the signal's band. Must be in
        units of frequency.
    chan_bw : `~astropy.units.Quantity`
        The bandwidth of a channel. The total bandwidth is `chan_bw *
        nchan`. Must be in units of frequency.
    freq_align : {'bottom', 'center', 'top'}, optional
        The alignment of channels relative to the `center_freq`. Default
        is `'center'` (as with odd-length complex DFTs). `'bottom'` and
        `'top'` only have an effect when `nchan` is even.
    sideband : bool or array-like
        Can be True (no spectral flip) or False (spectrally flipped). If
        channels have different sidebands, an array-like object of
        boolean elements can be passed (must have the same shape as the
        StreamReader's `sample_shape`).
    """
    def __init__(self, name, /, *, center_freq, freq_align='center',
                 sideband=True, **kwargs):
        super().__init__(name, **kwargs)

        try:
            self._center_freq = center_freq.to(u.MHz)
            assert self._center_freq.isscalar
        except Exception:
            err = ("Invalid center_freq. Must be a scalar "
                   "Quantity with units of Hz or equivalent.")
            raise ValueError(err)

        if type(sideband) is bool:
            self._sideband = sideband
        else:
            self._sideband = np.array(sideband).astype(bool)
            if self.sample_shape != self.sideband.shape:
                err = "StreamReader sample shape != sideband shape"
                raise ValueError(err)

        if freq_align in ['bottom', 'center', 'top']:
            self._freq_align = freq_align
        else:
            choices = "{'bottom', 'center', 'top'}"
            raise ValueError(f'Invalid freq_align. Expected: {choices}')

    @property
    def dtype(self):
        return np.complex64

    @property
    def sideband(self):
        """True if upper sideband, False if lower sideband."""
        return self._sideband

    @property
    def center_freq(self):
        """Center frequency."""
        return self._center_freq

    @property
    def freq_align(self):
        """Alignment of channel frequencies."""
        return self._freq_align

    @property
    def sample_rate(self):
        """Sample rate (number of samples per unit time)."""
        if self._info.complex_data:
            return self._info.sample_rate
        else:
            return self._info.sample_rate / 2

    def __len__(self):
        if self._info.complex_data:
            return self._info.shape[0]
        else:
            return self._info.shape[0] // 2

    def _read_array(self, n, offset):
        """Read n samples from current stream position into numpy array."""
        with self._get_fh() as fh:
            if self._info.complex_data:
                fh.seek(offset)
                z = fh.read(n)
            else:
                fh.seek(offset * 2)
                z = real_to_complex(fh.read(2*n), axis=0)

        if self.sideband is False:
            z = z.conj()
        elif self.sideband is not True:
            slc = ~self.sideband
            z[:, slc] = z[:, slc].conj()

        return z.astype(self.dtype, copy=False)

    def _to_signal(self, z, /, start_time):
        """Return Signal containing given data."""
        kwargs = {'sample_rate': self.sample_rate,
                  'center_freq': self.center_freq,
                  'freq_align': self.freq_align}
        return BasebandSignal(z, start_time=start_time, **kwargs)


class GUPPIRawReader(BasebandRawReader):
    """Baseband reader for GUPPI raw voltage data format.

    Parameters
    ----------
    name
        File name, filehandle, or sequence of file names to pass on to
        `~baseband.open` to create a GUPPIStreamReader object via
        `baseband.open(name, 'rs', format='guppi')`.
    """

    def __init__(self, name, /):
        kwargs = {'format': 'guppi'}

        with baseband.open(name, 'rs', **kwargs) as fh:
            header = fh.header0

        pol_dict = {'LIN': 'linear', 'CIRC': 'circular'}
        self._pol_type = pol_dict[header['FD_POLN']]

        super().__init__(name, sideband=header.sideband, freq_align='center',
                         center_freq=header['OBSFREQ'] * u.MHz, **kwargs)

    @property
    def pol_type(self):
        return self._pol_type

    def _to_signal(self, z, /, start_time):
        """Return Signal containing given data."""
        kwargs = {'sample_rate': self.sample_rate,
                  'center_freq': self.center_freq,
                  'freq_align': self.freq_align,
                  'pol_type': self.pol_type,
                  'start_time': start_time}
        return DualPolarizationSignal(z.transpose(0, 2, 1), **kwargs)


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
        kwargs = {'format': 'dada'}
        super().__init__(name, **kwargs)

        with self._get_fh() as fh:
            if not (fh.header0['NPOL'] == 4 and fh.header0['NDIM'] == 1):
                raise ValueError('Does not look like Stokes data')
            self._header = fh.header0

    @property
    def sideband(self):
        return self._header.sideband

    @property
    def center_freq(self):
        return self._header['FREQ'] * u.MHz

    @property
    def chan_bw(self):
        return abs(self._header['BW'] / self._header['NCHAN']) * u.MHz

    @property
    def freq_align(self):
        return 'bottom' if self.sideband else 'top'

    def _to_signal(self, z, /, start_time):
        """Return Signal containing given data."""
        kwargs = {'sample_rate': self.sample_rate,
                  'center_freq': self.center_freq,
                  'chan_bw': self.chan_bw,
                  'freq_align': self.freq_align,
                  'start_time': start_time}

        z = z.transpose(0, 2, 1)
        if not self.sideband:
            z = np.flip(z, axis=1)

        return FullStokesSignal(z, **kwargs)
