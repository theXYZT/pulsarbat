"""Tests for `pulsarbat.RadioSignal` and subclasses."""

import pytest
import numpy as np
import dask.array as da
import astropy.units as u
from astropy.time import Time
import pulsarbat as pb


@pytest.mark.parametrize("fn", [np.ones, da.ones])
@pytest.mark.parametrize("fcen", [400 * u.MHz, 800 * u.MHz])
@pytest.mark.parametrize("chan_bw", [1 * u.MHz, 10 * u.MHz])
@pytest.mark.parametrize("nchan", [7, 8, 9])
@pytest.mark.parametrize("align", ["bottom", "top", "center"])
def test_radiosignal(fn, fcen, chan_bw, nchan, align):
    z = pb.RadioSignal(fn((16, nchan, 2)), sample_rate=1*u.Hz,
                       center_freq=fcen, chan_bw=chan_bw, freq_align=align)

    bw = chan_bw * nchan
    f = np.fft.fftshift(np.fft.fftfreq(nchan, 1/bw))
    if align == 'center' and nchan % 2 == 0:
        chan_cens = fcen + f + (bw/nchan/2)
    elif align == 'top' and nchan % 2 == 0:
        chan_cens = (fcen - f)[::-1]
    else:
        chan_cens = fcen + f

    assert isinstance(z, pb.RadioSignal)
    print(z)
    repr(z)

    assert z.nchan == nchan
    assert u.isclose(z.center_freq, fcen)
    assert u.isclose(z.bandwidth, bw)
    assert u.isclose(z.chan_bw, chan_bw)
    assert u.isclose(z.max_freq, fcen + bw/2)
    assert u.isclose(z.min_freq, fcen - bw/2)
    print(z.channel_freqs)
    print()
    print(chan_cens, align)
    assert all(u.isclose(z.channel_freqs, chan_cens))

    if nchan % 2:
        assert z.freq_align == 'center'
    else:
        assert z.freq_align == align


def test_shape_errors():
    for shape in [(10,), (24, 0), (4, 4, 0), (0, 5, 10)]:
        with pytest.raises(ValueError):
            _ = pb.RadioSignal(np.empty(shape), sample_rate=1*u.Hz,
                               center_freq=1*u.GHz, chan_bw=1*u.MHz)


def test_center_freq_errors():
    for x in [99., 4*u.m, [5, 6]*u.Hz]:
        with pytest.raises(ValueError):
            _ = pb.RadioSignal(np.empty((8, 4)), sample_rate=1*u.Hz,
                               center_freq=x, chan_bw=10*u.MHz)


def test_chan_bw_errors():
    for x in [99., 4*u.m, [5, 6]*u.Hz, -1*u.Hz]:
        with pytest.raises(ValueError):
            _ = pb.RadioSignal(np.empty((8, 4)), sample_rate=1*u.Hz,
                               center_freq=400*u.MHz, chan_bw=x)


def test_freq_align_errors():
    for x in [None, (), 'invalid']:
        with pytest.raises(ValueError):
            _ = pb.RadioSignal(np.empty((8, 4)), sample_rate=1*u.Hz,
                               center_freq=400*u.MHz, chan_bw=10*u.MHz,
                               freq_align=x)

    for x in ['top', 'center', 'bottom']:
        _ = pb.RadioSignal(np.empty((8, 4)), sample_rate=1*u.Hz,
                           center_freq=400*u.MHz, chan_bw=10*u.MHz,
                           freq_align=x)


@pytest.mark.parametrize("sr", [1*u.MHz, 5*u.MHz])
@pytest.mark.parametrize("nchan", [7, 8])
def test_basebandsignal(sr, nchan):
    x = np.ones((16, nchan)).astype(np.complex64)
    center_freq = 800 * u.MHz

    z = pb.BasebandSignal(x, sample_rate=sr, center_freq=center_freq)
    assert u.isclose(z.sample_rate, sr)
    assert u.isclose(z.chan_bw, sr)
    assert u.isclose(z.bandwidth, sr*nchan)


def test_baseband_dtype():
    fcen, sr = 800*u.MHz, 10*u.MHz

    for dtype in [np.float32, np.float64]:
        x = np.ones((16, 8), dtype=dtype)
        with pytest.raises(ValueError):
            _ = pb.BasebandSignal(x, sample_rate=sr, center_freq=fcen)

    for dtype in [np.complex64, np.complex128]:
        x = np.ones((16, 8), dtype=dtype)
        _ = pb.BasebandSignal(x, sample_rate=sr, center_freq=fcen)


def test_intensity_dtype():
    fcen, sr, cbw = 800*u.MHz, 1*u.Hz, 10*u.MHz

    for dtype in [np.complex64, np.complex128]:
        x = np.ones((16, 8), dtype=dtype)
        with pytest.raises(ValueError):
            _ = pb.IntensitySignal(x, sample_rate=sr, center_freq=fcen,
                                   chan_bw=cbw)

    for dtype in [np.float32, np.float64]:
        x = np.ones((16, 8), dtype=dtype)
        _ = pb.IntensitySignal(x, sample_rate=sr, center_freq=fcen,
                               chan_bw=cbw)


@pytest.mark.parametrize("use_dask", [True, False])
@pytest.mark.parametrize("A", [1, 2, 4, 10])
@pytest.mark.parametrize("in_dtype, out_dtype", [(np.complex64, np.float32),
                                                 (np.complex128, np.float64)])
def test_baseband_to_intensity(A, use_dask, in_dtype, out_dtype):
    shape = (1024, 8, 2)

    if use_dask:
        x = da.random.uniform(-np.pi, np.pi, shape)
    else:
        x = np.random.uniform(-np.pi, np.pi, shape)
    z_phasor = np.exp(1j * x).astype(in_dtype)

    st = Time.now()
    z = pb.BasebandSignal(A * z_phasor, sample_rate=1*u.MHz, start_time=st,
                          center_freq=600 * u.MHz)

    assert isinstance(z.data, type(x))

    zi = z.to_intensity()
    assert isinstance(zi.data, type(x))
    assert np.allclose(A**2, np.array(zi))
    assert abs(st - zi.start_time) < 0.1 * u.ns
    assert u.isclose(z.time_length, zi.time_length)
    assert u.isclose(z.sample_rate, zi.sample_rate)
    assert u.isclose(z.chan_bw, zi.chan_bw)
    assert u.isclose(z.center_freq, zi.center_freq)
    assert z.freq_align == zi.freq_align == 'center'
    assert zi.dtype == out_dtype


def test_radiosignal_slice():
    x = np.ones((16, 16, 4, 2), dtype=np.float32)
    z = pb.RadioSignal(x, sample_rate=1*u.Hz, start_time=Time.now(),
                       center_freq=400*u.MHz, chan_bw=10*u.MHz,
                       freq_align='bottom')

    y = z[2:8, 2:8, 0]
    assert len(y) == 6
    assert u.isclose(y.time_length, 6 * u.s)
    assert np.abs(z.start_time - (y.start_time - 2*u.s)) < 0.1 * u.ns
    assert y.nchan == 6
    assert all(u.isclose(y.channel_freqs, z.channel_freqs[2:8]))
    assert y.freq_align == 'center'

    y = z[:, 1:13]
    assert y.nchan == 12
    assert all(u.isclose(y.channel_freqs, z.channel_freqs[1:13]))
    assert y.freq_align == 'center'

    b = np.random.random((4, 2)) > 0.5
    y = z[2:8, :, b]
    assert y.nchan == z.nchan
    assert y.freq_align == z.freq_align
    assert all(u.isclose(y.channel_freqs, z.channel_freqs))
    assert u.isclose(y.center_freq, z.center_freq)

    y = z[2:8]
    assert y.nchan == z.nchan
    assert y.freq_align == z.freq_align
    assert all(u.isclose(y.channel_freqs, z.channel_freqs))
    assert u.isclose(y.center_freq, z.center_freq)


def test_radiosignal_slice_errors():
    x = np.ones((16, 16, 4, 2), dtype=np.float32)
    z = pb.RadioSignal(x, sample_rate=1*u.Hz, start_time=Time.now(),
                       center_freq=400*u.MHz, chan_bw=10*u.MHz,
                       freq_align='bottom')

    with pytest.raises(IndexError):
        _ = z[:, 'key']

    with pytest.raises(IndexError):
        _ = z[:, 4]

    with pytest.raises(IndexError):
        b = np.random.random(z.nchan) > 0.5
        _ = z[:, b]

    with pytest.raises(AssertionError):
        _ = z[:, 5:5]

    with pytest.raises(AssertionError):
        _ = z[:, 5:5]

    with pytest.raises(AssertionError):
        _ = z[:, ::2]

    with pytest.raises(AssertionError):
        _ = z[:, ::2]


def test_fullstokes_shape():
    for shape in [(10,), (4, 8), (4, 4, 2), (12, 0, 4)]:
        with pytest.raises(ValueError):
            x = np.empty(shape, dtype=np.float32)
            _ = pb.FullStokesSignal(x, sample_rate=1*u.Hz, center_freq=1*u.GHz,
                                    chan_bw=1*u.MHz)

    for shape in [(8, 4, 4), (4, 4, 4, 2)]:
        x = np.empty(shape, dtype=np.float32)
        _ = pb.FullStokesSignal(x, sample_rate=1*u.Hz, center_freq=1*u.GHz,
                                chan_bw=1*u.MHz)


def test_dualpol_shape():
    pol = 'linear'

    for shape in [(10,), (4, 8), (4, 4, 4), (12, 0, 2)]:
        x = np.empty(shape, dtype=np.complex128)
        with pytest.raises(ValueError):
            _ = pb.DualPolarizationSignal(x, sample_rate=1*u.Hz, pol_type=pol,
                                          center_freq=1*u.GHz)

    for shape in [(8, 4, 2), (4, 4, 2, 2)]:
        x = np.empty(shape, dtype=np.complex128)
        _ = pb.DualPolarizationSignal(x, sample_rate=1*u.Hz, pol_type=pol,
                                      center_freq=1*u.GHz)


def test_dualpol_pol_type():
    for pol in ['invalid', 3, ()]:
        x = np.empty((16, 4, 2), dtype=np.complex128)
        with pytest.raises(ValueError):
            _ = pb.DualPolarizationSignal(x, sample_rate=1*u.Hz, pol_type=pol,
                                          center_freq=1*u.GHz)

    for pol in ['linear', 'circular']:
        x = np.empty((16, 4, 2), dtype=np.complex128)
        z = pb.DualPolarizationSignal(x, sample_rate=1*u.Hz, pol_type=pol,
                                      center_freq=1*u.GHz)
        print(z)
        assert z.pol_type == pol
