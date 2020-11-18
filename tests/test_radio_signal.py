"""Tests for `pulsarbat.RadioSignal` and subclasses."""

import pytest
import numpy as np
import dask.array as da
import astropy.units as u
from astropy.time import Time
import pulsarbat as pb


@pytest.mark.parametrize("fn", [np.ones, da.ones])
@pytest.mark.parametrize("fcen", [400 * u.MHz, 800 * u.MHz])
@pytest.mark.parametrize("bw", [100 * u.MHz, 200 * u.MHz])
@pytest.mark.parametrize("nchan", [7, 8, 9, 10])
@pytest.mark.parametrize("align", ["bottom", "top", "center"])
def test_radiosignal(fn, fcen, bw, nchan, align):
    z = pb.RadioSignal(fn((16, nchan, 2)), sample_rate=1*u.Hz,
                       center_freq=fcen, bandwidth=bw, freq_align=align)

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
    assert u.isclose(z.chan_bandwidth, bw/nchan)
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


@pytest.mark.parametrize("center_freq", [99., 400 * u.m, [50., 60.] * u.MHz])
def test_center_freq_errors(center_freq):
    with pytest.raises(ValueError):
        _ = pb.RadioSignal(np.empty((8, 4)), sample_rate=1*u.Hz,
                           bandwidth=100*u.MHz, center_freq=center_freq)


@pytest.mark.parametrize("bandwidth", [99., 400 * u.m, [50., 60.] * u.MHz])
def test_bandwidth_errors(bandwidth):
    with pytest.raises(ValueError):
        _ = pb.RadioSignal(np.empty((8, 4)), sample_rate=1*u.Hz,
                           center_freq=400*u.MHz, bandwidth=bandwidth)


@pytest.mark.parametrize("nchan", [7, 8, 9, 10])
def test_basebandsignal_verify(nchan):
    x = np.ones((16, nchan))
    sample_rate = 1 * u.MHz
    center_freq = 400 * u.MHz

    z = pb.BasebandSignal(x, sample_rate=sample_rate, center_freq=center_freq,
                          bandwidth=nchan * u.MHz)

    assert u.isclose(z.chan_bandwidth, sample_rate)

    with pytest.raises(ValueError):
        z = pb.BasebandSignal(x, sample_rate=sample_rate,
                              center_freq=center_freq,
                              bandwidth=100 * u.MHz)


@pytest.mark.parametrize("use_dask", [True, False])
@pytest.mark.parametrize("A", [1, 2, 4, 10])
def test_baseband_to_intensity(A, use_dask):
    shape = (1024, 4, 2)

    if use_dask:
        x = da.random.uniform(-np.pi, np.pi, shape)
    else:
        x = np.random.uniform(-np.pi, np.pi, shape)
    z_phasor = np.exp(1j * x).astype(np.complex64)

    st = Time.now()
    z = pb.BasebandSignal(A * z_phasor, sample_rate=1 * u.Hz,
                          center_freq=100 * u.Hz, bandwidth=4 * u.Hz,
                          start_time=st, freq_align='bottom')

    assert isinstance(z.data, type(x))
    zi = z.to_intensity()
    assert isinstance(zi.data, type(z.data))
    assert np.allclose(A**2, np.array(zi))
    assert abs(st - zi.start_time) < 0.1 * u.ns
    assert u.isclose(z.time_length, zi.time_length)
    assert z.freq_align == zi.freq_align == 'bottom'


def test_radiosignal_slice():
    x = np.ones((16, 16, 4, 2), dtype=np.float32)
    z = pb.RadioSignal(x, sample_rate=1*u.Hz, start_time=Time.now(),
                       center_freq=400*u.MHz, bandwidth=160*u.MHz,
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


def test_radiosignal_slice_errors():
    x = np.ones((16, 16, 4, 2), dtype=np.float32)
    z = pb.RadioSignal(x, sample_rate=1*u.Hz, start_time=Time.now(),
                       center_freq=400*u.MHz, bandwidth=160*u.MHz,
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
        _ = z[:, ::2]

    with pytest.raises(AssertionError):
        _ = z[:, ::2]
