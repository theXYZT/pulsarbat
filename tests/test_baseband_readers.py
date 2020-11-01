"""Tests for `pulsarbat.RadioSignal` and subclasses."""

import pytest
import numpy as np
import astropy.units as u
from astropy.time import Time
import pulsarbat as pb
import baseband
from pathlib import Path


def times_are_close(t1, t2):
    return np.all(np.abs(t1 - t2) < 0.1 * u.ns)


@pytest.fixture
def data_dir(request):
    tests_dir = Path(request.module.__file__).parent
    return tests_dir / 'data'


@pytest.fixture
def guppi_fh(data_dir):
    SAMPLE_GUPPI = [data_dir / f'fake.{i}.raw' for i in range(4)]
    fh = baseband.open(SAMPLE_GUPPI, 'rs', format='guppi')
    return fh


@pytest.fixture
def vdif_fh(data_dir):
    SAMPLE_VDIF = data_dir / 'sample.vdif'
    fh = baseband.open(str(SAMPLE_VDIF), 'rs', format='vdif')
    return fh


def test_basebandreader_vdif(vdif_fh):
    rdr = pb.reader.BasebandReader(vdif_fh)
    assert u.isclose(rdr.sample_rate, 32 * u.MHz)
    st = Time('2014-06-16T05:56:07.000', format='isot')
    assert times_are_close(rdr.start_time, st)
    assert len(rdr) == 40000
    assert times_are_close(rdr.stop_time, st + len(rdr) / rdr.sample_rate)

    for i in [0, 101, 1024, 12345]:
        rdr.seek(i)
        assert rdr.tell() == i
        assert times_are_close(rdr.time, rdr.start_time + i / rdr.sample_rate)

    rdr.seek(0)
    x = rdr.read(16)
    assert isinstance(x, pb.Signal)
    assert len(x) == 16
    assert u.isclose(x.sample_rate, 32 * u.MHz)
    assert times_are_close(x.start_time, st)
    y = rdr.read(16, 0)
    assert np.allclose(np.array(x), np.array(y))


def test_basebandreader_guppi(guppi_fh):
    rdr = pb.reader.BasebandReader(guppi_fh)
    assert u.isclose(rdr.sample_rate, 3.125 * u.MHz)
    st = Time('1997-07-11T12:34:56.000', format='isot')
    assert times_are_close(rdr.start_time, st)
    assert len(rdr) == 32768
    assert times_are_close(rdr.stop_time, st + len(rdr) / rdr.sample_rate)

    for i in [0, 101, 1024, 12345]:
        rdr.seek(i)
        assert rdr.tell() == i
        assert times_are_close(rdr.time, rdr.start_time + i / rdr.sample_rate)

    rdr.seek(0)
    x = rdr.read(16)
    assert isinstance(x, pb.Signal)
    assert len(x) == 16
    assert u.isclose(x.sample_rate, 3.125 * u.MHz)
    assert times_are_close(x.start_time, st)
    y = rdr.read(16, 0)
    assert np.allclose(np.array(x), np.array(y))


@pytest.mark.parametrize("sideband", [True, False])
def test_rawbaseband_vdif(vdif_fh, sideband):
    rdr = pb.reader.BasebandRawReader(vdif_fh, center_freq=400*u.MHz,
                                      bandwidth=128*u.MHz, sideband=sideband)
    assert u.isclose(rdr.sample_rate, 16 * u.MHz)
    st = Time('2014-06-16T05:56:07.000', format='isot')
    assert times_are_close(rdr.start_time, st)
    assert len(rdr) == 20000
    assert times_are_close(rdr.stop_time, st + len(rdr) / rdr.sample_rate)

    for i in [0, 101, 1024, 12345]:
        rdr.seek(i)
        assert rdr.tell() == i
        assert times_are_close(rdr.time, rdr.start_time + i / rdr.sample_rate)

    rdr.seek(0)
    x = rdr.read(16)
    assert isinstance(x, pb.BasebandSignal)
    assert len(x) == 16
    assert u.isclose(x.sample_rate, 16 * u.MHz)
    assert times_are_close(x.start_time, st)
    assert u.isclose(x.center_freq, 400 * u.MHz)
    assert u.isclose(x.bandwidth, 128 * u.MHz)
    assert x.nchan == 8
    y = rdr.read(16, 0)
    assert np.allclose(np.array(x), np.array(y))


def test_guppireader(guppi_fh):
    rdr = pb.reader.GUPPIRawReader(guppi_fh)
    assert u.isclose(rdr.sample_rate, 3.125 * u.MHz)
    st = Time('1997-07-11T12:34:56.000', format='isot')
    assert times_are_close(rdr.start_time, st)
    assert len(rdr) == 32768
    assert times_are_close(rdr.stop_time, st + len(rdr) / rdr.sample_rate)

    for i in [0, 101, 1024, 12345]:
        rdr.seek(i)
        assert rdr.tell() == i
        assert times_are_close(rdr.time, rdr.start_time + i / rdr.sample_rate)

    rdr.seek(0)
    x = rdr.read(16)
    print(x)
    assert isinstance(x, pb.BasebandSignal)
    assert len(x) == 16
    assert u.isclose(x.sample_rate, 3.125 * u.MHz)
    assert times_are_close(x.start_time, st)
    assert u.isclose(x.center_freq, 344.1875 * u.MHz)
    assert u.isclose(x.bandwidth, 12.5 * u.MHz)
    assert x.nchan == 4
    y = rdr.read(16, 0)
    assert np.allclose(np.array(x), np.array(y))


@pytest.mark.parametrize("offset", [(1000 + i)/10 for i in range(-4, 5)])
def test_baseband_offsets(vdif_fh, offset):
    rdr = pb.reader.BasebandReader(vdif_fh)
    off = rdr.seek(rdr.start_time + offset / rdr.sample_rate)
    assert off == 100
    t1 = rdr.time
    offset = rdr.tell()
    rdr.seek(offset)
    t2 = rdr.time
    assert times_are_close(t1, t2)


@pytest.mark.parametrize("offset", [(1000 + i)/10 for i in range(-4, 5)])
def test_rawbaseband_offsets(vdif_fh, offset):
    kw = {'center_freq': 400*u.MHz, 'bandwidth': 128*u.MHz, 'sideband': True}
    rdr = pb.reader.BasebandRawReader(vdif_fh, **kw)
    off = rdr.seek(rdr.start_time + offset / rdr.sample_rate)
    assert off == 100
    t1 = rdr.time
    offset = rdr.tell()
    rdr.seek(offset)
    t2 = rdr.time
    assert times_are_close(t1, t2)


@pytest.mark.parametrize("offset", [(1000 + i)/10 for i in range(-4, 5)])
def test_guppi_offsets(guppi_fh, offset):
    rdr = pb.reader.GUPPIRawReader(guppi_fh)
    off = rdr.seek(rdr.start_time + offset / rdr.sample_rate)
    assert off == 100
    t1 = rdr.time
    offset = rdr.tell()
    rdr.seek(offset)
    t2 = rdr.time
    assert times_are_close(t1, t2)


@pytest.mark.parametrize("use_dask", [True, False])
@pytest.mark.parametrize("bb_format", ['guppi', 'vdif'])
def test_reader_read(use_dask, bb_format, guppi_fh, vdif_fh):
    kw = {'center_freq': 400*u.MHz, 'bandwidth': 128*u.MHz, 'sideband': True}

    if use_dask:
        if bb_format == 'guppi':
            fh = pb.reader.DaskGUPPIRawReader(guppi_fh)
        elif bb_format == 'vdif':
            fh = pb.reader.DaskBasebandRawReader(vdif_fh, **kw)
    else:
        if bb_format == 'guppi':
            fh = pb.reader.GUPPIRawReader(guppi_fh)
        elif bb_format == 'vdif':
            fh = pb.reader.BasebandRawReader(vdif_fh, **kw)

    N = 1024
    x = fh.read(N, 0)
    assert isinstance(x, pb.Signal)
    assert len(x) == 1024
    y = fh.read(N, 0)
    assert np.allclose(np.array(x), np.array(y))
    y = fh.read(N, 1024)
    assert times_are_close(x.stop_time, y.start_time)
