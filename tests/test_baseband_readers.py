"""Tests for `pulsarbat.RadioSignal` and subclasses."""

import pytest
import numpy as np
import astropy.units as u
from astropy.time import Time
import pulsarbat as pb
from pulsarbat.utils import times_are_close
import baseband
from pathlib import Path


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


@pytest.mark.parametrize("sideband", [True, False])
def test_real_raw_baseband(vdif_fh, sideband):
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


def test_guppi_reader(guppi_fh):
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


@pytest.mark.parametrize("sb", [True, False])
@pytest.mark.parametrize("offset", [99.5, 99.6, 99.75, 99.9, 100.0,
                                    100.1, 100.25, 100.4, 100.5])
def test_baseband_offsets(vdif_fh, offset, sb):
    kw = {'center_freq': 400*u.MHz, 'bandwidth': 128*u.MHz, 'sideband': sb}
    rdr = pb.reader.BasebandRawReader(vdif_fh, **kw)
    rdr.seek(rdr.start_time + offset / rdr.sample_rate)
    t1 = rdr.time
    offset = rdr.tell()
    rdr.seek(offset)
    t2 = rdr.time
    assert times_are_close(t1, t2)


@pytest.mark.parametrize("offset", [99.5, 99.6, 99.75, 99.9, 100.0,
                                    100.1, 100.25, 100.4, 100.5])
def test_guppi_offsets(guppi_fh, offset):
    rdr = pb.reader.GUPPIRawReader(guppi_fh)
    rdr.seek(rdr.start_time + offset / rdr.sample_rate)
    t1 = rdr.time
    offset = rdr.tell()
    rdr.seek(offset)
    t2 = rdr.time
    assert times_are_close(t1, t2)


@pytest.fixture(params=[True, False])
def use_dask(request):
    return request.param


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
