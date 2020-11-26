"""Tests for `pulsarbat.RadioSignal` and subclasses."""

import pytest
from pathlib import Path
from collections import namedtuple
import baseband
import numpy as np
import astropy.units as u
from astropy.time import Time
import pulsarbat as pb


Sample = namedtuple("Sample", ("name", "kwargs"))

DATA_DIR = Path(__file__).parent.absolute() / 'data'

SAMPLE_GUPPI = Sample(name=[DATA_DIR / f'fake.{i}.raw' for i in range(4)],
                      kwargs={'format': 'guppi'})
SAMPLE_DADA = Sample(name=str(DATA_DIR / 'sample.dada'),
                     kwargs={'format': 'dada'})
SAMPLE_STOKES_DADA = Sample(name=str(DATA_DIR / 'stokes_ef.dada'),
                            kwargs={'format': 'dada'})
SAMPLE_VDIF = Sample(name=str(DATA_DIR / 'sample.vdif'),
                     kwargs={'format': 'vdif'})


@pytest.mark.parametrize("reader", [pb.reader.DaskBasebandReader,
                                    pb.reader.BasebandReader])
@pytest.mark.parametrize("sample", [SAMPLE_GUPPI, SAMPLE_DADA, SAMPLE_VDIF,
                                    SAMPLE_STOKES_DADA])
def test_basebandreader(reader, sample):
    fh = baseband.open(sample.name, 'rs', **sample.kwargs)
    rdr = reader(sample.name, **sample.kwargs)

    assert u.isclose(fh.sample_rate, rdr.sample_rate)
    assert abs(fh.start_time - rdr.start_time) < 0.1 * u.ns
    assert abs(fh.stop_time - rdr.stop_time) < 0.1 * u.ns
    assert fh.sample_shape == rdr.sample_shape
    assert fh.dtype == rdr.dtype
    assert fh.shape[0] == len(rdr)

    x = fh.read(fh.shape[0])
    y = rdr.read(len(rdr))
    assert isinstance(y, pb.Signal)
    assert np.allclose(np.asarray(x), np.asarray(y))


def test_seek_and_tell():
    r = pb.reader.GUPPIRawReader(SAMPLE_GUPPI.name)
    N, dt = len(r), 320*u.ns

    for i in [0, N, N//2, N//3]:
        M = r.seek(i)
        assert r.tell() == M == i
        assert u.isclose(r.tell(unit=u.s), i*dt)
        assert abs(r.time - (r.start_time + i*dt)) < 0.1 * u.ns
        M = r.seek(i, whence=0)
        assert r.tell() == M == i
        M = r.seek(i*dt, whence='start')
        assert r.tell() == M == i

    for i in [-1, -10, -100, -1234]:
        M = r.seek(i, whence=2)
        assert r.tell() == N + i == M
        M = r.seek(i*dt, whence='end')
        assert r.tell() == N + i == M

    for i, j in [(100, -1), (200, 50), (4333, 0)]:
        r.seek(i)
        M = r.seek(j, whence=1)
        assert r.tell() == M == i + j
        r.seek(i*dt)
        M = r.seek(j*dt, whence='current')
        assert r.tell() == M == i + j

    t = r.start_time + 1000 * dt
    for whence in [0, 1, 2, 'start', 'current', 'end']:
        M = r.seek(t, whence=whence)
        assert r.tell() == M == 1000


def test_seek_errors():
    r = pb.reader.GUPPIRawReader(SAMPLE_GUPPI.name)
    bad_seeks = [-1, 40000, -1*u.s, -320*u.ns, 1*u.s,
                 Time('1997-07-11T12:34:55.99', format='isot'),
                 Time('1997-07-11T12:34:56.02', format='isot')]

    for i in bad_seeks:
        with pytest.raises(EOFError):
            r.seek(i)

    for whence in [3, 'curr', 'side']:
        with pytest.raises(ValueError):
            r.seek(10, whence=whence)


@pytest.mark.parametrize("reader", [pb.reader.BasebandRawReader,
                                    pb.reader.DaskBasebandRawReader])
@pytest.mark.parametrize("sideband", [True, False, np.arange(8) % 3 == 0])
def test_basebandrawreader(reader, sideband):
    r = reader(SAMPLE_VDIF.name, center_freq=0*u.Hz, sideband=sideband,
               **SAMPLE_VDIF.kwargs)
    assert len(r) == 20000
    _ = np.asarray(r.read(10000))
    assert r.tell() == 10000
    _ = np.asarray(r.read(9999))
    assert r.tell() == 19999
    _ = np.asarray(r.read(1))
    assert r.tell() == 20000
    with pytest.raises(EOFError):
        _ = r.read(1)


def test_basebandraw_errors():
    reader = pb.reader.BasebandRawReader

    for sideband in [1, (), np.arange(5) % 3 == 0]:
        with pytest.raises(ValueError):
            _ = reader(SAMPLE_VDIF.name, sideband=sideband,
                       center_freq=0*u.Hz, **SAMPLE_VDIF.kwargs)

    for fcen in [0, [12, 13]*u.Hz, 65*u.m]:
        with pytest.raises(ValueError):
            _ = reader(SAMPLE_VDIF.name, sideband=True,
                       center_freq=fcen, **SAMPLE_VDIF.kwargs)

    for falign in ['middle', 'sideways', 0]:
        with pytest.raises(ValueError):
            _ = reader(SAMPLE_VDIF.name, sideband=True, freq_align=falign,
                       center_freq=0*u.Hz, **SAMPLE_VDIF.kwargs)


@pytest.mark.parametrize("sb", [True, False, np.arange(8) % 3 == 0])
def test_basebandrawreader_consistency(sb):
    r1 = pb.reader.BasebandRawReader(SAMPLE_VDIF.name, center_freq=0*u.Hz,
                                     sideband=sb, **SAMPLE_VDIF.kwargs)

    r2 = pb.reader.DaskBasebandRawReader(SAMPLE_VDIF.name, center_freq=0*u.Hz,
                                         sideband=sb, **SAMPLE_VDIF.kwargs)

    x = r1.read(123)
    y = r2.read(123)
    assert np.allclose(np.asarray(x), np.asarray(y))

    x = r1.read(8192)
    y = r2.read(8192)
    assert np.allclose(np.asarray(x), np.asarray(y))


@pytest.mark.parametrize("reader", [pb.reader.GUPPIRawReader,
                                    pb.reader.DaskGUPPIRawReader])
def test_guppirawreader(reader):
    r = reader(SAMPLE_GUPPI.name)
    assert u.isclose(r.sample_rate, 3.125*u.MHz)
    assert u.isclose(r.center_freq, 344.1875*u.MHz)
    assert r.pol_type == 'linear'
    assert r.dtype == np.complex64
    assert r.freq_align == 'center'
    assert r.sideband is True

    st = Time('1997-07-11T12:34:56', format='isot', precision=9)
    z = r.read(1)
    assert abs(z.start_time - st) < 0.1 * u.ns
    assert u.isclose(z.bandwidth, 12.5*u.MHz)
    fs = [339.5, 342.625, 345.75, 348.875] * u.MHz
    assert u.allclose(fs, z.channel_freqs)
    y = np.array([[5-5j, -31-14j], [-17+6j, -21+0j], [12-22j, 2-12j],
                  [15+17j, -5+21j]], dtype=np.complex64)
    assert np.allclose(np.array(z), y)


@pytest.mark.parametrize("reader", [pb.reader.DADAStokesReader,
                                    pb.reader.DaskDADAStokesReader])
def test_dadastokesreader(reader):
    r = reader(SAMPLE_STOKES_DADA.name)
    assert u.isclose(r.sample_rate, 1 / (131.072 * u.us))
    assert u.isclose(r.center_freq, 7*u.GHz)
    assert r.dtype == np.float32
    assert r.freq_align == 'top'
    assert u.isclose(r.chan_bw, (2 * u.GHz) / 2048)

    st = Time('2019-01-13T15:57:41', format='isot', precision=9)
    z = r.read(1)
    assert abs(z.start_time - st) < 0.1 * u.ns
    assert u.isclose(z.bandwidth, 2*u.GHz)
    assert u.isclose(z.channel_freqs[-1], 8*u.GHz)
    assert u.isclose(z.channel_freqs[1023], z.center_freq)
    y = np.array([[[21, 1, -3, -1], [22, 2, -2, -1], [24, 4, -3, 0],
                   [22, 3, 3, 0]]], dtype=np.float32)
    assert np.allclose(np.array(z[:, 500:504]), y)


def test_dadastokesreader_error():
    with pytest.raises(ValueError):
        _ = pb.reader.DADAStokesReader(SAMPLE_DADA.name)
