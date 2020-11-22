"""Tests for `pulsarbat.Signal` class."""

import pytest
import numpy as np
import dask.array as da
import astropy.units as u
from astropy.time import Time
import pulsarbat as pb


@pytest.mark.parametrize("fn", [np.ones, da.ones])
@pytest.mark.parametrize("ts", [None, Time.now()])
@pytest.mark.parametrize("shape", [(16,), (8, 4)])
@pytest.mark.parametrize("sample_rate", [1 * u.Hz, 33 * u.kHz])
@pytest.mark.parametrize("dtype", [np.float32, np.complex64])
def test_signal_basic(fn, ts, shape, sample_rate, dtype):
    x = fn(shape, dtype=dtype)
    z = pb.Signal(x, sample_rate=sample_rate, start_time=ts)

    assert isinstance(z, pb.Signal)
    print(z)
    repr(z)

    assert z.shape == shape
    assert z.ndim == len(shape)
    assert len(z) == shape[0]
    assert z.sample_shape == shape[1:]
    assert z.dtype == x.dtype
    assert type(z.data) is type(x)
    assert np.all(np.array(z) == np.array(x))

    assert z.sample_rate == sample_rate
    assert z.sample_rate is not sample_rate
    assert u.isclose(z.dt, 1 / sample_rate)
    assert u.isclose(z.time_length, shape[0] / sample_rate)

    if ts is None:
        assert z.start_time is None
        assert z.stop_time is None
        assert Time.now() not in z
    else:
        assert z.start_time == ts
        assert z.stop_time == ts + (shape[0] / sample_rate)
        assert z.start_time in z
        assert z.stop_time not in z
        assert ts + 1 * u.ns in z


@pytest.mark.parametrize("dt", [0 * u.s, np.sqrt(2) * u.day, 1/(4/7 * u.Hz)])
def test_time_contains(dt):
    st = Time.now()
    x = np.ones((16, 2), dtype=np.float32)
    z = pb.Signal(x, sample_rate=1*u.Hz, start_time=st)
    N = 10
    ts = ((st + np.arange(-N, len(z)+N) * u.s) + dt) - dt
    check = np.array([t in z for t in ts])
    assert np.all(check[N:-N])
    assert not np.all(check[:N])
    assert not np.all(check[-N:])


@pytest.mark.parametrize("shape", [(), (0, ), (0, 4, 2)])
def test_shape_errors(shape):
    with pytest.raises(ValueError):
        _ = pb.Signal(np.empty(shape), sample_rate=1 * u.Hz)


def test_sample_rate_errors():
    for x in [99., 4*u.m, [5, 6]*u.Hz, -1*u.Hz]:
        with pytest.raises(ValueError):
            _ = pb.Signal(np.empty((8, 4)), sample_rate=x)


@pytest.mark.parametrize("ts", [Time([59843, 45678, 65678], format='mjd'),
                                59867.2442234])
def test_start_time_errors(ts):
    with pytest.raises(ValueError):
        _ = pb.Signal(np.empty((8, 4)), sample_rate=1 * u.Hz, start_time=ts)


class ArbitrarySignal(pb.Signal):
    _req_dtype = (np.int8, np.int16)
    _req_shape = (7, None, 4, None)

    def __init__(self, z, /, *, sample_rate, foo):
        self._foo = foo
        super().__init__(z, sample_rate=sample_rate)

    @property
    def foo(self):
        return self._foo


@pytest.mark.parametrize("dtype", [np.int8, np.int16])
def test_dtype_constraints(dtype):
    x = np.random.uniform(-50, 50, (7, 2, 4, 1)).astype(dtype)
    z = ArbitrarySignal(x, sample_rate=1*u.Hz, foo='bar')
    assert z.dtype == dtype


@pytest.mark.parametrize("dtype", [np.float32, np.int32])
def test_dtype_constraints_error(dtype):
    x = np.random.uniform(-50, 50, (7, 2, 4, 1)).astype(dtype)
    with pytest.raises(ValueError):
        _ = ArbitrarySignal(x, sample_rate=1*u.Hz, foo='bar')


@pytest.mark.parametrize("shape", [(7,), (7, 2), (7, 1, 4), (7, 2, 3, 2)])
def test_shape_constraints(shape):
    x = np.random.uniform(-50, 50, shape).astype(np.int8)
    with pytest.raises(ValueError):
        _ = ArbitrarySignal(x, sample_rate=1*u.Hz, foo='bar')


def test_signal_like():
    x1 = np.random.uniform(-50, 50, (7, 1, 4, 1)).astype(np.int8)
    x2 = np.random.uniform(-50, 50, (7, 1, 4, 1)).astype(np.int8)

    z1 = ArbitrarySignal(x1, sample_rate=1*u.Hz, foo='bar')
    z2 = pb.Signal.like(z1)

    assert np.array_equal(np.array(z1), np.array(z2))
    assert z1.sample_rate == z2.sample_rate
    assert z1.start_time == z2.start_time
    assert type(z2) is pb.Signal

    z2 = pb.Signal.like(z1, x2)
    assert not np.array_equal(np.array(z1), np.array(z2))

    z2 = pb.Signal.like(z1, sample_rate=2*z1.sample_rate)
    assert u.isclose(z2.sample_rate, 2 * z1.sample_rate)

    with pytest.raises(ValueError):
        _ = ArbitrarySignal.like(z2)

    z3 = ArbitrarySignal.like(z2, foo='bar')
    assert np.array_equal(np.array(z1), np.array(z3))
    assert z1.foo == z3.foo == 'bar'
    assert u.isclose(z3.sample_rate, 2 * z1.sample_rate)


def test_signal_slice():
    x = np.ones((32, 4, 2), dtype=np.float32)
    z = pb.Signal(x, sample_rate=1*u.Hz, start_time=Time.now())

    y = z[2:8]
    assert len(y) == 6
    assert np.abs(z.start_time - (y.start_time - 2*u.s)) < 0.1 * u.ns
    assert u.isclose(z.sample_rate, y.sample_rate)
    assert u.isclose(y.time_length, 6 * u.s)

    y = z[::4]
    assert len(y) == 8
    assert np.abs(z.start_time - y.start_time) < 0.1 * u.ns
    assert u.isclose(y.sample_rate, 0.25 * u.Hz)
    assert u.isclose(z.time_length, y.time_length)

    y = z[:, [0, 2]]
    assert len(y) == len(z)
    assert np.abs(z.start_time - y.start_time) < 0.1 * u.ns
    assert u.isclose(z.sample_rate, y.sample_rate)
    assert u.isclose(z.time_length, y.time_length)

    b = np.random.random((4, 2)) > 0.5
    y = z[:, b]
    assert len(y) == len(z)
    assert np.abs(z.start_time - y.start_time) < 0.1 * u.ns
    assert u.isclose(z.sample_rate, y.sample_rate)
    assert u.isclose(z.time_length, y.time_length)

    z = pb.Signal(x, sample_rate=1*u.Hz)
    assert z[2:].start_time is None
    assert z[:6].start_time is None
    assert z[::4].start_time is None


def test_signal_slice_errors():
    x = np.ones((32, 4, 2), dtype=np.float32)
    st = Time.now()
    z = pb.Signal(x, sample_rate=1*u.Hz, start_time=st)

    with pytest.raises(IndexError):
        _ = z['key']

    with pytest.raises(IndexError):
        _ = z[4]

    with pytest.raises(IndexError):
        b = np.random.random(len(z)) > 0.5
        _ = z[b]

    with pytest.raises(AssertionError):
        _ = z[::-2]

    with pytest.raises(AssertionError):
        _ = z[5:5]

    with pytest.raises(AssertionError):
        _ = z[10:5]
