"""Tests for `pulsarbat.Signal` class."""

import pytest
import numpy as np
import dask.array as da
import astropy.units as u
from astropy.time import Time
import pulsarbat as pb

RAND = np.random.default_rng(seed=42)


@pytest.mark.parametrize("fn", [np.random.standard_normal, da.random.standard_normal])
@pytest.mark.parametrize("shape", [(16,), (8, 4)])
def test_signal_basic(fn, shape):
    for x in [fn(shape), fn(shape) + 1j * fn(shape)]:
        z = pb.Signal(x, sample_rate=1 * u.Hz)

        assert isinstance(z, pb.Signal)
        assert z.shape == shape
        assert z.ndim == len(shape)
        assert len(z) == shape[0]
        assert z.sample_shape == shape[1:]

        assert z.dtype == x.dtype
        assert type(z.data) is type(x)
        assert np.all(np.array(z) == np.array(x))


@pytest.mark.parametrize("ts", [None, Time.now()])
@pytest.mark.parametrize("sample_rate", [1 * u.Hz, 33 * u.kHz])
@pytest.mark.parametrize("meta", [None, {"name": "Hello!"}])
def test_signal_attrs(ts, sample_rate, meta):
    shape = (8, 2)
    z = pb.Signal(np.ones(shape), sample_rate=sample_rate, start_time=ts, meta=meta)

    print(z)
    repr(z)

    assert z.sample_rate == sample_rate
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
        assert ts + 0.1 * u.us in z


def test_time_contains():
    def check(n, a, b):
        x = np.ones(n, dtype=bool)
        x[:a] = False
        x[-b:] = False
        return x

    t0 = Time("2018-08-01T13:56:23", format="isot", precision=9)
    z = pb.Signal(np.zeros(10), sample_rate=1 * u.Hz, start_time=t0)

    ts = t0 + np.arange(-2, 12) * z.dt
    k = check(len(ts), 2, 2)
    assert np.allclose(k, z.contains(ts))
    assert np.allclose(k, [t in z for t in ts])

    ts -= 1 * u.ns
    k = check(len(ts), 3, 1)
    assert np.allclose(k, z.contains(ts))
    assert np.allclose(k, [t in z for t in ts])


def test_meta_errors():
    for meta in ["Hello", (0,), {0, 4}]:
        with pytest.raises(ValueError):
            _ = pb.Signal(np.empty((16, 4)), sample_rate=1 * u.Hz, meta=meta)


def test_shape_errors():
    for shape in [(), (15, 0, 4), (4, 0)]:
        with pytest.raises(ValueError):
            _ = pb.Signal(np.empty(shape), sample_rate=1 * u.Hz)


def test_sample_rate_errors():
    for x in [99.0, 4 * u.m, [5, 6] * u.Hz, -1 * u.Hz]:
        with pytest.raises(ValueError):
            _ = pb.Signal(np.empty((8, 4)), sample_rate=x)


@pytest.mark.parametrize(
    "ts", [Time([59843, 45678, 65678], format="mjd"), 59867.2442234]
)
def test_start_time_errors(ts):
    with pytest.raises(ValueError):
        _ = pb.Signal(np.empty((8, 4)), sample_rate=1 * u.Hz, start_time=ts)


class ArbitrarySignal(pb.Signal):
    _req_dtype = (np.int16, np.int32)
    _req_shape = (None, 7, 4, None)
    _axes_labels = {"time": 0, "fish": 1, "foo": 2, "bar": 3}

    def __init__(self, z, /, *, sample_rate, foo):
        self._foo = foo
        super().__init__(z, sample_rate=sample_rate)

    @property
    def foo(self):
        return self._foo


def test_dtype_constraints():
    shape = (10, 7, 4, 1)

    for dtype in [np.int8, np.int16]:
        x = np.zeros(shape, dtype=dtype)
        z = ArbitrarySignal(x, sample_rate=1 * u.Hz, foo="bar")
        assert z.dtype == np.int16

    for dtype in [np.float32, np.int64]:
        x = np.empty(shape, dtype=dtype)
        with pytest.raises(ValueError):
            _ = ArbitrarySignal(x, sample_rate=1 * u.Hz, foo="bar")


def test_shape_constraints():
    for shape in [(), (0, 7, 4), (11, 7, 4, 0), (3, 6, 5, 1)]:
        x = np.empty(shape).astype(np.int8)
        with pytest.raises(ValueError):
            _ = ArbitrarySignal(x, sample_rate=1 * u.Hz, foo="bar")


def test_signal_like():
    x1 = RAND.integers(-50, 50, (10, 7, 4, 1)).astype(np.int8)

    z1 = ArbitrarySignal(x1, sample_rate=1 * u.Hz, foo="bar")
    z2 = pb.Signal.like(z1)

    assert np.array_equal(np.array(z1), np.array(z2))
    assert z1.sample_rate == z2.sample_rate
    assert z1.start_time == z2.start_time
    assert type(z2) is pb.Signal

    x2 = RAND.integers(-50, 50, (10, 7, 4, 1)).astype(np.int8)
    z2 = pb.Signal.like(z1, x2)
    assert not np.array_equal(np.array(z1), np.array(z2))

    z2 = pb.Signal.like(z1, sample_rate=2 * z1.sample_rate)
    assert u.isclose(z2.sample_rate, 2 * z1.sample_rate)

    with pytest.raises(ValueError):
        _ = ArbitrarySignal.like(z2)

    z3 = ArbitrarySignal.like(z2, foo="bar")
    assert np.array_equal(np.array(z1), np.array(z3))
    assert z1.foo == z3.foo == "bar"
    assert u.isclose(z3.sample_rate, 2 * z1.sample_rate)


def test_signal_slice():
    x = np.ones((32, 4, 2), dtype=np.float32)
    z = pb.Signal(x, sample_rate=1 * u.Hz, start_time=Time.now())

    y = z[2:8]
    assert len(y) == 6
    Time.isclose(y.start_time - 2 * u.s, z.start_time)
    assert u.isclose(z.sample_rate, y.sample_rate)
    assert u.isclose(y.time_length, 6 * u.s)

    y = z[::4]
    assert len(y) == 8
    Time.isclose(y.start_time, z.start_time)
    assert u.isclose(y.sample_rate, 0.25 * u.Hz)
    assert u.isclose(z.time_length, y.time_length)

    y = z[:, [0, 2]]
    assert len(y) == len(z)
    Time.isclose(y.start_time, z.start_time)
    assert u.isclose(z.sample_rate, y.sample_rate)
    assert u.isclose(z.time_length, y.time_length)

    b = RAND.random((4, 2)) > 0.5
    y = z[:, b]
    assert len(y) == len(z)
    Time.isclose(y.start_time, z.start_time)
    assert u.isclose(z.sample_rate, y.sample_rate)
    assert u.isclose(z.time_length, y.time_length)

    y = z[5:3]
    assert len(y) == 0
    assert Time.isclose(y.start_time - 5 * u.s, z.start_time)
    assert Time.isclose(y.stop_time - 5 * u.s, z.start_time)
    assert u.isclose(y.time_length, 0 * u.s)

    z = pb.Signal(x, sample_rate=1 * u.Hz)
    assert z[2:].start_time is None
    assert z[:6].start_time is None
    assert z[::4].start_time is None


def test_signal_slice_errors():
    x = np.ones((32, 4, 2), dtype=np.float32)
    st = Time.now()
    z = pb.Signal(x, sample_rate=1 * u.Hz, start_time=st)

    with pytest.raises(IndexError):
        _ = z["key"]

    with pytest.raises(IndexError):
        _ = z[4]

    with pytest.raises(IndexError):
        b = np.random.random(len(z)) > 0.5
        _ = z[b]

    with pytest.raises(AssertionError):
        _ = z[::-2]


def test_get_axis():
    x = np.zeros((5, 7, 4, 3), dtype=np.int32)
    z = ArbitrarySignal(x, sample_rate=1 * u.Hz, foo="bar")

    for a in [-5, 4]:
        with pytest.raises(ValueError):
            _ = z.get_axis(a)

    for a in range(-4, 4):
        assert a == z.get_axis(a)

    for i, a in enumerate(["time", "fish", "foo", "bar"]):
        assert i == z.get_axis(a)


class TestSignalUfuncs:
    @pytest.mark.parametrize("use_dask", [True, False])
    def test_basic_ufunc(self, use_dask):
        arange = da.arange if use_dask else np.arange
        x = arange(8, dtype=np.int64) + 11
        y = arange(8, dtype=np.int64) + 2

        a = pb.Signal(x, sample_rate=2 * u.Hz)
        b = pb.Signal(y, sample_rate=5 * u.Hz)

        for s in [a + 2, a + y, 2 + a, y + a, a + b]:
            assert type(s) == type(a)
            assert s.sample_rate == a.sample_rate

        assert np.allclose(np.array(a + 2), np.array(x + 2))
        assert np.allclose(np.array(a + y), np.array(x + y))
        assert np.allclose(np.array(a + b), np.array(x + y))

        z = x + y
        s = a
        a += b
        assert np.allclose(np.array(a), np.array(z))
        assert np.allclose(np.array(s), np.array(z))

        empty = da.empty if use_dask else np.empty
        c = pb.Signal(empty(a.shape), sample_rate=10 * u.Hz)
        d = np.add(a, b, c)

        assert d is c
        assert c.sample_rate == 10 * u.Hz
        assert np.allclose(np.array(c), np.array(a + b))

    def test_multiple_outs(self):
        x = np.arange(16) / 4.0
        y, z = np.modf(x)

        a = pb.Signal(x, sample_rate=1 * u.Hz)
        b, c = np.modf(a)

        assert type(a) == type(b) == type(c)
        assert a.sample_rate == b.sample_rate == c.sample_rate
        assert np.allclose(b.data, y) and np.allclose(c.data, z)

        b = pb.Signal(np.zeros(x.shape), sample_rate=2 * u.Hz)
        c = pb.Signal(np.zeros(x.shape), sample_rate=5 * u.Hz)

        bb, cc = np.modf(a, b, c)

        assert bb is b and cc is c
        assert type(a) == type(bb) == type(cc)
        assert bb.sample_rate == 2 * u.Hz and cc.sample_rate == 5 * u.Hz
        assert np.allclose(bb.data, y) and np.allclose(cc.data, z)

        b = pb.Signal(np.zeros(x.shape), sample_rate=2 * u.Hz)
        c = pb.Signal(np.zeros(x.shape), sample_rate=5 * u.Hz)

        bb, cc = np.modf(a, b)

        assert bb is b and cc is not c
        assert type(a) == type(bb) == type(cc)
        assert bb.sample_rate == 2 * u.Hz and cc.sample_rate == a.sample_rate
        assert np.allclose(bb.data, y) and np.allclose(cc.data, z)

        b = pb.Signal(np.zeros(x.shape), sample_rate=2 * u.Hz)
        c = pb.Signal(np.zeros(x.shape), sample_rate=5 * u.Hz)

        bb, cc = np.modf(a, out=(None, c))

        assert bb is not b and cc is c
        assert type(a) == type(bb) == type(cc)
        assert bb.sample_rate == a.sample_rate and cc.sample_rate == 5 * u.Hz
        assert np.allclose(bb.data, y) and np.allclose(cc.data, z)


class TestSignalDaskFuncs:
    def test_compute(self):
        a = da.random.standard_normal((256, 16)).astype(np.float32)
        b = a.compute()

        x = pb.Signal(a, sample_rate=1 * u.Hz)

        for kw in [{}, {"scheduler": "threads"}]:
            y = x.compute(**kw)
            assert isinstance(y, type(x))
            assert isinstance(y.data, type(b))
            assert np.allclose(y.data, b)

            x = pb.Signal(b, sample_rate=1 * u.Hz)
            y = x.compute(**kw)
            assert isinstance(y, type(x))
            assert isinstance(y.data, type(b))
            assert np.allclose(y.data, b)

    def test_dask_persist(self):
        def bunch_of_operations(x):
            return np.abs(2 * (x - (x / 2))) + 1

        a = da.random.standard_normal((256, 16)).astype(np.float32)
        b = bunch_of_operations(a).compute()

        x = pb.Signal(a, sample_rate=1 * u.Hz)
        x = bunch_of_operations(x)

        for kw in [{}, {"scheduler": "threads"}]:
            y = x.persist(**kw)

            assert isinstance(y, pb.Signal)
            assert isinstance(y.data, type(a))

            assert len(x.data.__dask_graph__()) == 7
            assert len(y.data.__dask_graph__()) == 1
            assert np.allclose(y.data.compute(), b)

            z = pb.Signal(b, sample_rate=1 * u.Hz)
            y = z.persist(**kw)

            assert isinstance(y, pb.Signal)
            assert isinstance(y.data, type(b))
            assert np.allclose(y.data, b)

    def test_to_dask(self):
        a = RAND.standard_normal((256, 16), dtype=np.float32)
        x = pb.Signal(a, sample_rate=1 * u.Hz)
        y = x.to_dask_array()

        assert isinstance(y, pb.Signal)
        assert isinstance(y.data, da.Array)
        assert np.allclose(a, y.data.compute())

    def test_rechunk(self):
        a = RAND.standard_normal((256, 16), dtype=np.float32)
        x = pb.Signal(a, sample_rate=1 * u.Hz)
        y = x.rechunk((16, 4), balance=True)

        assert isinstance(y, pb.Signal)
        assert isinstance(y.data, da.Array)
        assert y.data.chunksize == (16, 4)
        assert np.allclose(a, y.data.compute())

        y = y.rechunk()

        assert isinstance(y, pb.Signal)
        assert isinstance(y.data, da.Array)
        assert y.data.chunksize[0] == 256
        assert np.allclose(a, y.data.compute())
