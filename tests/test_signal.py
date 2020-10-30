"""Tests for `pulsarbat.Signal` object."""

import pytest
import numpy as np
import dask.array as da
import astropy.units as u
from astropy.time import Time
import pulsarbat as pb


@pytest.mark.parametrize("use_dask", [True, False])
@pytest.mark.parametrize("use_time", [True, False])
@pytest.mark.parametrize("shape", [(16,), (8, 4)])
@pytest.mark.parametrize("sample_rate", [1 * u.Hz, 33 * u.kHz])
@pytest.mark.parametrize("dtype", [np.float16, np.float64])
def test_signal_basic(use_dask, use_time, shape, sample_rate, dtype):
    if use_dask:
        x = da.random.standard_normal(shape).astype(dtype)
    else:
        x = np.random.standard_normal(shape).astype(dtype)

    timestamp = Time.now()
    if use_time:
        z = pb.Signal(x, sample_rate=sample_rate, start_time=timestamp)
    else:
        z = pb.Signal(x, sample_rate=sample_rate)

    assert isinstance(z, pb.Signal)
    assert z.shape == shape
    assert z.ndim == len(shape)
    assert len(z) == shape[0]
    assert z.sample_shape == shape[1:]
    assert z.sample_rate == sample_rate
    assert z.sample_rate is not sample_rate
    assert u.isclose(z.dt, 1 / sample_rate)
    assert u.isclose(z.time_length, shape[0] / sample_rate)
    assert type(z.data) is type(x)
    assert np.all(np.array(z) == np.array(x))
    print(z)
    repr(z)

    if use_time:
        assert z.start_time == timestamp
        assert z.stop_time == timestamp + (shape[0] / sample_rate)
        assert z.start_time in z
        assert z.stop_time not in z
    else:
        assert z.start_time is None
        assert z.stop_time is None
        assert timestamp not in z


@pytest.mark.parametrize("shape", [(), (0, ), (0, 4, 2)])
def test_empty_signal(shape):
    with pytest.raises(ValueError):
        _ = pb.Signal(np.empty(shape), sample_rate=1 * u.Hz)


def test_sample_rate():
    x = np.random.standard_normal((8, 4, 2))

    with pytest.raises(TypeError):
        _ = pb.Signal(x, sample_rate=400)

    SR_array = np.arange(100, 200, 10) * u.MHz
    with pytest.raises(ValueError):
        _ = pb.Signal(x, sample_rate=SR_array)

    with pytest.raises(u.UnitTypeError):
        _ = pb.Signal(x, sample_rate=40 * u.m)

    SR = 123456 / u.min
    z = pb.Signal(x, sample_rate=SR)
    assert u.isclose(SR, z.sample_rate)


def test_start_time():
    SR = 1 * u.Hz
    x = np.random.standard_normal((16, 4))

    with pytest.raises(ValueError):
        _ = pb.Signal(x, sample_rate=SR, start_time=52000.)

    ts = Time(52000., format='mjd') + np.arange(16) * u.s
    with pytest.raises(ValueError):
        _ = pb.Signal(x, sample_rate=SR, start_time=ts)


class ArbitrarySignal(pb.Signal):
    _dtype = np.int8
    _shape = (7, None, 4, None)


@pytest.mark.parametrize("dtype", [np.float16, np.float64, np.int32])
def test_dtype_constraints(dtype):
    x = np.random.uniform(-50, 50, (7, 2, 4, 1)).astype(dtype)
    SR = 1 * u.Hz
    z = ArbitrarySignal(x, sample_rate=SR)
    assert z.dtype == z.data.dtype == np.int8


@pytest.mark.parametrize("shape", [(7,), (7, 2), (7, 2, 4), (7, 2, 3, 2)])
def test_shape_constraints(shape):
    x = np.random.uniform(-50, 50, shape).astype(np.int8)
    SR = 1 * u.Hz
    with pytest.raises(ValueError):
        _ = ArbitrarySignal(x, sample_rate=SR)


def test_signal_like():
    x = np.random.standard_normal((7, 2, 4, 1))
    SR = 1 * u.Hz
    z1 = ArbitrarySignal(x, sample_rate=SR)
    z2 = pb.Signal.like(z1, z1.data)
    assert np.all(z1.data == z2.data)
    assert z1.sample_rate == z2.sample_rate
    assert z1.start_time == z2.start_time
    assert type(z2) is pb.Signal

    with pytest.raises(TypeError):
        _ = pb.RadioSignal.like(z2, z2.data)
