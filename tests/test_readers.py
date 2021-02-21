"""Tests for readers in `pulsarbat.reader`."""

import pytest
import numpy as np
import astropy.units as u
from astropy.time import Time
import pulsarbat as pb


class IndexReader(pb.reader.AbstractReader):
    """A reader that returns indices of samples."""
    def __init__(self, /, *, shape, dtype=np.int32, sample_rate=1*u.Hz,
                 start_time=None, signal_type=pb.Signal, **signal_kwargs):
        super().__init__(signal_type=signal_type, shape=shape, dtype=dtype,
                         sample_rate=sample_rate, start_time=start_time,
                         **signal_kwargs)

    def _read_array(self, n, offset, /):
        x = np.arange(offset, offset+n)
        x = x.reshape((-1,) + (self.ndim - 1) * (1,))
        x = x * np.ones(self.sample_shape)
        return x.astype(self.dtype)

    def read(self, n, /, **kwargs):
        return super().read(n, **kwargs)


class TestAbstractReader:
    def test_basic_functionality(self):
        shape = (100, 4)
        dtype = np.uint16
        t0 = Time.now()
        SR = 1 * u.Hz

        r = IndexReader(shape=shape, dtype=dtype, sample_rate=SR,
                        start_time=t0)

        _ = repr(r)
        _ = str(r)
        assert r.shape == shape
        assert len(r) == shape[0]
        assert r.ndim == len(shape)
        assert r.sample_shape == shape[1:]
        assert r.dtype == dtype
        assert u.isclose(SR, r.sample_rate)
        assert u.isclose(1/SR, r.dt)
        assert u.isclose(shape[0]/SR, r.time_length)
        assert Time.isclose(t0, r.start_time)
        assert Time.isclose(t0 + shape[0]/SR, r.stop_time)
        assert r.offset == 0

        for a, b in [(0, 1), (4, 10), (10, 49)]:
            m = r.seek(a)
            assert m == a == r.tell()
            assert Time.isclose(t0 + a/SR, r.time)

            y = r.read(b)
            assert type(y) == pb.Signal
            assert y.dtype == dtype
            assert y.sample_shape == r.sample_shape
            assert np.allclose(np.array(y), np.arange(a, a+b)[:, None])

            m = r.tell()
            assert m == a + b
            assert u.isclose(r.tell(u.s), (a+b)/SR)
            n = r.seek(r.tell(u.s))
            k = r.seek(r.time)
            assert n == m == k

        for k in [3, 14, 27]:
            for i in [k + i/10 for i in range(-4, 5, 2)]:
                a = r.seek(i/SR)
                b = r.seek(t0 + i/SR)
                assert a == b == k

        for c in [3, 5, 15]:
            m = r.seek(-c, whence='end')
            assert m == shape[0] - c

        for c in [-10, shape[0] + 10]:
            with pytest.raises(EOFError):
                r.seek(c)

        with pytest.raises(EOFError):
            r.seek(-1, whence='end')
            _ = r.read(2)

        with pytest.raises(ValueError):
            r.seek(10)
            _ = r.read(-1)

    def test_extra_functionality(self):
        shape = (1024, 4, 2)
        dtype = np.complex64
        SR = 1*u.MHz
        sigtype = pb.DualPolarizationSignal
        sigkw = {'center_freq': 1*u.GHz, 'pol_type': 'linear'}

        r = IndexReader(shape=shape, dtype=dtype, sample_rate=SR,
                        start_time=None, signal_type=sigtype, **sigkw)

        assert r.start_time is None
        assert r.stop_time is None
        assert r.time is None

        r.seek(5)
        x = r.read(17)
        assert type(x) == sigtype
        assert x.pol_type == 'linear'
        assert u.isclose(x.center_freq, sigkw['center_freq'])
        assert x.start_time is None

        r.seek(5)
        y = r.dask_read(17)
        assert np.allclose(np.array(x), np.array(y))

    def test_bad_arguments(self):
        with pytest.raises(ValueError):
            _ = IndexReader(shape=())

        with pytest.raises(ValueError):
            _ = IndexReader(shape=(100,), signal_type=np.ndarray)

        for SR in (-5*u.MHz, 'fish', [10, 20] * u.Hz):
            with pytest.raises(ValueError):
                _ = IndexReader(shape=(100,), sample_rate=SR)

        for t0 in (1*u.s, 'fish', Time([424.23, 23424.42], format='unix')):
            with pytest.raises(ValueError):
                _ = IndexReader(shape=(100,), start_time=t0)

        r = IndexReader(shape=(100,))
        with pytest.raises(ValueError):
            r.seek(10, whence=3)

    def test_broken_reader(self):
        class BrokenReader(pb.reader.AbstractReader):
            """A reader that returns indices of samples."""
            def __init__(self, /, *, shape=(100,), dtype=np.int32, N=0):
                self._N = N
                super().__init__(shape=shape, dtype=dtype, sample_rate=1*u.Hz)

            def _read_array(self, n, offset, /):
                return np.zeros(self._N, dtype=np.int32)

            def read(self, n, /, **kwargs):
                return super().read(n, **kwargs)

        with pytest.raises(ValueError):
            _ = BrokenReader(dtype=np.float64)

        with pytest.raises(ValueError):
            _ = BrokenReader(shape=(100, 4, 2))

        with pytest.raises(ValueError):
            _ = BrokenReader(N=10)


class TestConcatReader:
    def test_simple(self):
        readers = [IndexReader(shape=(100, 1)) for _ in range(10)]
        r = pb.reader.ConcatenatedReader(readers, axis=1)

        x = r.read(20)
        assert x.sample_shape == r.sample_shape == (10,)
        assert np.allclose(np.array(x), np.arange(20)[:, None])

    def test_bad_arguments(self):
        readers = [IndexReader(shape=(100, 1)) for _ in range(10)]

        for axis in [0, 'time', 2]:
            with pytest.raises(ValueError):
                _ = pb.reader.ConcatenatedReader(readers, axis=axis)

        readers.append(np.ones(16))

        with pytest.raises(TypeError):
            _ = pb.reader.ConcatenatedReader(readers, axis=1)
