"""Tests for readers in `pulsarbat.reader`."""

import pytest
import numpy as np
import astropy.units as u
from astropy.time import Time
import pulsarbat as pb
import pulsarbat.readers as pbr


class IndexReader(pbr.BaseReader):
    """A reader that returns indices of samples."""

    def __init__(
        self,
        /,
        *,
        shape,
        dtype=np.int32,
        sample_rate=1 * u.Hz,
        start_time=None,
        signal_type=pb.Signal,
        **signal_kwargs,
    ):
        super().__init__(
            signal_type=signal_type,
            shape=shape,
            dtype=dtype,
            sample_rate=sample_rate,
            start_time=start_time,
            **signal_kwargs,
        )

    def _read_array(self, offset, n, /):
        x = np.arange(offset, offset + n)
        x = x.reshape((-1,) + (self.ndim - 1) * (1,))
        x = x * np.ones(self.sample_shape)
        return x.astype(self.dtype)


class TestBaseReader:
    def test_basic_functionality(self):
        shape = (100, 4)
        dtype = np.uint16
        t0 = Time.now()
        SR = 1 * u.Hz

        r = IndexReader(shape=shape, dtype=dtype, sample_rate=SR, start_time=t0)

        _ = repr(r)
        _ = str(r)
        _ = dir(r)

        assert r.shape == shape
        assert len(r) == shape[0]
        assert r.ndim == len(shape)
        assert r.sample_shape == shape[1:]
        assert r.dtype == dtype

        assert u.isclose(SR, r.sample_rate)
        assert u.isclose(1 / SR, r.dt)
        assert u.isclose(shape[0] / SR, r.time_length)

        assert Time.isclose(t0, r.start_time)
        assert Time.isclose(t0 + shape[0] / SR, r.stop_time)

        assert r.offset_at(t0) == 0
        assert r.offset_at(r.stop_time) == len(r)

        t = r.time_at(0, unit=u.s)
        assert t.unit == u.s
        assert u.isclose(t, 0 * u.s)

        t = r.time_at(60, unit=u.min)
        assert t.unit == u.min
        assert u.isclose(t, 1 * u.min)

        for offset, n in [(0, 1), (4, 10), (10, 49)]:
            assert r.offset_at(offset / SR) == offset

            t = r.time_at(offset)
            assert Time.isclose(t, t0 + offset / SR)

            x = r.read(offset, n)
            assert type(x) == pb.Signal
            assert x.dtype == dtype
            assert x.sample_shape == r.sample_shape
            assert np.allclose(np.array(x), np.arange(offset, offset + n)[:, None])

        for temp in [r.start_time - 10 * u.s, r.stop_time + 10 * u.s]:
            with pytest.raises(EOFError):
                _ = r.offset_at(temp)

        with pytest.raises(EOFError):
            _ = r.read(99, 2)

        with pytest.raises(ValueError):
            _ = r.read(-1, 10)

        with pytest.raises(ValueError):
            _ = r.read(10, -1)

    def test_contains(self):
        shape = (4,)
        t0 = Time("2020-01-01T12:34:56.000", format="isot", precision=9)
        ts = t0 + np.arange(-2, 4 + 2) * u.s

        r = IndexReader(
            shape=shape, dtype=np.uint16, sample_rate=1 * u.Hz, start_time=t0
        )

        assert t0 + 2 * u.s in r
        assert all(r.contains(ts) == [0, 0, 1, 1, 1, 1, 0, 0])

        r = IndexReader(shape=shape, dtype=np.uint16, sample_rate=1 * u.Hz)
        assert not (t0 + 2 * u.s in r)
        assert all(r.contains(ts) == [0] * 8)

    @pytest.mark.parametrize(
        "sigtype, dtype, shape, sigkw",
        [
            (
                pb.DualPolarizationSignal,
                np.complex64,
                (1024, 4, 2),
                {"center_freq": 1 * u.GHz, "pol_type": "linear"},
            ),
            (
                pb.FullStokesSignal,
                np.float64,
                (1024, 4, 4),
                {"center_freq": 1 * u.GHz, "chan_bw": 1 * u.MHz},
            ),
        ],
    )
    def test_extra_functionality(self, sigtype, dtype, shape, sigkw):
        SR = 1 * u.MHz
        r = IndexReader(
            shape=shape,
            dtype=dtype,
            sample_rate=SR,
            start_time=None,
            signal_type=sigtype,
            **sigkw,
        )

        assert r.start_time is None
        assert r.stop_time is None

        offset, n = 5, 17
        x = r.read(offset, n)

        assert type(x) == sigtype
        assert u.allclose(x.center_freq, sigkw["center_freq"])
        assert x.start_time is None

        y = r.dask_read(offset, n)
        assert np.allclose(np.array(x), np.array(y))
        assert u.allclose(x.channel_freqs, y.channel_freqs)

    def test_bad_arguments(self):
        with pytest.raises(ValueError):
            _ = IndexReader(shape=())

        with pytest.raises(ValueError):
            _ = IndexReader(shape=(100,), signal_type=np.ndarray)

        for SR in (-5 * u.MHz, "fish", [10, 20] * u.Hz):
            with pytest.raises(ValueError):
                _ = IndexReader(shape=(100,), sample_rate=SR)

        for t0 in (1 * u.s, "fish", Time([424.23, 23424.42], format="unix")):
            with pytest.raises(ValueError):
                _ = IndexReader(shape=(100,), start_time=t0)

    def test_broken_reader(self):
        class BrokenReader(pbr.BaseReader):
            """A reader that returns indices of samples."""

            def __init__(self, /, *, shape=(100,), dtype=np.int32, N=0):
                self._N = N
                super().__init__(shape=shape, dtype=dtype, sample_rate=1 * u.Hz)

            def _read_array(self, offset, n, /):
                return np.zeros(self._N, dtype=np.int32)

            def read(self, offset, n, /, **kwargs):
                return super().read(offset, n, **kwargs)

        with pytest.raises(ValueError):
            _ = BrokenReader(dtype=np.float64)

        with pytest.raises(ValueError):
            _ = BrokenReader(shape=(100, 4, 2))

        with pytest.raises(ValueError):
            _ = BrokenReader(N=10)
