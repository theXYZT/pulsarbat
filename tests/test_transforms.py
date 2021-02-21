"""Tests for core signal functions."""

import pytest
import itertools
import numpy as np
import astropy.units as u
import pulsarbat as pb
from astropy.time import Time


def assert_equal_signals(x, y):
    assert np.allclose(np.array(x), np.array(y))
    assert Time.isclose(x.start_time, y.start_time)
    assert u.isclose(x.sample_rate, y.sample_rate)


def assert_equal_radiosignals(x, y):
    assert_equal_signals(x, y)
    assert u.allclose(x.channel_freqs, y.channel_freqs)


class TestConcatenate:
    def test_concatenate_basic(self):
        """Basic concatenation functionality."""
        shape = (16, 16)
        z = pb.Signal(np.random.default_rng().standard_normal(shape),
                      sample_rate=1*u.Hz, start_time=Time.now())

        x, y = z[:10], z[10:]
        for axis in [0, 'time']:
            z2 = pb.concatenate([x, y], axis=axis)
            assert_equal_signals(z, z2)

            with pytest.raises(ValueError):
                _ = pb.concatenate([y, x], axis=axis)

        y.sample_rate = x.sample_rate * 2
        with pytest.raises(ValueError):
            _ = pb.concatenate([x, y], axis=0)

        x, y = z[:, :10], z[:, 10:]
        z2 = pb.concatenate([x, y], axis=1)
        assert_equal_signals(z, z2)

        with pytest.raises(TypeError):
            _ = pb.concatenate([x, y], axis='freq')

        z = pb.RadioSignal(np.random.default_rng().standard_normal(shape),
                           sample_rate=1*u.Hz, start_time=Time.now(),
                           chan_bw=1*u.MHz, center_freq=1*u.GHz)

        x, y = z[:, :10], z[:, 10:]
        for axis in [1, 'freq']:
            z2 = pb.concatenate([x, y], axis=axis)
            assert_equal_radiosignals(z, z2)

            with pytest.raises(ValueError):
                _ = pb.concatenate([y, x], axis=axis)

        y.chan_bw = x.chan_bw * 2
        with pytest.raises(ValueError):
            _ = pb.concatenate([x, y], axis=1)

    def test_concatenate_valid_signal(self):
        """
             🡒 t
           ┏━━━┳━━━┓
           ┃ A ┃ B ┃
         ⭣ ┣━━━╋━━━┫
         x ┃ C ┃ D ┃
           ┗━━━┻━━━┛
        valid along t = AB, AD, CB, CD
        valid along x = AA, AC, BB, BD, CC, CA, DD, DB
        """
        shape = (16, 16)
        z = pb.Signal(np.random.default_rng().standard_normal(shape),
                      sample_rate=1*u.Hz, start_time=Time.now())

        A, B, C, D = z[:8, :8], z[8:, :8], z[:8, 8:], z[8:, 8:]

        y = pb.concatenate([pb.concatenate([A, B], axis=0),
                            pb.concatenate([C, D], axis=0)], axis=1)
        assert_equal_signals(z, y)

        y = pb.concatenate([pb.concatenate([A, C], axis=1),
                            pb.concatenate([B, D], axis=1)], axis=0)
        assert_equal_signals(z, y)

        for XY in itertools.product([A, B, C, D], repeat=2):
            if XY in [(A, B), (A, D), (C, B), (C, D)]:
                _ = pb.concatenate(XY, axis=0)
            else:
                with pytest.raises(ValueError):
                    _ = pb.concatenate(XY, axis=0)

            if XY in [(A, A), (A, C), (B, B), (B, D),
                      (C, C), (C, A), (D, D), (D, B)]:
                _ = pb.concatenate(XY, axis=1)
            else:
                with pytest.raises(ValueError):
                    _ = pb.concatenate(XY, axis=1)

    def test_concatenate_valid_radiosignal(self):
        """
             🡒 t
           ┏━━━┳━━━┓
           ┃ A ┃ B ┃
         ⭣ ┣━━━╋━━━┫
         f ┃ C ┃ D ┃
           ┗━━━┻━━━┛
        valid along t = AB, CD
        valid along f = AC, BD
        """
        shape = (16, 16)
        z = pb.RadioSignal(np.random.default_rng().standard_normal(shape),
                           sample_rate=1*u.Hz, start_time=Time.now(),
                           chan_bw=1*u.MHz, center_freq=1*u.GHz)

        A, B, C, D = z[:8, :8], z[8:, :8], z[:8, 8:], z[8:, 8:]

        y = pb.concatenate([pb.concatenate([A, B], axis='time'),
                            pb.concatenate([C, D], axis='time')], axis='freq')
        assert_equal_radiosignals(z, y)

        y = pb.concatenate([pb.concatenate([A, C], axis='freq'),
                            pb.concatenate([B, D], axis='freq')], axis='time')
        assert_equal_radiosignals(z, y)

        for XY in itertools.product([A, B, C, D], repeat=2):
            if XY in [(A, B), (C, D)]:
                _ = pb.concatenate(XY, axis='time')
            else:
                with pytest.raises(ValueError):
                    _ = pb.concatenate(XY, axis='time')

            if XY in [(A, C), (B, D)]:
                _ = pb.concatenate(XY, axis='freq')
            else:
                with pytest.raises(ValueError):
                    _ = pb.concatenate(XY, axis='freq')

    def test_concatenate_start_time_none(self):
        shape = (32, 32)
        k = (0, 13, 19, 19, 23, 32)

        z = pb.Signal(np.random.default_rng().standard_normal(shape),
                      sample_rate=1*u.Hz, start_time=Time.now())

        for c in range(len(k) - 1):
            zs = [z[i:j] for i, j in zip(k, k[1:])]

            for j, x in enumerate(zs):
                if j != c:
                    x.start_time = None

            y = pb.concatenate(zs, axis='time')
            assert_equal_signals(z, y)

        zs = [z[i:j] for i, j in zip(k, k[1:])]
        for x in zs:
            x.start_time = None
        y = pb.concatenate(zs, axis='time')
        assert y.start_time is None

    def test_concatenate_edge_cases(self):
        shape = (16, 16)

        z = pb.Signal(np.random.default_rng().standard_normal(shape),
                      sample_rate=1*u.Hz, start_time=Time.now())

        with pytest.raises(ValueError):
            _ = pb.concatenate([])

        with pytest.raises(TypeError):
            _ = pb.concatenate([z.data])

        with pytest.raises(TypeError):
            _ = pb.concatenate([z[:8], z[8:].data])
