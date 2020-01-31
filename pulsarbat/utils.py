"""Signal-to-signal transforms."""

import numpy as np
import astropy.units as u
from astropy.time import Time
from .core import BasebandSignal

__all__ = ['generate_fake_baseband']


def complex_noise(N, S):
    """Generates complex gaussian noise of length N and power S."""
    r = np.random.normal(0, 1 / np.sqrt(2), N)
    i = np.random.normal(0, 1 / np.sqrt(2), N)
    return (r + 1j * i) * np.sqrt(S)


def generate_fake_baseband(shape):
    assert len(shape) > 1
    z = complex_noise(shape, 1)
    sample_rate = 1 * u.MHz
    center_freq = 400 * u.MHz
    start_time = Time('2020-01-01T06:00:00.000', precision=9)
    bandwidth = shape[1] * sample_rate

    return BasebandSignal(z, sample_rate, start_time, center_freq, bandwidth)
