import astropy.units as u
from astropy.time import Time
from .core import BasebandSignal
from .utils import complex_noise


def generate_fake_baseband(shape):
    assert len(shape) > 1
    z = complex_noise(shape, 1)
    sample_rate = 1 * u.MHz
    center_freq = 400 * u.MHz
    start_time = Time('2020-01-01T06:00:00.000', precision=9)
    bandwidth = shape[1] * sample_rate

    return BasebandSignal(z, sample_rate, start_time, center_freq, bandwidth)
