"""Observation management."""

import astropy.units as u
from astropy.time import Time
from .core import BasebandSignal

__all__ = ['Observation', 'PUPPIObservation']


class Observation:
    def __init__(self, file_handle):
        self.fh = file_handle
        self.start_time = Time(self.fh.start_time, format='isot', precision=9)
        self.stop_time = Time(self.fh.stop_time, format='isot', precision=9)


class PUPPIObservation(Observation):
    def __init__(self, file_handle):
        super().__init__(file_handle)
        self.hdr = self.fh.header0

        self.sample_rate = self.hdr['CHAN_BW'] * u.MHz
        self.center_freq = self.hdr['OBSBW'] * u.MHz
        self.bandwidth = self.sample_rate * self.hdr['OBSNCHAN']

    def read(self, timestamp, num_samples):
        assert self.start_time <= timestamp
        assert timestamp + num_samples * self.sample_rate < self.stop_time

        self.fh.seek(timestamp)
        read_start_time = self.fh.tell('time')

        z = self.fh.read(num_samples)
        return BasebandSignal(z=z.transpose(0, 2, 1),
                              sample_rate=self.sample_rate,
                              start_time=read_start_time,
                              center_freq=self.center_freq,
                              bandwidth=self.bandwidth)
