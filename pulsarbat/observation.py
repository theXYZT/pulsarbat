"""Observation management."""

from abc import ABC, abstractmethod
import astropy.units as u
from astropy.time import Time
from .core import BasebandSignal
from .utils import verify_scalar_quantity
from .predictor import Polyco

__all__ = ['PUPPIRawObservation']


class Observation(ABC):
    """Base class for all observations.

    Observation objects are used to process a stream of data into Signal
    objects for further processing. Subclasses must define a method to
    read samples of data with the signature `obj.read(n)` which will
    read `n` time samples from the current position.
    """
    def __init__(self, file_handle, sample_rate):
        self._fh = file_handle
        if verify_scalar_quantity(sample_rate, u.Hz):
            self.sample_rate = sample_rate.to(u.Hz)

    @abstractmethod
    def read(self, n):
        """Reads n time samples and returns a Signal object."""


class BasebandObservation(Observation):
    """Observations that use file readers from the baseband package."""
    def __init__(self, file_handle):
        super.__init__(file_handle, file_handle.sample_rate)
        self.start_time = Time(self._fh.start_time, format='isot', precision=9)
        self.stop_time = Time(self._fh.stop_time, format='isot', precision=9)


class PulsarObservation(Observation):
    polyco = None
    DM = None
    ref_freq = None

    def add_polyco_from_file(self, filename):
        self.polyco = Polyco(filename)


class PUPPIRawObservation(BasebandObservation, PulsarObservation):
    def __init__(self, file_handle):
        super().__init__(file_handle)
        self._header = self.fh.header0
        assert self.sample_rate == self._header['CHAN_BW'] * u.MHz

        self.center_freq = self._header['OBSFREQ'] * u.MHz
        self.bandwidth = self.sample_rate * self._header['OBSNCHAN']

    def read(self, num_samples, timestamp=None):
        if timestamp is not None:
            assert self.start_time <= timestamp
            assert timestamp + num_samples / self.sample_rate < self.stop_time
            self.fh.seek(timestamp)

        read_start_time = self.fh.tell('time')

        z = self.fh.read(num_samples)
        return BasebandSignal(z.transpose(0, 2, 1),
                              sample_rate=self.sample_rate,
                              start_time=read_start_time,
                              center_freq=self.center_freq,
                              bandwidth=self.bandwidth)
