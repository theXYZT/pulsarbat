"""Signal-to-signal transforms."""

import numpy as np
from astropy.time import Time
import baseband

__all__ = []


class BasebandReader:
    """Basic baseband file reader."""
    def __init__(self, files, **baseband_kwargs):
        self.files = files
        self.fh = baseband.open(self.files, 'rs', **baseband_kwargs)

        self.start_time = Time(self.fh.start_time, format='isot', precision=9)
        self.stop_time = Time(self.fh.stop_time, format='isot', precision=9)
