"""Tests for `pulsarbat.RadioSignal` and subclasses."""

# flake8: noqa

import pytest
import numpy as np
import dask.array as da
import astropy.units as u
import pulsarbat as pb
from scipy.signal import sosfilt, butter


def test_pass():
    assert True
