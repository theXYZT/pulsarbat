"""Tests for `pulsarbat.RadioSignal` and subclasses."""

import pytest
import numpy as np
import dask.array as da
import astropy.units as u
import pulsarbat as pb
import baseband
from pathlib import Path


@pytest.fixture
def data_dir(request):
    tests_dir = Path(request.module.__file__).parent
    return tests_dir / 'data'


@pytest.fixture
def guppi_fh(data_dir):
    SAMPLE_GUPPI = [data_dir / f'fake.{i}.raw' for i in range(4)]
    fh = baseband.open(SAMPLE_GUPPI, 'rs', format='guppi')
    return fh


def test_guppi_reader(guppi_fh):
    _ = pb.reader.GUPPIRawReader(guppi_fh)
