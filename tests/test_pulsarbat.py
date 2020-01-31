#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `pulsarbat` package."""

import pytest
import pulsarbat as pb


def test_generate():
    shape = (1024, 4, 2)
    z = pb.utils.generate_fake_baseband(shape)
    assert isinstance(z, pb.BasebandSignal)
    assert z.shape == shape
