"""Base module for generators."""

# flake8: noqa

import operator
import numpy as np
import pulsarbat as pb

__all__ = [
    'Generator',
]


class Generator:
    def __init__(self, func=None):
        pass

    def generate(self, a, b):
        pass
