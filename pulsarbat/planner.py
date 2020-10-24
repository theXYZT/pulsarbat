"""Planners for sequential data processing."""

from collections import namedtuple

Overlap = namedtuple("Overlap", ["before", "after"])


class Planner:
    def __init__(self, reader, ):
        self._reader = reader
        self.overlap = Overlap(0, 0)
