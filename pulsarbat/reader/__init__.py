"""Reader module for pulsarbat."""

# flake8: noqa


from .base import AbstractReader

try:
    from .baseband_readers import BasebandReader, GUPPIRawReader
except:
    pass
