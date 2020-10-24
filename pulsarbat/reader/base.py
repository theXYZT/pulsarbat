"""Base module for reader classes."""

from abc import ABC, abstractmethod


__all__ = ['AbstractReader', ]


class AbstractReader(ABC):
    """Abstract base class for data readers.

    Readers are used to read in data into standardized `~pulsarbat.Signal`
    objects. All Reader classes must define their own `read()`, `seek()`,
    and `tell()` methods.
    """
    @abstractmethod
    def seek(self, offset, whence=0):
        """Change read position to the given offset."""
        pass

    @abstractmethod
    def tell(self, unit=None):
        """Return current read position."""
        pass

    @abstractmethod
    def read(self, n, start=None):
        """Reads n time samples of data from an optional start position."""
        pass
