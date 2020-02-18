"""Collection of handy utilities."""

import numpy as np
import astropy.units as u


__all__ = ['verify_scalar_quantity', 'complex_noise']


def verify_scalar_quantity(a, unit):
    if not isinstance(a, u.Quantity):
        raise TypeError(f'Expected astropy Quantity, got {type(a)}')

    if not a.unit.is_equivalent(unit):
        expected = f'Expected units of {unit.physical_type}'
        raise u.UnitTypeError(f'{expected}, got units of {a.unit}')

    if not a.isscalar:
        raise ValueError(f'Expected a scalar quantity.')

    return True


def complex_noise(shape: tuple, power: float):
    """Generates complex gaussian noise with given shape and power."""
    r = np.random.normal(0, 1 / np.sqrt(2), shape)
    i = np.random.normal(0, 1 / np.sqrt(2), shape)
    return (r + 1j * i) * np.sqrt(power)


def csq(x):
    """Returns the complex square of a complex array."""
    return x.real**2 + x.imag**2
