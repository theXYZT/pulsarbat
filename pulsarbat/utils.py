"""Signal-to-signal transforms."""

import astropy.units as u


__all__ = ['verify_scalar_quantity']


def verify_scalar_quantity(a, unit):
    if not isinstance(a, u.Quantity):
        raise TypeError(f'Expected astropy Quantity, got {type(a)}')

    if not a.unit.is_equivalent(unit):
        expected = f'Expected units of {unit.physical_type}'
        raise u.UnitTypeError(f'{expected}, got units of {a.unit}')

    if not a.isscalar:
        raise ValueError(f'Expected a scalar quantity.')

    return True
