"""
Unit conversion utilities for the rfm RFM implementation.

Functions for converting units of DUM and Macromodel structures.
"""

import numpy as np
from typing import Union, Tuple
from ..dum import DUM
from ..macromodel import Macromodel


def convert_dum_units(dum: DUM, freq_unit: str = 'HZ',
                      time_unit: str = 'S') -> DUM:
    """
    Convert units of a DUM object.

    Parameters
    ----------
    dum : DUM
        DUM object to convert
    freq_unit : str, optional
        Target frequency unit: 'HZ', 'KHZ', 'MHZ', 'GHZ', 'THz', default is 'HZ'
    time_unit : str, optional
        Target time unit: 'S', 'MS', 'US', 'NS', 'PS', default is 'S'

    Returns
    -------
    DUM
        New DUM object with converted units
    """
    # Create a copy of the DUM
    new_dum = DUM()
    new_dum.Nports = dum.Nports
    new_dum.isfreq = dum.isfreq
    new_dum.istime = dum.istime

    # Convert frequency units (stored in Hz by convention)
    if hasattr(dum, 'f') and len(dum.f) > 0:
        freq_divisor = {
            'HZ': 1.0, 'KHZ': 1e3, 'MHZ': 1e6, 'GHZ': 1e9, 'THZ': 1e12
        }.get(freq_unit.upper(), 1.0)
        new_dum.f = dum.f / freq_divisor
    else:
        new_dum.f = np.array([])

    # Convert time units (stored in seconds by convention)
    if hasattr(dum, 't') and len(dum.t) > 0:
        time_factor = {
            'S': 1.0, 'MS': 1e3, 'US': 1e6, 'NS': 1e9, 'PS': 1e12
        }.get(time_unit.upper(), 1.0)
        new_dum.t = dum.t * time_factor
    else:
        new_dum.t = np.array([])

    # S-parameters are unitless, so they remain the same
    if hasattr(dum, 'S'):
        new_dum.S = dum.S.copy()
    else:
        new_dum.S = np.array([])

    # Copy other attributes if they exist
    for attr in ['Y', 'Z']:
        if hasattr(dum, attr):
            setattr(new_dum, attr, getattr(dum, attr).copy())
        else:
            setattr(new_dum, attr, np.array([]))

    return new_dum


def convert_macromodel_units(model: Macromodel, freq_unit: str = 'HZ') -> Macromodel:
    """
    Convert units of a Macromodel object.

    Parameters
    ----------
    model : Macromodel
        Macromodel object to convert
    freq_unit : str, optional
        Target frequency unit: 'HZ', 'KHZ', 'MHZ', 'GHZ', 'THz', default is 'HZ'

    Returns
    -------
    Macromodel
        New Macromodel object with converted units
    """
    # Create a copy of the model
    new_model = Macromodel(
        representation=getattr(model, 'representation', 'pole-residue')
    )
    new_model.Nports = int(getattr(model, 'Nports', 0))

    freq_divisor = {
        'HZ': 1.0, 'KHZ': 1e3, 'MHZ': 1e6, 'GHZ': 1e9, 'THZ': 1e12
    }.get(freq_unit.upper(), 1.0)

    # Poles share the frequency unit.  To keep H(s) = r/(s - p) + d unchanged
    # when s and p are divided by the same factor, residues divide by it too.
    if hasattr(model, 'poles') and len(model.poles) > 0:
        new_model.poles = model.poles / freq_divisor
    else:
        new_model.poles = np.array([])

    if hasattr(model, 'residues') and len(model.residues) > 0:
        new_model.residues = model.residues / freq_divisor
    else:
        new_model.residues = np.array([])

    # DC gain is frequency-independent.
    new_model.dc_gain = np.array(getattr(model, 'dc_gain', np.array([])), copy=True)

    # Copy state-space matrices if they exist
    for attr in ['A', 'B', 'C', 'D']:
        if hasattr(model, attr):
            setattr(
                new_model, attr,
                np.array(getattr(model, attr), dtype=complex, copy=True),
            )
        else:
            setattr(new_model, attr, np.array([]))

    return new_model
