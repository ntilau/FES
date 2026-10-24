"""
Process macromodel representation and produce new structure by filling all empty fields.

This function processes the macromodel representation stored in structure MOD
and produces a new structure MODfull by filling all empty fields. In particular,

- if only the state-space representation is available, the poles-residues
  representation is computed and stored. This conversion is exact. In case of
  sparse systems, it is assumed that the ABCD matrices are in Gilbert canonical
  form.

- if only the poles-residues representation is available, a Gilbert state-space
  realization is computed and stored. For full systems, this conversion is not
  exact, and requires a threshold for the determination of ranks of residue
  matrices. This threshold is passed in optional argument tol, the default
  being 1e-3. For sparse systems, the conversion is performed by appending all
  poles exciting each column of the transfer matrix. In case of common poles
  through different columns, these poles are replicated and a larger (and
  sparser) system is realized.

- if both representations are already present (or none), no action is taken.
"""

import numpy as np
from typing import Union, Optional
from pathlib import Path
from ..macromodel import Macromodel
from .SS2ResPoles import SS2ResPoles
from .ResPoles2ABC import ResPoles2ABC


def MODFill(MOD: Macromodel, tol: float = 1e-3, RootFigHandle=None) -> Macromodel:
    """
    Process macromodel representation and produce new structure by filling all empty fields.

    Args:
        MOD: Macromodel structure to process
        tol: Tolerance for determining ranks of residue matrices (default: 1e-3)
        RootFigHandle: Figure handle for GUI compatibility (unused in RFM implementation)

    Returns:
        Macromodel structure with both state-space and pole-residue representations filled

    Note:
        - if only state-space is available, computes exact pole-residue representation
        - if only pole-residue is available, computes state-space realization
          (exact for full systems, approximate for sparse systems based on tol)
        - if both or neither are present, returns appropriate result
    """