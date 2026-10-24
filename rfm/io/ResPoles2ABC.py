"""
Compute A,B,C matrices from poles and residues via Gilbert's realizations.

This function computes the A,B,C matrices from the poles and the
matrices of residues via Gilbert's realizations.

Important: this function should be used only if the matrices are full.
Sparse matrices are not supported (and will never supported by this algorithm).

Input parameters:
  rp: vector containing all real poles
  cp: vector containing all complex poles with positive imaginary part
  Rr{k}(i,j): cell array with matrices of residues for real poles
              (port i, port j, real pole k)
  Rc{k}(i,j): cell array with matrices of residues for complex poles
              (port i, port j, complex pole k)
  tol: tolerance

Output parameters:
  A, B, C: Gilbert's realization matrices
"""

import numpy as np
from typing import Union, List, Tuple, Optional
from ..dum import DUM
from ..macromodel import Macromodel


def ResPoles2ABC(rp: Union[np.ndarray, List[float]],
                 cp: Union[np.ndarray, List[complex]],
                 Rr: List[List[Optional[np.ndarray]]],
                 Rc: List[List[Optional[np.ndarray]]],
                 tol: float = 1e-10,
                 RootFigHandle=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Compute A,B,C matrices from poles and residues via Gilbert's realizations.

    Args:
        rp: vector containing all real poles
        cp: vector containing all complex poles with positive imaginary part
        Rr: list of lists containing residue matrices for real poles
            Rr[i][j] corresponds to residue for port i+1, port j+1
        Rc: list of lists containing residue matrices for complex poles
            Rc[i][j] corresponds to residue for port i+1, port j+1
        tol: tolerance for numerical computations (default: 1e-10)
        RootFigHandle: Figure handle for GUI compatibility (unused in RFM implementation)

    Returns:
        Tuple of (A, B, C, Nr) where:
            A, B, C: Gilbert's realization matrices
            Nr: Number of real poles
    """

    raise NotImplementedError(
        "ResPoles2ABC is a placeholder in the upstream RFM code and is not "
        "implemented in the bundled package"
    )
