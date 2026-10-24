"""
Returns the poles/residues expansion of the matrix transfer function for state space system.

This function returns the poles/residues expansion of the matrix transfer function
for spate space system speficied by matrices A,B,C. Note that matrix D is not used,
and can also be skipped as input argument. Complex partial fraction expansion
is returned. Poles is a column vector. Res is a cell array of 2-D matrices such that
Res{k} is the matrix of residues for pole at location (k) of array Poles.

Note: all complex poles and residues are included, not only those with positive
imaginary part!

Important: this function should be used only if the matrices are full. Sparse
matrices are not supported (and will never supported by this algorithm).
"""

import numpy as np
from typing import Union, Tuple, List, Optional
from pathlib import Path


def SS2ResPoles(A: Union[np.ndarray, List[List[float]]],
                B: Union[np.ndarray, List[List[float]]],
                C: Union[np.ndarray, List[List[float]]],
                D: Union[np.ndarray, List[List[float]]] = None,
                RootFigHandle=None) -> Tuple[np.ndarray, List[Optional[np.ndarray]]]:
    """
    Returns the poles/residues expansion of the matrix transfer function for state space system.

    Args:
        A: State matrix (n×n)
        B: Input matrix (n×p)
        C: Output matrix (q×n)
        D: Feedthrough matrix (q×p) - not used in computation
        RootFigHandle: Figure handle for GUI compatibility (unused in RFM implementation)

    Returns:
        Tuple of (Poles, Res) where:
            Poles: column vector of poles (complex eigenvalues of A)
            Res: list of matrices such that Res{k} is the matrix of residues for pole at location (k) of array Poles
                 Each Res{k} is a q×p matrix (output×input)

    Note:
        all complex poles and residues are included, not only those with positive imaginary part!
        This function should be used only if the matrices are full. Sparse matrices are not supported.
    """

    raise NotImplementedError(
        "SS2ResPoles is a placeholder in the upstream RFM code and is not "
        "implemented in the bundled package"
    )
