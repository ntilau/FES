"""Jacobian computation for triangular elements.

Port of CalcJacobian.m.
"""

import numpy as np


def jacobian_2d(xy):
    """Compute the Jacobian determinant and inverse transpose for a triangle.

    Parameters
    ----------
    xy : ndarray, shape (3, 2)
        Coordinates of the three triangle vertices.

    Returns
    -------
    detJ : float
        Absolute value of the Jacobian determinant.
    invJt : ndarray, shape (2, 2)
        Inverse transpose of the Jacobian matrix.
    """
    J = np.array([
        [xy[1, 0] - xy[0, 0], xy[2, 0] - xy[0, 0]],
        [xy[1, 1] - xy[0, 1], xy[2, 1] - xy[0, 1]]
    ])
    detJ = abs(np.linalg.det(J))
    invJt = np.linalg.inv(J).T
    return detJ, invJt
