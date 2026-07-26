"""Gaussian quadrature on simplex reference domains.

Port of CalcSimplexQuad.m (original by Greg von Winckel).

For 1D: standard Gauss-Legendre on [0,1].
For 2D: tensor product Gauss-Legendre with Duffy transformation
        mapping the unit square to the reference triangle.
"""

import numpy as np


def _gauss_legendre_1d(N):
    """Return Gauss-Legendre nodes and weights on [0, 1].

    Uses the Golub-Welsch algorithm (tridiagonal Jacobi matrix).
    """
    if N == 1:
        return np.array([0.5]), np.array([1.0])

    n = np.arange(1, N)
    beta = n / np.sqrt(4 * n**2 - 1)
    T = np.diag(beta, -1) + np.diag(beta, 1)
    eigvals, eigvecs = np.linalg.eigh(T)
    idx = np.argsort(eigvals)
    x = (eigvals[idx] + 1) / 2
    w = eigvecs[0, idx]**2
    return x, w


def simplex_quad(N, dim):
    """Gauss quadrature on an n-dimensional simplex.

    Parameters
    ----------
    N : int
        Number of quadrature points per dimension (order).
    dim : int
        Spatial dimension (1 or 2).

    Returns
    -------
    X : ndarray, shape (N**dim, dim)
        Quadrature points in the reference simplex.
    W : ndarray, shape (N**dim,)
        Quadrature weights.
    """
    if dim == 1:
        q, w = _gauss_legendre_1d(N)
        return q.reshape(-1, 1), w

    elif dim == 2:
        # Tensor product of 1D Gauss-Legendre quadrature on [0,1]
        q1, w1 = _gauss_legendre_1d(N)
        q2, w2 = _gauss_legendre_1d(N)

        Q1, Q2 = np.meshgrid(q1, q2, indexing='ij')
        W1, W2 = np.meshgrid(w1, w2, indexing='ij')

        q1_flat = Q1.ravel()
        q2_flat = Q2.ravel()
        w_flat = (W1 * W2).ravel()

        # Duffy transformation: (u, v) in [0,1]^2 -> (x, y) in reference triangle
        # x = u, y = v * (1 - u)
        # Jacobian: det = |1-u|
        X = np.column_stack([q1_flat, q2_flat * (1.0 - q1_flat)])
        W = w_flat * (1.0 - q1_flat)

        return X, W

    else:
        raise ValueError(f"Simplex quadrature not implemented for dim={dim}")


# Alias for MATLAB compatibility
calc_simplex_quad = simplex_quad
