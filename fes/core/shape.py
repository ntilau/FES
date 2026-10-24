"""Reference shape functions in the v0.1 C++ ordering.

This module reproduces ``cpp/src/shape.cpp`` for two-dimensional simplices
exactly, including the column ordering, the hierarchical scalar basis, and the
sign conventions.  It exists so that the Python port can assemble matrices that
are element-by-element identical to the frozen v0.1 C++ binary.

Conventions (see ``cpp/src/shape.cpp``, ``case 2: // TRIANGLE``):

* Reference triangle: ``v0=(0,0)``, ``v1=(1,0)``, ``v2=(0,1)``.
* Barycentric coordinates: ``l0 = 1-x-y``, ``l1 = x``, ``l2 = y``.
* Scalar (H1) basis, orders 1-4, ordered as
  ``[v0, v1, v2, e01 (levels), e02 (levels), e12 (levels), faces]``.
* H(curl) basis, orders 1-4, ordered as
  ``[edge(0,1), edge(0,2), edge(1,2), bubbles (level 2), faces (p=2),
  bubbles (level 3), faces (p=3), bubbles (level 4), faces (p=4)]``.
* The scalar vertex functions are linear for p=2 and p=3 (the C++ basis is
  hierarchical) and quartic for p=4.  This is intentional: it matches the
  reference C++ basis, not the MATLAB nodal basis in :mod:`fes.core.shape_functions`.

Arrays returned by :func:`shape_2d`:

``Ns``     shape ``(nq, nums)``   scalar basis values
``dNs``    shape ``(nq, nums, 2)`` scalar physical gradients
``Nv``     shape ``(nq, numv, 2)`` vector basis values
``dNv``    shape ``(nq, numv)``   scalar curl ``dNy/dx - dNx/dy``
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


class Shape2D(NamedTuple):
    """Container for the 2D reference-basis arrays."""

    Ns: np.ndarray
    dNs: np.ndarray
    Nv: np.ndarray
    dNv: np.ndarray


def scalar_dof_count(order: int) -> int:
    """Number of scalar DOFs per triangle, matching ``CalcOrderMatSize.m``."""
    return (order + 1) * (order + 2) // 2


def vector_dof_count(order: int) -> int:
    """Number of H(curl) DOFs per triangle: ``order*(order+2)``."""
    return order * (order + 2)


def _barycentric(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(l, dlam)`` for reference points.

    ``l`` has shape ``(nq, 3)`` and holds ``(l0, l1, l2)``.
    ``dlam`` has shape ``(3, 2)`` and holds the *reference* gradients of the
    barycentric coordinates (constant for an affine simplex).
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    if pts.shape[1] != 2:
        raise ValueError("points must have shape (nq, 2)")
    l0 = 1.0 - pts[:, 0] - pts[:, 1]
    l1 = pts[:, 0]
    l2 = pts[:, 1]
    l = np.column_stack([l0, l1, l2])
    dlam_ref = np.array([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]])
    return l, dlam_ref


def _scalar_basis(order: int, l: np.ndarray, dl: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build the hierarchical scalar basis in C++ order.

    Parameters
    ----------
    order : int
        Polynomial order 1-4.
    l : ndarray, shape (nq, 3)
        Barycentric values.
    dl : ndarray, shape (3, 2)
        Physical barycentric gradients.

    Returns
    -------
    Ns, dNs
        Values ``(nq, nums)`` and gradients ``(nq, nums, 2)``.
    """
    l0, l1, l2 = l[:, 0], l[:, 1], l[:, 2]
    g0, g1, g2 = dl[0], dl[1], dl[2]
    nq = l.shape[0]
    nums = scalar_dof_count(order)

    Ns = np.zeros((nq, nums))
    dNs = np.zeros((nq, nums, 2))

    Ns[:, 0], Ns[:, 1], Ns[:, 2] = l0, l1, l2
    dNs[:, 0] = g0
    dNs[:, 1] = g1
    dNs[:, 2] = g2

    if order == 1:
        return Ns, dNs

    # Edge level 1: 4*l_i*l_j for (0,1), (0,2), (1,2).
    Ns[:, 3] = 4.0 * l0 * l1
    dNs[:, 3] = 4.0 * (l0[:, None] * g1 + l1[:, None] * g0)
    Ns[:, 4] = 4.0 * l0 * l2
    dNs[:, 4] = 4.0 * (l0[:, None] * g2 + l2[:, None] * g0)
    Ns[:, 5] = 4.0 * l1 * l2
    dNs[:, 5] = 4.0 * (l1[:, None] * g2 + l2[:, None] * g1)

    if order == 2:
        return Ns, dNs

    # Edge level 2: l_i*l_j*(l_i-l_j).
    Ns[:, 6] = l0 * l1 * (l0 - l1)
    dNs[:, 6] = ((l0[:, None] * g1 + l1[:, None] * g0) * (l0 - l1)[:, None]
                 + (l0 * l1)[:, None] * (g0 - g1))
    Ns[:, 7] = l0 * l2 * (l0 - l2)
    dNs[:, 7] = ((l0[:, None] * g2 + l2[:, None] * g0) * (l0 - l2)[:, None]
                 + (l0 * l2)[:, None] * (g0 - g2))
    Ns[:, 8] = l1 * l2 * (l1 - l2)
    dNs[:, 8] = ((l1[:, None] * g2 + l2[:, None] * g1) * (l1 - l2)[:, None]
                 + (l1 * l2)[:, None] * (g1 - g2))
    # Interior bubble.
    Ns[:, 9] = l0 * l1 * l2
    dNs[:, 9] = (l1 * l2)[:, None] * g0 + (l0 * l2)[:, None] * g1 + (l0 * l1)[:, None] * g2

    if order == 3:
        return Ns, dNs

    # Vertex functions are quartic for p=4 in the C++ implementation.
    # v(t) = t*(4t-1)*(4t-2)*(4t-3)/6, so
    # v'(t) = (b*c*d + 4*a*c*d + 4*a*b*d + 4*a*b*c)/6.
    def _quartic_vertex(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        a = t
        b = 4.0 * t - 1.0
        c = 4.0 * t - 2.0
        d = 4.0 * t - 3.0
        value = a * b * c * d / 6.0
        deriv = (b * c * d + 4.0 * (a * c * d + a * b * d + a * b * c)) / 6.0
        return value, deriv

    for idx, (t, grad) in enumerate(((l0, g0), (l1, g1), (l2, g2))):
        value, deriv = _quartic_vertex(t)
        Ns[:, idx] = value
        dNs[:, idx] = deriv[:, None] * grad

    # Edge level 3: l_i*l_j*(l_i^2-l_j^2).  The C++ code writes the (0,1)
    # edge as l1*l0*(l1^2-l0^2), which is the negative of the symmetric form;
    # keep the oracle's sign.
    Ns[:, 9] = l0 * l1 * (l1 * l1 - l0 * l0)
    dNs[:, 9] = ((l0[:, None] * g1 + l1[:, None] * g0) * (l1 * l1 - l0 * l0)[:, None]
                 + (l0 * l1)[:, None] * (2.0 * l1[:, None] * g1 - 2.0 * l0[:, None] * g0))
    Ns[:, 10] = l0 * l2 * (l0 * l0 - l2 * l2)
    dNs[:, 10] = ((l0[:, None] * g2 + l2[:, None] * g0) * (l0 * l0 - l2 * l2)[:, None]
                  + (l0 * l2)[:, None] * (2.0 * l0[:, None] * g0 - 2.0 * l2[:, None] * g2))
    Ns[:, 11] = l1 * l2 * (l1 * l1 - l2 * l2)
    dNs[:, 11] = ((l1[:, None] * g2 + l2[:, None] * g1) * (l1 * l1 - l2 * l2)[:, None]
                  + (l1 * l2)[:, None] * (2.0 * l1[:, None] * g1 - 2.0 * l2[:, None] * g2))

    # Face functions.
    Ns[:, 12] = l0 * l1 * l2
    dNs[:, 12] = (l1 * l2)[:, None] * g0 + (l0 * l2)[:, None] * g1 + (l0 * l1)[:, None] * g2
    Ns[:, 13] = l0 * l1 * l2 * (l2 - l1)
    dNs[:, 13] = ((l1 * l2 * (l2 - l1))[:, None] * g0
                  + (l0 * l2 * (l2 - l1))[:, None] * g1
                  + (l0 * l1 * (l2 - l1))[:, None] * g2
                  + (l0 * l1 * l2)[:, None] * (g2 - g1))
    Ns[:, 14] = l0 * l1 * l2 * (l1 + l2 - 2.0 * l0)
    dNs[:, 14] = ((l1 * l2 * (l1 + l2 - 2.0 * l0))[:, None] * g0
                  + (l0 * l2 * (l1 + l2 - 2.0 * l0))[:, None] * g1
                  + (l0 * l1 * (l1 + l2 - 2.0 * l0))[:, None] * g2
                  + (l0 * l1 * l2)[:, None] * (g1 + g2 - 2.0 * g0))

    return Ns, dNs


def _vector_basis(order: int, l: np.ndarray, dl: np.ndarray,
                  curl_factor: float) -> tuple[np.ndarray, np.ndarray]:
    """Build the H(curl) basis in C++ order.

    ``curl_factor`` is ``det([grad l1, grad l2])``; the C++ code multiplies the
    reference-coordinate curl by it at the end (see ``dNv *=
    dNs(0,1)*dNs(1,2)-dNs(1,1)*dNs(0,2)``).
    """
    l0, l1, l2 = l[:, 0], l[:, 1], l[:, 2]
    g0, g1, g2 = dl[0], dl[1], dl[2]
    nq = l.shape[0]
    numv = vector_dof_count(order)

    Nv = np.zeros((nq, numv, 2))
    dNv = np.zeros((nq, numv))

    # Lowest-order edge functions, oriented as in the oracle.
    Nv[:, 0] = l0[:, None] * g1 - l1[:, None] * g0
    Nv[:, 1] = l2[:, None] * g0 - l0[:, None] * g2
    Nv[:, 2] = l1[:, None] * g2 - l2[:, None] * g1
    dNv[:, 0] = 2.0
    dNv[:, 1] = -2.0
    dNv[:, 2] = 2.0

    if order == 1:
        return Nv, dNv * curl_factor

    # Quadratic edge bubbles.
    Nv[:, 3] = 4.0 * (l0[:, None] * g1 + l1[:, None] * g0)
    Nv[:, 4] = 4.0 * (l0[:, None] * g2 + l2[:, None] * g0)
    Nv[:, 5] = 4.0 * (l1[:, None] * g2 + l2[:, None] * g1)
    # Quadratic face bubbles.
    Nv[:, 6] = (l0 * l1)[:, None] * g2 - (l0 * l2)[:, None] * g1
    Nv[:, 7] = (l0 * l1)[:, None] * g2 - (l1 * l2)[:, None] * g0
    dNv[:, 6] = 2.0 * l0 - l1 - l2
    dNv[:, 7] = l0 + l2 - 2.0 * l1

    if order == 2:
        return Nv, dNv * curl_factor

    # Cubic edge functions.
    Nv[:, 8] = l0[:, None] * (l0 - 2.0 * l1)[:, None] * g1 \
        + l1[:, None] * (2.0 * l0 - l1)[:, None] * g0
    Nv[:, 9] = l0[:, None] * (l0 - 2.0 * l2)[:, None] * g2 \
        + l2[:, None] * (2.0 * l0 - l2)[:, None] * g0
    Nv[:, 10] = l1[:, None] * (l1 - 2.0 * l2)[:, None] * g2 \
        + l2[:, None] * (2.0 * l1 - l2)[:, None] * g1
    # Cubic face bubble.
    Nv[:, 11] = (l1 * l0)[:, None] * g2 + (l0 * l2)[:, None] * g1 \
        + (l1 * l2)[:, None] * g0
    # Cubic face functions.
    Nv[:, 12] = -(l0 * l1 * (l0 - 2.0 * l2))[:, None] * g2 \
        - (l1 * l2 * (2.0 * l0 - l2))[:, None] * g0 \
        + 3.0 * (l0 * l2 * (l0 - l2))[:, None] * g1
    Nv[:, 13] = -(l1 * l0 * (l1 - 2.0 * l2))[:, None] * g2 \
        - (l0 * l2 * (2.0 * l1 - l2))[:, None] * g1 \
        + 3.0 * (l1 * l2 * (l1 - l2))[:, None] * g0
    Nv[:, 14] = -(l0 * l2 * (l0 - 2.0 * l1))[:, None] * g1 \
        - (l1 * l2 * (2.0 * l0 - l1))[:, None] * g0 \
        + 3.0 * (l0 * l1 * (l0 - l1))[:, None] * g2
    dNv[:, 12] = (-4.0 * l0 * (l0 - 2.0 * l2)
                  + 4.0 * l2 * (2.0 * l0 - l2))
    dNv[:, 13] = (4.0 * l1 * (l1 - 2.0 * l2)
                  - 4.0 * l2 * (2.0 * l1 - l2))
    dNv[:, 14] = (4.0 * l0 * (l0 - 2.0 * l1)
                  - 4.0 * l1 * (2.0 * l0 - l1))

    if order == 3:
        return Nv, dNv * curl_factor

    # Quartic edge functions.
    Nv[:, 15] = (l0 * l0 * (l0 - 3.0 * l1))[:, None] * g1 \
        + (l1 * l1 * (l1 - 3.0 * l0))[:, None] * g0
    Nv[:, 16] = (l0 * l0 * (l0 - 3.0 * l2))[:, None] * g2 \
        + (l2 * l2 * (l2 - 3.0 * l0))[:, None] * g0
    Nv[:, 17] = (l1 * l1 * (l1 - 3.0 * l2))[:, None] * g2 \
        + (l2 * l2 * (l2 - 3.0 * l1))[:, None] * g1
    # Quartic face functions.
    Nv[:, 18] = (l0 * l0 * l1)[:, None] * g2 - (l0 * l0 * l2)[:, None] * g1
    Nv[:, 19] = (l0 * l1 * l1)[:, None] * g2 - (l0 * l1 * l2)[:, None] * g1
    Nv[:, 20] = (l0 * l1 * l2)[:, None] * g2 - (l0 * l2 * l2)[:, None] * g1
    Nv[:, 21] = (l0 * l0 * l1)[:, None] * g2 - (l0 * l1 * l2)[:, None] * g0
    Nv[:, 22] = (l0 * l1 * l1)[:, None] * g2 - (l1 * l1 * l2)[:, None] * g0
    Nv[:, 23] = (l0 * l1 * l2)[:, None] * g2 - (l1 * l2 * l2)[:, None] * g0
    dNv[:, 15] = 3.0 * (l0 * l0 - l1 * l1)
    dNv[:, 16] = 3.0 * (l2 * l2 - l0 * l0)
    dNv[:, 17] = 3.0 * (l1 * l1 - l2 * l2)
    dNv[:, 18] = 2.0 * l0 * l0 - 2.0 * l0 * (l1 + l2)
    dNv[:, 19] = -l1 * l1 + 3.0 * l0 * l1 - l1 * l2
    dNv[:, 20] = -l1 * l2 + 3.0 * l0 * l2 - l2 * l2
    dNv[:, 21] = l0 * l0 - 3.0 * l0 * l1 + l0 * l2
    dNv[:, 22] = 2.0 * l0 * l1 + 2.0 * l1 * l2 - 2.0 * l1 * l1
    dNv[:, 23] = l0 * l2 + l2 * l2 - 3.0 * l1 * l2

    return Nv, dNv * curl_factor


def shape_2d(
    order: int,
    points: np.ndarray,
    inv_j: np.ndarray | None = None,
) -> Shape2D:
    """Evaluate the v0.1 C++ scalar and H(curl) bases on a triangle.

    Parameters
    ----------
    order : int
        Polynomial order, 1-4.
    points : ndarray, shape (nq, 2)
        Points in reference coordinates ``(x, y)``.
    inv_j : ndarray, shape (2, 2), optional
        Inverse of the C++ Jacobian ``J = dNs_ref @ geo``.  When omitted the
        reference triangle is used (identity mapping).

    Returns
    -------
    Shape2D
        ``Ns`` (nq, nums), ``dNs`` (nq, nums, 2), ``Nv`` (nq, numv, 2),
        ``dNv`` (nq, numv).

    Raises
    ------
    ValueError
        If ``order`` is outside 1-4.
    """
    if order not in (1, 2, 3, 4):
        raise ValueError(f"2D shape functions are implemented for orders 1-4, got {order}")

    l, dlam_ref = _barycentric(points)
    if inv_j is None:
        inv_j = np.eye(2)
    inv_j = np.asarray(inv_j, dtype=float)
    if inv_j.shape != (2, 2):
        raise ValueError("inv_j must have shape (2, 2)")

    # Physical barycentric gradients: dlam = invJ @ dlam_ref^T, i.e. row i is
    # the physical gradient of l_i.
    dl = dlam_ref @ inv_j.T
    curl_factor = dl[1, 0] * dl[2, 1] - dl[1, 1] * dl[2, 0]

    Ns, dNs = _scalar_basis(order, l, dl)
    Nv, dNv = _vector_basis(order, l, dl, curl_factor)
    return Shape2D(Ns=Ns, dNs=dNs, Nv=Nv, dNv=dNv)
