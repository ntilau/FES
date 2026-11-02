"""Reference shape functions on the triangle in the cFES ordering.

This module transcribes ``cFES/src/Shape.cpp`` (``case 2: // TRIANGLE``)
exactly, including the column ordering, the sign conventions, and the p=4
cardinal basis.  It is the two-dimensional companion to
:mod:`fes.core.shape3d`.

Conventions (see ``Shape.cpp``, ``case 2: // TRIANGLE``):

* Reference triangle: ``v0=(0,0)``, ``v1=(1,0)``, ``v2=(0,1)``.
* Barycentric coordinates: ``l0 = 1-x-y``, ``l1 = x``, ``l2 = y``.
* The C++ code writes ``cPos = (x, y)`` and then appends ``cPos(2) = l0``, so
  ``cPos(0), cPos(1), cPos(2) = l1, l2, l0``.  Every expression below is
  transcribed in terms of ``l0, l1, l2``.
* Scalar (H1) basis, orders 1-4.  Orders 2-3 are hierarchical; order 4 is the
  P4 cardinal basis on the 15 Lagrange nodes.  Edges are ordered
  ``{1,2}, {0,2}, {0,1}`` (the ``facEdges`` column order).
* H(curl) basis, orders 1-4, ordered as
  ``[edge type 0 (3), type 1 (3), type 2 (3), type 3 (3), faces (12)]`` for
  p=4; the lower orders interleave the edge/face blocks as in ``Shape.cpp``.
* The p=4 face functions are shared with the tetrahedron through
  :func:`_face_p4`, so the 2D port basis is the tangential trace of the 3D
  face basis.

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


def _product_terms(values, grads):
    """Value and gradient of ``prod(values)``.

    ``values`` is a list of ``(nq,)`` arrays, ``grads`` the matching
    ``(nq, 2)`` gradients.  Returns ``(value, gradient)``.
    """
    nq = values[0].shape[0]
    value = np.ones(nq)
    for item in values:
        value = value * item
    grad = np.zeros((nq, 2))
    for i, gi in enumerate(grads):
        term = np.array(gi, dtype=float, copy=True)
        for j, item in enumerate(values):
            if j != i:
                term = term * item[:, None]
        grad = grad + term
    return value, grad


def _scalar_basis(order: int, l: np.ndarray, dl: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build the scalar basis in C++ order.

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

    # Vertices: l0, l1, l2 (Shape.cpp:704-716).
    Ns[:, 0], Ns[:, 1], Ns[:, 2] = l0, l1, l2
    dNs[:, 0] = g0
    dNs[:, 1] = g1
    dNs[:, 2] = g2

    if order == 1:
        return Ns, dNs

    if order == 2:
        # Shape.cpp:761-773: edges {1,2}, {0,2}, {0,1}.
        Ns[:, 3] = 4.0 * l1 * l2
        dNs[:, 3] = 4.0 * (l1[:, None] * g2 + l2[:, None] * g1)
        Ns[:, 4] = 4.0 * l0 * l2
        dNs[:, 4] = 4.0 * (l0[:, None] * g2 + l2[:, None] * g0)
        Ns[:, 5] = 4.0 * l0 * l1
        dNs[:, 5] = 4.0 * (l0[:, None] * g1 + l1[:, None] * g0)
        return Ns, dNs

    if order == 3:
        # Shape.cpp:784-847: the p=2 edge functions are kept, then the cubic
        # edge functions and the interior bubble are added.
        Ns[:, 3] = 4.0 * l1 * l2
        dNs[:, 3] = 4.0 * (l1[:, None] * g2 + l2[:, None] * g1)
        Ns[:, 4] = 4.0 * l0 * l2
        dNs[:, 4] = 4.0 * (l0[:, None] * g2 + l2[:, None] * g0)
        Ns[:, 5] = 4.0 * l0 * l1
        dNs[:, 5] = 4.0 * (l0[:, None] * g1 + l1[:, None] * g0)
        terms = (
            (6, (l1, l2, l1 - l2), (g1, g2, g1 - g2)),
            (7, (l0, l2, l0 - l2), (g0, g2, g0 - g2)),
            (8, (l0, l1, l0 - l1), (g0, g1, g0 - g1)),
            (9, (l0, l1, l2), (g0, g1, g2)),
        )
        for idx, values, grads in terms:
            Ns[:, idx], dNs[:, idx] = _product_terms(values, grads)
        return Ns, dNs

    # order == 4: P4 cardinal basis (Shape.cpp:911-984).
    def L0(t):
        return np.ones_like(t)

    def L1(t):
        return t

    def L2(t):
        return t * (t - 1.0) / 2.0

    def L3(t):
        return t * (t - 1.0) * (t - 2.0) / 6.0

    def L4(t):
        return t * (t - 1.0) * (t - 2.0) * (t - 3.0) / 24.0

    def dL0(t):
        return np.zeros_like(t)

    def dL1(t):
        return np.ones_like(t)

    def dL2(t):
        return t - 0.5

    def dL3(t):
        return (3.0 * t * t - 6.0 * t + 2.0) / 6.0

    def dL4(t):
        return (
            (t - 1.0) * (t - 2.0) * (t - 3.0)
            + t * (t - 2.0) * (t - 3.0)
            + t * (t - 1.0) * (t - 3.0)
            + t * (t - 1.0) * (t - 2.0)
        ) / 24.0

    Lf = (L0, L1, L2, L3, L4)
    dLf = (dL0, dL1, dL2, dL3, dL4)
    kn = (
        (4, 0, 0), (0, 4, 0), (0, 0, 4),        # vertices
        (0, 3, 1), (3, 0, 1), (3, 1, 0),        # edge type 0
        (0, 2, 2), (2, 0, 2), (2, 2, 0),        # edge type 1
        (0, 1, 3), (1, 0, 3), (1, 3, 0),        # edge type 2
        (2, 1, 1), (1, 2, 1), (1, 1, 2),        # face interior
    )
    # lambda_0 = l0, lambda_1 = l1, lambda_2 = l2.
    for idx, (k0, k1, k2) in enumerate(kn):
        v0 = Lf[k0](4.0 * l0)
        v1 = Lf[k1](4.0 * l1)
        v2 = Lf[k2](4.0 * l2)
        w0 = dLf[k0](4.0 * l0)
        w1 = dLf[k1](4.0 * l1)
        w2 = dLf[k2](4.0 * l2)
        Ns[:, idx] = v0 * v1 * v2
        # Physical gradient via the chain rule; in reference coordinates this
        # is Shape.cpp's 4*(-w0 v1 v2 + v0 w1 v2), 4*(-w0 v1 v2 + v0 v1 w2)
        # followed by invJ.
        dNs[:, idx, :] = (
            4.0 * (w0 * v1 * v2)[:, None] * g0
            + 4.0 * (v0 * w1 * v2)[:, None] * g1
            + 4.0 * (v0 * v1 * w2)[:, None] * g2
        )
    return Ns, dNs


def _face_p4(
    k: int,
    l0: np.ndarray,
    l1: np.ndarray,
    l2: np.ndarray,
    gA: np.ndarray,
    gB: np.ndarray,
    gC: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """p=4 face function *k* on a triangle.

    Direct port of ``FaceP4`` (``Shape.cpp:17-81``).  ``l0,l1,l2`` are the
    barycentric values of the triangle and ``gA,gB,gC`` their gradients.  In
    3D the gradients are true 3-vectors and ``C`` is the curl; in 2D they are
    embedded with ``z=0`` and the z-component of ``C`` is the scalar curl.

    Returns ``(N, C)`` with shape ``(nq, 3)`` each.
    """
    l0 = np.asarray(l0, dtype=float)
    l1 = np.asarray(l1, dtype=float)
    l2 = np.asarray(l2, dtype=float)
    gA = np.asarray(gA, dtype=float)
    gB = np.asarray(gB, dtype=float)
    gC = np.asarray(gC, dtype=float)

    zero = np.zeros_like(gA)
    b = l0 * l1 * l2
    gb = (l1 * l2)[:, None] * gA + (l0 * l2)[:, None] * gB + (l0 * l1)[:, None] * gC

    def sc(value: np.ndarray, vector: np.ndarray) -> np.ndarray:
        return value[:, None] * vector

    def cr(a: np.ndarray, bvec: np.ndarray) -> np.ndarray:
        return np.cross(a, bvec)

    if k == 0:
        N = sc(l0 * l1, gC) - sc(l0 * l2, gB)
        C = cr(sc(l1, gA) + sc(l0, gB), gC) - cr(sc(l2, gA) + sc(l0, gC), gB)
    elif k == 1:
        N = sc(l0 * l1, gC) - sc(l1 * l2, gA)
        C = cr(sc(l1, gA) + sc(l0, gB), gC) - cr(sc(l2, gB) + sc(l1, gC), gA)
    elif k == 2:
        N = gb
        C = zero
    elif k == 3:
        N = (-sc(l0 * l1 * (l1 - 2.0 * l2), gC)
             - sc(l0 * l2 * (2.0 * l1 - l2), gB)
             + sc(3.0 * l1 * l2 * (l1 - l2), gA))
        C = (-4.0 * sc(l1 * (l1 - 2.0 * l2), cr(gA, gC))
             - 4.0 * sc(l2 * (2.0 * l1 - l2), cr(gA, gB)))
    elif k == 4:
        N = (-sc(l0 * l1 * (l0 - 2.0 * l2), gC)
             - sc(l1 * l2 * (2.0 * l0 - l2), gA)
             + sc(3.0 * l0 * l2 * (l0 - l2), gB))
        C = (-4.0 * sc(l0 * (l0 - 2.0 * l2), cr(gB, gC))
             + 4.0 * sc(l2 * (2.0 * l0 - l2), cr(gA, gB)))
    elif k == 5:
        N = (-sc(l0 * l2 * (l0 - 2.0 * l1), gB)
             - sc(l1 * l2 * (2.0 * l0 - l1), gA)
             + sc(3.0 * l0 * l1 * (l0 - l1), gC))
        C = (4.0 * sc(l0 * (l0 - 2.0 * l1), cr(gB, gC))
             + 4.0 * sc(l1 * (2.0 * l0 - l1), cr(gA, gC)))
    elif k == 6:
        N = sc(l0, gb) + sc(b, gA)
        C = zero
    elif k == 7:
        N = sc(l1, gb) + sc(b, gB)
        C = zero
    elif k == 8:
        N = sc(b * l0, gB - gA)
        base = sc(l0, gb) + sc(b, gA)
        C = cr(base, gB) - cr(base, gA)
    elif k == 9:
        N = sc(b * l0, gC - gA)
        base = sc(l0, gb) + sc(b, gA)
        C = cr(base, gC) - cr(base, gA)
    elif k == 10:
        N = sc(b * l1, gB - gA)
        base = sc(l1, gb) + sc(b, gB)
        C = cr(base, gB) - cr(base, gA)
    elif k == 11:
        N = sc(b * l1, gC - gA)
        base = sc(l1, gb) + sc(b, gB)
        C = cr(base, gC) - cr(base, gA)
    else:
        raise ValueError(f"FaceP4 index out of range: {k}")
    return N, C


def _vector_basis(order: int, l: np.ndarray, dl: np.ndarray,
                  curl_factor: float) -> tuple[np.ndarray, np.ndarray]:
    """Build the H(curl) basis in C++ order.

    ``curl_factor`` is ``det([grad l1, grad l2])``; the C++ code multiplies
    every curl entry by it at the end (``dNv *= dNs(0,1)*dNs(1,2) -
    dNs(1,1)*dNs(0,2)``).
    """
    l0, l1, l2 = l[:, 0], l[:, 1], l[:, 2]
    g0, g1, g2 = dl[0], dl[1], dl[2]
    nq = l.shape[0]
    numv = vector_dof_count(order)

    Nv = np.zeros((nq, numv, 2))
    dNv = np.zeros((nq, numv))

    # Lowest-order edge functions (Shape.cpp:723-725).
    #   Nv.col(0) = l1*g2 - l2*g1
    #   Nv.col(1) = l0*g2 - l2*g0
    #   Nv.col(2) = l0*g1 - l1*g0
    Nv[:, 0] = l1[:, None] * g2 - l2[:, None] * g1
    Nv[:, 1] = l0[:, None] * g2 - l2[:, None] * g0
    Nv[:, 2] = l0[:, None] * g1 - l1[:, None] * g0
    dNv[:, 0] = 2.0
    dNv[:, 1] = -2.0
    dNv[:, 2] = 2.0

    if order == 1:
        return Nv, dNv * curl_factor

    # Quadratic edge bubbles (Shape.cpp:744-746).
    Nv[:, 3] = 4.0 * (l1[:, None] * g2 + l2[:, None] * g1)
    Nv[:, 4] = 4.0 * (l0[:, None] * g2 + l2[:, None] * g0)
    Nv[:, 5] = 4.0 * (l0[:, None] * g1 + l1[:, None] * g0)
    # Quadratic face bubbles (Shape.cpp:747-748).
    Nv[:, 6] = (l0 * l1)[:, None] * g2 - (l0 * l2)[:, None] * g1
    Nv[:, 7] = (l0 * l1)[:, None] * g2 - (l1 * l2)[:, None] * g0
    dNv[:, 6] = 2.0 * l0 - l1 - l2
    dNv[:, 7] = l0 + l2 - 2.0 * l1

    if order == 2:
        return Nv, dNv * curl_factor

    # Cubic edge functions (Shape.cpp:787-792).
    Nv[:, 8] = (l1 * (l1 - 2.0 * l2))[:, None] * g2 \
        + (l2 * (2.0 * l1 - l2))[:, None] * g1
    Nv[:, 9] = (l0 * (l0 - 2.0 * l2))[:, None] * g2 \
        + (l2 * (2.0 * l0 - l2))[:, None] * g0
    Nv[:, 10] = (l0 * (l0 - 2.0 * l1))[:, None] * g1 \
        + (l1 * (2.0 * l0 - l1))[:, None] * g0
    # Cubic face bubble and face functions (Shape.cpp:793-804).
    Nv[:, 11] = (l1 * l0)[:, None] * g2 + (l0 * l2)[:, None] * g1 \
        + (l1 * l2)[:, None] * g0
    Nv[:, 12] = -(l1 * l0 * (l1 - 2.0 * l2))[:, None] * g2 \
        - (l0 * l2 * (2.0 * l1 - l2))[:, None] * g1 \
        + 3.0 * (l1 * l2 * (l1 - l2))[:, None] * g0
    Nv[:, 13] = -(l0 * l1 * (l0 - 2.0 * l2))[:, None] * g2 \
        - (l1 * l2 * (2.0 * l0 - l2))[:, None] * g0 \
        + 3.0 * (l0 * l2 * (l0 - l2))[:, None] * g1
    Nv[:, 14] = -(l0 * l2 * (l0 - 2.0 * l1))[:, None] * g1 \
        - (l1 * l2 * (2.0 * l0 - l1))[:, None] * g0 \
        + 3.0 * (l0 * l1 * (l0 - l1))[:, None] * g2
    # Only the face curls are filled (Shape.cpp:805-815).
    dNv[:, 12] = (4.0 * l1 * (l1 - 2.0 * l2)
                  - 4.0 * l2 * (2.0 * l1 - l2))
    dNv[:, 13] = (-4.0 * l0 * (l0 - 2.0 * l2)
                  + 4.0 * l2 * (2.0 * l0 - l2))
    dNv[:, 14] = (4.0 * l0 * (l0 - 2.0 * l1)
                  - 4.0 * l1 * (2.0 * l0 - l1))

    if order == 3:
        return Nv, dNv * curl_factor

    # p=4 (Shape.cpp:850-910): four edge types on {1,2}, {0,2}, {0,1}.
    # Shape.cpp zeroes Nv/dNv before filling this block; unlike orders 1-3 the
    # p=4 curls are already physical and are not scaled by the curl factor.
    Nv = np.zeros_like(Nv)
    dNv = np.zeros_like(dNv)
    ed = ((1, 2), (0, 2), (0, 1))
    for e, (i, j) in enumerate(ed):
        li, lj = l[:, i], l[:, j]
        gi, gj = dl[i], dl[j]
        # type 0: Whitney, type 1: grad(li*lj)
        Nv[:, e] = li[:, None] * gj - lj[:, None] * gi
        dNv[:, e] = 2.0 * (gi[0] * gj[1] - gi[1] * gj[0])
        Nv[:, 3 + e] = 4.0 * (li[:, None] * gj + lj[:, None] * gi)
        # type 2: grad(li*lj*(li-lj))
        dfi = 2.0 * li * lj - lj * lj
        dfj = li * li - 2.0 * li * lj
        Nv[:, 6 + e] = dfi[:, None] * gi + dfj[:, None] * gj
        # type 3: grad(li*lj*(li-lj)^2)
        dhi = 3.0 * li * li * lj - 4.0 * li * lj * lj + lj * lj * lj
        dhj = li * li * li - 4.0 * li * li * lj + 3.0 * li * lj * lj
        Nv[:, 9 + e] = dhi[:, None] * gi + dhj[:, None] * gj

    # Face functions, gradients embedded with z=0.
    g3 = np.zeros((nq, 3))
    g3[:, :2] = dl[0]
    g3a = g3.copy()
    g3[:, :2] = dl[1]
    g3b = g3.copy()
    g3[:, :2] = dl[2]
    g3c = g3.copy()
    for k in range(12):
        N, C = _face_p4(k, l0, l1, l2, g3a, g3b, g3c)
        Nv[:, 12 + k, 0] = N[:, 0]
        Nv[:, 12 + k, 1] = N[:, 1]
        dNv[:, 12 + k] = C[:, 2]

    return Nv, dNv


def shape_2d(
    order: int,
    points: np.ndarray,
    inv_j: np.ndarray | None = None,
) -> Shape2D:
    """Evaluate the cFES scalar and H(curl) bases on a triangle.

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
