"""Reference shape functions on the tetrahedron in the v0.1 C++ ordering.

This module transcribes ``cpp/src/shape.cpp`` (``case 3: // TETRAHEDRON``)
exactly, including the column ordering and the places where the reference C++
leaves derivative columns at zero.  It is the three-dimensional companion to
:mod:`fes.core.shape`.

Conventions:

* Reference tetrahedron: ``v0=(0,0,0)``, ``v1=(1,0,0)``, ``v2=(0,1,0)``,
  ``v3=(0,0,1)``.
* The C++ code fills ``cPos`` with ``(x, y, z, l3)`` where ``l3 = 1-x-y-z``;
  this module calls that ordering ``c`` (see :func:`_barycentric3`).
* Scalar (H1) basis ordering follows the reference C++ and is *not* uniform across
  orders: p=1 and p=4 put ``l3`` first, while p=2 and p=3 put it last.  The
  edge/face blocks follow ``tetEdges``/``tetFaces`` (see :mod:`fes.core.dof`).
* H(curl) basis: edge functions first, then face and volume bubbles.  The
  reference C++ only fills the derivative columns it needs; the remaining columns are
  zero (``dNv.fill(0)`` for p=3/4, and Armadillo's zero-initialising
  ``resize`` for p=2).  The zero columns are reproduced rather than corrected.

Arrays returned by :func:`shape_3d`:

``Ns``     shape ``(nq, nums)``    scalar basis values
``dNs``    shape ``(nq, nums, 3)`` scalar physical gradients
``Nv``     shape ``(nq, numv, 3)`` vector basis values
``dNv``    shape ``(nq, numv, 3)`` vector curls (zeros where the reference C++
                                   leaves them unset)
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

__all__ = [
    "Shape3D",
    "scalar_dof_count_3d",
    "shape_3d",
    "vector_dof_count_3d",
]


class Shape3D(NamedTuple):
    """Container for the 3D reference-basis arrays."""

    Ns: np.ndarray
    dNs: np.ndarray
    Nv: np.ndarray
    dNv: np.ndarray


#: H(curl) DOF counts per tetrahedron, exactly as ``Nv.resize`` in shape.cpp.
_VECTOR_DOF_COUNTS = {1: 6, 2: 20, 3: 45, 4: 84}


def scalar_dof_count_3d(order: int) -> int:
    """Number of scalar DOFs per tetrahedron: ``C(order+3, 3)``."""
    return (order + 1) * (order + 2) * (order + 3) // 6


def vector_dof_count_3d(order: int) -> int:
    """Number of H(curl) DOFs per tetrahedron (6, 20, 45, 84)."""
    try:
        return _VECTOR_DOF_COUNTS[int(order)]
    except (KeyError, ValueError):
        raise ValueError(
            f"3D shape functions are implemented for orders 1-4, got {order!r}"
        ) from None


def _barycentric3(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(l, dlam)`` for reference points.

    ``l`` has shape ``(nq, 4)`` and holds ``(l0, l1, l2, l3)`` with
    ``l0 = 1-x-y-z``.  ``dlam`` has shape ``(4, 3)`` and holds the *reference*
    gradients of the barycentric coordinates.
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    if pts.shape[1] != 3:
        raise ValueError("points must have shape (nq, 3)")
    l0 = 1.0 - pts[:, 0] - pts[:, 1] - pts[:, 2]
    l = np.column_stack([l0, pts[:, 0], pts[:, 1], pts[:, 2]])
    dlam_ref = np.array(
        [
            [-1.0, -1.0, -1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return l, dlam_ref


def _scalar_basis3(
    order: int, l: np.ndarray, dl: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Build the tetrahedral H1 basis in C++ order.

    ``dl`` holds the *reference* barycentric gradients; the caller maps the
    result to physical coordinates with ``dNs @ invJ.T``, matching
    ``dNs = cJac->invJ * dNs`` in the oracle.
    """
    # C++ ordering of the barycentric coordinates.
    c = np.column_stack([l[:, 1], l[:, 2], l[:, 3], l[:, 0]])
    g = np.stack([dl[1], dl[2], dl[3], dl[0]])

    nq = l.shape[0]
    nums = scalar_dof_count_3d(order)
    Ns = np.zeros((nq, nums))
    dNs = np.zeros((nq, nums, 3))

    def coord(i: int) -> tuple[np.ndarray, np.ndarray]:
        return c[:, i], np.broadcast_to(g[i], (nq, 3))

    def lin(i: int, a: float, b: float) -> tuple[np.ndarray, np.ndarray]:
        v, gr = coord(i)
        return a * v + b, a * gr

    def combo(terms: list[tuple[int, float]]) -> tuple[np.ndarray, np.ndarray]:
        value = np.zeros(nq)
        grad = np.zeros(3)
        for i, a in terms:
            value = value + a * c[:, i]
            grad = grad + a * g[i]
        return value, np.broadcast_to(grad, (nq, 3))

    def place(
        idx: int,
        factors: list[tuple[np.ndarray, np.ndarray]],
        scale: float = 1.0,
    ) -> None:
        """Place ``scale * prod(factors)`` and its reference gradient."""
        values = [factor[0] for factor in factors]
        grads = [factor[1] for factor in factors]
        value = np.ones(nq)
        for val in values:
            value = value * val
        Ns[:, idx] = scale * value

        grad = np.zeros((nq, 3))
        for i, gi in enumerate(grads):
            term = np.array(gi, dtype=float, copy=True)
            for j, val in enumerate(values):
                if j != i:
                    term = term * val[:, None]
            grad = grad + term
        dNs[:, idx, :] = scale * grad

    def vertex(idx: int, i: int, roots: tuple[tuple[float, float], ...]) -> None:
        """Nodal vertex function ``t * prod(a*t - b) / (degree!)``."""
        factors = [coord(i)] + [lin(i, a, -b) for a, b in roots]
        place(idx, factors, 1.0 / _factorial(len(roots)))

    if order == 1:
        for idx in range(4):  # l0, l1, l2, l3
            Ns[:, idx] = l[:, idx]
            dNs[:, idx, :] = np.broadcast_to(dl[idx], (nq, 3))
        return Ns, dNs

    if order == 2:
        for idx in range(4):
            vertex(idx, idx, ((2.0, 1.0),))
        edges = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
        for k, (a, b) in enumerate(edges):
            place(4 + k, [coord(a), coord(b)], 4.0)
        return Ns, dNs

    if order == 3:
        for idx in range(4):
            vertex(idx, idx, ((3.0, 1.0), (3.0, 2.0)))
        # Level 0: the "third" vertex scales the edge when the pair includes 3.
        level0 = (
            (0, 1, 0), (0, 2, 0), (3, 0, 3), (1, 2, 1), (3, 1, 3), (3, 2, 3)
        )
        level1 = (
            (0, 1, 1), (0, 2, 2), (3, 0, 0), (1, 2, 2), (3, 1, 1), (3, 2, 2)
        )
        for k, (a, b, s) in enumerate(level0):
            place(4 + k, [coord(a), coord(b), lin(s, 3.0, -1.0)], 4.5)
        for k, (a, b, s) in enumerate(level1):
            place(10 + k, [coord(a), coord(b), lin(s, 3.0, -1.0)], 4.5)
        faces = ((3, 1, 2), (3, 0, 2), (3, 0, 1), (0, 1, 2))
        for k, (a, b, d) in enumerate(faces):
            place(16 + k, [coord(a), coord(b), coord(d)], 27.0)
        return Ns, dNs

    # order == 4
    for idx, i in enumerate((3, 0, 1, 2)):
        vertex(idx, i, ((4.0, 1.0), (4.0, 2.0), (4.0, 3.0)))

    edges = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    for k, (a, b) in enumerate(edges):
        place(4 + k, [coord(a), coord(b)], 4.0)

    level_a = ((0, 1, 0), (0, 2, 0), (0, 3, 0), (1, 2, 1), (1, 3, 1), (2, 3, 2))
    level_b = ((0, 1, 1), (0, 2, 2), (0, 3, 3), (1, 2, 2), (1, 3, 3), (2, 3, 3))
    for k, (a, b, s) in enumerate(level_a):
        place(10 + k, [coord(a), coord(b), lin(s, 3.0, -1.0)], 4.5)
    for k, (a, b, s) in enumerate(level_b):
        place(16 + k, [coord(a), coord(b), lin(s, 3.0, -1.0)], 4.5)

    faces = ((1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2))
    for k, (a, b, d) in enumerate(faces):
        place(22 + k, [coord(a), coord(b), coord(d)], 27.0)

    # Face level A: l_i*l_j*l_k*(l_j-l_i).
    face_a = (
        (1, 2, 3, 2, 1),  # c2 - c1
        (0, 2, 3, 2, 0),  # c2 - c0
        (0, 1, 3, 1, 0),  # c1 - c0
        (0, 1, 2, 1, 0),  # c1 - c0
    )
    for k, (a, b, d, plus, minus) in enumerate(face_a):
        place(
            26 + k,
            [coord(a), coord(b), coord(d), combo([(plus, 1.0), (minus, -1.0)])],
        )

    # Face level B: l_i*l_j*l_k*(l_i+l_j-2*l_k).
    face_b = (
        (1, 2, 3, 3), (0, 2, 3, 3), (0, 1, 3, 3), (0, 1, 2, 2),
    )
    for k, (a, b, d, e) in enumerate(face_b):
        place(
            30 + k,
            [
                coord(a), coord(b), coord(d),
                combo([(a, 1.0), (b, 1.0), (e, -2.0)]),
            ],
        )

    place(34, [coord(0), coord(1), coord(2), coord(3)], 256.0)

    # Oracle quirk: ``dNs(0,26)`` in shape.cpp carries an extra ``l3`` factor
    # (the true x-derivative of ``l1*l2*l3*(l2-l1)`` omits it).  It is a typo
    # in the v0.1 source, but the parity gate compares against the oracle, so
    # reproduce the value verbatim.
    dNs[:, 26, 0] = -(c[:, 1] * c[:, 2] * c[:, 3] * (c[:, 2] - c[:, 1]))
    return Ns, dNs


def _factorial(n: int) -> float:
    out = 1.0
    for i in range(2, n + 1):
        out *= i
    return out


def _vector_basis3(
    order: int, l: np.ndarray, dl: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Build the tetrahedral H(curl) basis in C++ order.

    ``dl`` holds the *physical* barycentric gradients, because the oracle maps
    ``dNs`` through ``invJ`` before evaluating the vector functions.
    """
    c0, c1, c2, c3 = l[:, 1], l[:, 2], l[:, 3], l[:, 0]
    g0, g1, g2, g3 = dl[1], dl[2], dl[3], dl[0]

    nq = l.shape[0]
    numv = vector_dof_count_3d(order)
    Nv = np.zeros((nq, numv, 3))
    dNv = np.zeros((nq, numv, 3))

    x01 = np.cross(g0, g1)
    x02 = np.cross(g0, g2)
    x03 = np.cross(g0, g3)
    x12 = np.cross(g1, g2)
    x13 = np.cross(g1, g3)
    x23 = np.cross(g2, g3)

    def col(idx: int, value: np.ndarray) -> None:
        Nv[:, idx, :] = value

    def curl(idx: int, value: np.ndarray) -> None:
        dNv[:, idx, :] = value

    def sc(value: np.ndarray, vector: np.ndarray) -> np.ndarray:
        """Multiply a vector by a per-point scalar."""
        return value[:, None] * vector

    # Lowest-order edge functions and their curls.
    col(0, sc(c0, g1) - sc(c1, g0))
    col(1, sc(c0, g2) - sc(c2, g0))
    col(2, sc(c0, g3) - sc(c3, g0))
    col(3, sc(c1, g2) - sc(c2, g1))
    col(4, sc(c1, g3) - sc(c3, g1))
    col(5, sc(c2, g3) - sc(c3, g2))
    curl(0, 2.0 * x01)
    curl(1, 2.0 * x02)
    curl(2, 2.0 * x03)
    curl(3, 2.0 * x12)
    curl(4, 2.0 * x13)
    curl(5, 2.0 * x23)

    if order == 1:
        return Nv, dNv

    # Second-order edge functions (curls intentionally left at zero for p=2).
    col(6, 4.0 * (sc(c0, g1) + sc(c1, g0)))
    col(7, 4.0 * (sc(c0, g2) + sc(c2, g0)))
    col(8, 4.0 * (sc(c0, g3) + sc(c3, g0)))
    col(9, 4.0 * (sc(c1, g2) + sc(c2, g1)))
    col(10, 4.0 * (sc(c1, g3) + sc(c3, g1)))
    col(11, 4.0 * (sc(c2, g3) + sc(c3, g2)))

    # Second-order face functions.
    col(12, sc(c1 * c2, g3) - sc(c1 * c3, g2))
    col(13, sc(c1 * c2, g3) - sc(c2 * c3, g1))
    col(14, sc(c0 * c2, g3) - sc(c0 * c3, g2))
    col(15, sc(c0 * c2, g3) - sc(c2 * c3, g0))
    col(16, sc(c0 * c1, g3) - sc(c0 * c3, g1))
    col(17, sc(c0 * c1, g3) - sc(c1 * c3, g0))
    col(18, sc(c0 * c1, g2) - sc(c0 * c2, g1))
    col(19, sc(c0 * c1, g2) - sc(c1 * c2, g0))

    curl(12, sc(2.0 * c1, x23) + sc(c2, x13) - sc(c3, x12))
    curl(13, sc(c1, x23) + sc(2.0 * c2, x13) + sc(c3, x12))
    curl(14, sc(2.0 * c0, x23) + sc(c2, x03) - sc(c3, x02))
    curl(15, sc(c0, x23) + sc(2.0 * c2, x03) + sc(c3, x02))
    curl(16, sc(2.0 * c0, x13) + sc(c1, x03) - sc(c3, x01))
    curl(17, sc(c0, x13) + sc(2.0 * c1, x03) + sc(c3, x01))
    curl(18, sc(2.0 * c0, x12) + sc(c1, x02) - sc(c2, x01))
    curl(19, sc(c0, x12) + sc(2.0 * c1, x02) + sc(c2, x01))

    if order == 2:
        return Nv, dNv

    # Cubic edge functions.
    col(20, sc(c0 * (c0 - 2.0 * c1), g1) + sc(c1 * (-c1 + 2.0 * c0), g0))
    col(21, sc(c0 * (c0 - 2.0 * c2), g2) + sc(c2 * (-c2 + 2.0 * c0), g0))
    col(22, sc(c0 * (c0 - 2.0 * c3), g3) + sc(c3 * (-c3 + 2.0 * c0), g0))
    col(23, sc(c1 * (c1 - 2.0 * c2), g2) + sc(c2 * (-c2 + 2.0 * c1), g1))
    col(24, sc(c1 * (c1 - 2.0 * c3), g3) + sc(c3 * (-c3 + 2.0 * c1), g1))
    col(25, sc(c2 * (c2 - 2.0 * c3), g3) + sc(c3 * (-c3 + 2.0 * c2), g2))

    # Cubic face bubble and face functions.
    col(26, sc(c1 * c2, g3) + sc(c1 * c3, g2) + sc(c2 * c3, g1))
    col(27, -sc(c1 * c2 * (c2 - 2.0 * c3), g3)
            - sc(c1 * c3 * (-c3 + 2.0 * c2), g2)
            + sc(3.0 * c2 * c3 * (c2 - c3), g1))
    col(28, -sc(c1 * c2 * (c1 - 2.0 * c3), g3)
            - sc(c2 * c3 * (-c3 + 2.0 * c1), g1)
            + sc(3.0 * c1 * c3 * (c1 - c3), g2))
    col(29, -sc(c1 * c3 * (c1 - 2.0 * c2), g2)
            - sc(c2 * c3 * (-c2 + 2.0 * c1), g1)
            + sc(3.0 * c1 * c2 * (c1 - c2), g3))
    col(30, sc(c0 * c2, g3) + sc(c0 * c3, g2) + sc(c2 * c3, g0))
    col(31, -sc(c0 * c2 * (c2 - 2.0 * c3), g3)
            - sc(c0 * c3 * (-c3 + 2.0 * c2), g2)
            + sc(3.0 * c2 * c3 * (c2 - c3), g0))
    col(32, -sc(c0 * c2 * (c0 - 2.0 * c3), g3)
            - sc(c2 * c3 * (-c3 + 2.0 * c0), g0)
            + sc(3.0 * c0 * c3 * (c0 - c3), g2))
    col(33, -sc(c0 * c3 * (c0 - 2.0 * c2), g2)
            - sc(c2 * c3 * (-c2 + 2.0 * c0), g0)
            + sc(3.0 * c0 * c2 * (c0 - c2), g3))
    col(34, sc(c0 * c1, g3) + sc(c0 * c3, g1) + sc(c1 * c3, g0))
    col(35, -sc(c0 * c1 * (c1 - 2.0 * c3), g3)
            - sc(c0 * c3 * (-c3 + 2.0 * c1), g1)
            + sc(3.0 * c1 * c3 * (c1 - c3), g0))
    col(36, -sc(c0 * c1 * (c0 - 2.0 * c3), g3)
            - sc(c1 * c3 * (-c3 + 2.0 * c0), g0)
            + sc(3.0 * c0 * c3 * (c0 - c3), g1))
    col(37, -sc(c0 * c3 * (c0 - 2.0 * c1), g1)
            - sc(c1 * c3 * (-c1 + 2.0 * c0), g0)
            + sc(3.0 * c0 * c1 * (c0 - c1), g3))
    col(38, sc(c0 * c1, g2) + sc(c0 * c2, g1) + sc(c1 * c2, g0))
    col(39, -sc(c0 * c1 * (c1 - 2.0 * c2), g2)
            - sc(c0 * c2 * (-c2 + 2.0 * c1), g1)
            + sc(3.0 * c1 * c2 * (c1 - c2), g0))
    col(40, -sc(c0 * c1 * (c0 - 2.0 * c2), g2)
            - sc(c1 * c2 * (-c2 + 2.0 * c0), g0)
            + sc(3.0 * c0 * c2 * (c0 - c2), g1))
    col(41, -sc(c0 * c2 * (c0 - 2.0 * c1), g1)
            - sc(c1 * c2 * (-c1 + 2.0 * c0), g0)
            + sc(3.0 * c0 * c1 * (c0 - c1), g2))

    # Cubic volume functions.
    col(42, -sc(c0 * c1 * c2, g3) - sc(c1 * c0 * c3, g2)
            - sc(c0 * c2 * c3, g1) + sc(3.0 * c1 * c2 * c3, g0))
    col(43, -sc(c0 * c1 * c2, g3) - sc(c1 * c0 * c3, g2)
            - sc(c1 * c2 * c3, g0) + sc(3.0 * c0 * c2 * c3, g1))
    col(44, -sc(c0 * c1 * c2, g3) - sc(c0 * c2 * c3, g1)
            - sc(c1 * c2 * c3, g0) + sc(3.0 * c1 * c0 * c3, g2))

    # Cubic curls (only the columns the oracle fills).
    curl(27, -sc(4.0 * c2 * (c2 - 2.0 * c3), x13)
             - sc(4.0 * c3 * (2.0 * c2 - c3), x12))
    curl(28, -sc(4.0 * c1 * (c1 - 2.0 * c3), x23)
             + sc(4.0 * c3 * (2.0 * c1 - c3), x12))
    curl(29, sc(4.0 * c1 * (c1 - 2.0 * c2), x23)
            + sc(4.0 * c2 * (2.0 * c1 - c2), x13))
    curl(31, -sc(4.0 * c2 * (c2 - 2.0 * c3), x03)
             - sc(4.0 * c3 * (2.0 * c2 - c3), x02))
    curl(32, -sc(4.0 * c0 * (c0 - 2.0 * c3), x23)
             + sc(4.0 * c3 * (2.0 * c0 - c3), x02))
    curl(33, sc(4.0 * c0 * (c0 - 2.0 * c2), x23)
            + sc(4.0 * c2 * (2.0 * c0 - c2), x03))
    curl(35, -sc(4.0 * c1 * (c1 - 2.0 * c3), x03)
             - sc(4.0 * c3 * (2.0 * c1 - c3), x01))
    curl(36, -sc(4.0 * c0 * (c0 - 2.0 * c3), x13)
             + sc(4.0 * c3 * (2.0 * c0 - c3), x01))
    curl(37, sc(4.0 * c0 * (c0 - 2.0 * c1), x13)
            + sc(4.0 * c1 * (2.0 * c0 - c1), x03))
    curl(39, -sc(4.0 * c1 * (c1 - 2.0 * c2), x02)
             - sc(4.0 * c2 * (2.0 * c1 - c2), x01))
    curl(40, -sc(4.0 * c0 * (c0 - 2.0 * c2), x12)
             + sc(4.0 * c2 * (2.0 * c0 - c2), x01))
    curl(41, sc(4.0 * c0 * (c0 - 2.0 * c1), x12)
            + sc(4.0 * c1 * (2.0 * c0 - c1), x02))
    curl(42, -sc(4.0 * c1 * c2, x03) - sc(4.0 * c1 * c3, x02)
             - sc(4.0 * c2 * c3, x01))
    curl(43, -sc(4.0 * c0 * c2, x13) - sc(4.0 * c0 * c3, x12)
             + sc(4.0 * c2 * c3, x01))
    curl(44, -sc(4.0 * c1 * c0, x23) + sc(4.0 * c0 * c3, x12)
             + sc(4.0 * c1 * c3, x02))

    if order == 3:
        return Nv, dNv

    # Quartic edge functions (curl-free, curls left at zero).
    col(45, sc(c0 * c0 * (c0 - 3.0 * c1), g1) + sc(c1 * c1 * (c1 - 3.0 * c0), g0))
    col(46, sc(c0 * c0 * (c0 - 3.0 * c2), g2) + sc(c2 * c2 * (c2 - 3.0 * c0), g0))
    col(47, sc(c0 * c0 * (c0 - 3.0 * c3), g3) + sc(c3 * c3 * (c3 - 3.0 * c0), g0))
    col(48, sc(c1 * c1 * (c1 - 3.0 * c2), g2) + sc(c2 * c2 * (c2 - 3.0 * c1), g1))
    col(49, sc(c1 * c1 * (c1 - 3.0 * c3), g3) + sc(c3 * c3 * (c3 - 3.0 * c1), g1))
    col(50, sc(c2 * c2 * (c2 - 3.0 * c3), g3) + sc(c3 * c3 * (c3 - 3.0 * c2), g2))

    # Quartic face functions, face 0-1-2 (opposite vertex 3).
    col(51, sc(c0 * c1 * c2, g3) + sc(c1 * c2 * c3, g0)
            + sc(c0 * c2 * c3, g1) + sc(c0 * c1 * c3, g2))
    col(52, -sc(c0 * c1 * c2 * (c0 + c1), g3)
             + sc(c0 * c1 * c3 * (c0 + c1), g2)
             + sc(c0 * c2 * c3 * (c0 + c2), g1)
             + sc(c1 * c2 * c3 * (c1 + c2), g0))
    col(53, sc(c0 * c1 * c2 * (c2 - c1), g3)
            + sc(c0 * c1 * c3 * (c2 - c1), g2)
            + sc(c0 * c2 * c3 * (c0 - c3), g1)
            + sc(c1 * c2 * c3 * (c0 - c3), g0))
    col(54, sc(c0 * c1 * c2 * (c0 - c2), g3)
            + sc(c0 * c1 * c3 * (c0 - c2), g2)
            + sc(c0 * c2 * c3 * (c1 - c3), g1)
            + sc(c1 * c2 * c3 * (c1 - c3), g0))
    col(55, sc(c0 * c0 * c1 * c2, g3) + sc(c1 * c2 * c3 * c3, g0)
            + sc(c0 * c2 * c3 * c0, g1) + sc(c0 * c1 * c3 * c1, g2))
    col(56, sc(c0 * c1 * c1 * c2, g3) + sc(c1 * c2 * c3 * c2, g0)
            + sc(c0 * c2 * c3 * c3, g1) + sc(c0 * c1 * c3 * c0, g2))

    # Face 0-1-3 (opposite vertex 2).
    col(57, sc(c0 * c1 * c3, g2) + sc(c1 * c2 * c3, g0)
            + sc(c0 * c2 * c3, g1) + sc(c0 * c1 * c2, g3))
    col(58, -sc(c0 * c1 * c3 * (c0 + c1), g2)
             + sc(c0 * c1 * c2 * (c0 + c1), g3)
             + sc(c0 * c2 * c3 * (c0 + c3), g1)
             + sc(c1 * c2 * c3 * (c1 + c3), g0))
    col(59, sc(c0 * c1 * c3 * (c3 - c1), g2)
            + sc(c0 * c1 * c2 * (c3 - c1), g3)
            + sc(c0 * c2 * c3 * (c0 - c2), g1)
            + sc(c1 * c2 * c3 * (c0 - c2), g0))
    col(60, sc(c0 * c1 * c3 * (c0 - c3), g2)
            + sc(c0 * c1 * c2 * (c0 - c3), g3)
            + sc(c0 * c2 * c3 * (c1 - c2), g1)
            + sc(c1 * c2 * c3 * (c1 - c2), g0))
    col(61, sc(c0 * c0 * c1 * c3, g2) + sc(c1 * c2 * c3 * c2, g0)
            + sc(c0 * c2 * c3 * c0, g1) + sc(c0 * c1 * c2 * c1, g3))
    col(62, sc(c0 * c1 * c1 * c3, g2) + sc(c1 * c2 * c3 * c3, g0)
            + sc(c0 * c2 * c3 * c2, g1) + sc(c0 * c1 * c2 * c0, g3))

    # Face 0-2-3 (opposite vertex 1).
    col(63, sc(c0 * c2 * c3, g1) + sc(c1 * c2 * c3, g0)
            + sc(c0 * c1 * c3, g2) + sc(c0 * c1 * c2, g3))
    col(64, -sc(c0 * c2 * c3 * (c0 + c3), g1)
             + sc(c0 * c1 * c3 * (c0 + c3), g2)
             + sc(c0 * c1 * c2 * (c0 + c2), g3)
             + sc(c1 * c2 * c3 * (c2 + c3), g0))
    col(65, sc(c0 * c2 * c3 * (c3 - c2), g1)
            + sc(c0 * c1 * c3 * (c3 - c2), g2)
            + sc(c0 * c1 * c2 * (c0 - c1), g3)
            + sc(c1 * c2 * c3 * (c0 - c1), g0))
    col(66, sc(c0 * c2 * c3 * (c0 - c3), g1)
            + sc(c0 * c1 * c3 * (c0 - c3), g2)
            + sc(c0 * c1 * c2 * (c2 - c1), g3)
            + sc(c1 * c2 * c3 * (c2 - c1), g0))
    col(67, sc(c0 * c0 * c2 * c3, g1) + sc(c1 * c2 * c3 * c1, g0)
            + sc(c0 * c1 * c3 * c0, g2) + sc(c0 * c1 * c2 * c2, g3))
    col(68, sc(c0 * c2 * c2 * c3, g1) + sc(c1 * c2 * c3 * c3, g0)
            + sc(c0 * c1 * c3 * c2, g2) + sc(c0 * c1 * c2 * c0, g3))

    # Face 1-2-3 (opposite vertex 0).
    col(69, sc(c1 * c2 * c3, g0) + sc(c0 * c2 * c3, g1)
            + sc(c0 * c1 * c3, g2) + sc(c0 * c1 * c2, g3))
    col(70, -sc(c1 * c2 * c3 * (c2 + c3), g0)
             + sc(c0 * c2 * c3 * (c2 + c3), g1)
             + sc(c0 * c1 * c3 * (c1 + c3), g2)
             + sc(c0 * c1 * c2 * (c1 + c2), g3))
    col(71, sc(c1 * c2 * c3 * (c3 - c2), g0)
            + sc(c0 * c2 * c3 * (c0 - c1), g1)
            + sc(c0 * c1 * c3 * (c3 - c1), g2)
            + sc(c0 * c1 * c2 * (c2 - c1), g3))
    col(72, sc(c1 * c2 * c3 * (c1 - c3), g0)
            + sc(c0 * c2 * c3 * (c0 - c2), g1)
            + sc(c0 * c1 * c3 * (c1 - c3), g2)
            + sc(c0 * c1 * c2 * (c0 - c1), g3))
    col(73, sc(c1 * c1 * c2 * c3, g0) + sc(c0 * c2 * c3 * c2, g1)
            + sc(c0 * c1 * c3 * c1, g2) + sc(c0 * c1 * c2 * c3, g3))
    col(74, sc(c1 * c2 * c2 * c3, g0) + sc(c0 * c2 * c3 * c3, g1)
            + sc(c0 * c1 * c3 * c0, g2) + sc(c0 * c1 * c2 * c1, g3))

    # Quartic volume functions.
    col(75, -sc(c0 * c1 * c2 * c3, g0 - g1 + g2 - g3))
    col(76, sc(c0 * c1 * c2 * c3, g0 + g1 - g2 - g3))
    col(77, sc(c0 * c1 * c2 * c3, g0 - g1 - g2 + g3))
    col(78, sc(c0 * c1 * c2 * c3, 2.0 * g0 - g1 - g2))
    col(79, sc(c0 * c1 * c2 * c3, 2.0 * g1 - g0 - g2))
    col(80, sc(c0 * c1 * c2 * c3, 2.0 * g2 - g0 - g1))
    col(81, sc(c0 * c1 * c2 * c3, 2.0 * g3 - g0 - g1))
    col(82, sc(c0 * c1 * c2 * c3, g0 + g1 + g2 - 3.0 * g3))
    col(83, sc(c0 * c1 * c2 * c3, g0 + g2 - 2.0 * g1))

    # Quartic face curls.
    curl(51, sc(c1 * c2, x03) + sc(c0 * c2, x13) + sc(c0 * c1, x23))
    curl(52, sc(c2 * (c0 + c1), sc(c1, x03) + sc(c0, x13))
             + sc(c0 * c1, x03 + x13))
    curl(53, sc(c2 * (c2 - c1), sc(c1, x03) + sc(c0, x13))
             + sc(c0 * c1 * (c2 - c1), x23))
    curl(54, sc(c0 * c1 * (c0 - c2), x23))
    curl(55, sc(2.0 * c0 * c1 * c2, x03) + sc(c0 * c0 * c2, x13)
             + sc(c0 * c0 * c1, x23))
    curl(56, sc(c1 * c1 * c2, x03) + sc(2.0 * c0 * c1 * c2, x13)
             + sc(c0 * c1 * c1, x23))

    curl(57, sc(c0 * c1, x23))
    curl(58, sc(c0 * c1 * (c0 + c1), x23))
    curl(59, sc(c0 * c1 * (c3 - c1), x23))
    curl(60, sc(c0 * c1 * (c0 - c3), x23))
    curl(61, sc(2.0 * c0 * c1 * c3, x23) + sc(c0 * c0 * c3, x13)
             + sc(c1 * c1 * c3, x03))
    curl(62, sc(c0 * c1 * c3, x03) + sc(c0 * c1 * c3, x13))

    curl(63, sc(c0 * c2, x13))
    curl(64, sc(c0 * c2 * (c0 + c3), x13))
    curl(65, sc(c0 * c2 * (c3 - c2), x13))
    curl(66, sc(c0 * c2 * (c0 - c3), x13))
    curl(67, sc(2.0 * c0 * c2 * c3, x13))
    curl(68, sc(c0 * c2 * c2, x13) + sc(c0 * c2 * c3, x13))

    curl(69, sc(c1 * c2, x03))
    curl(70, sc(c1 * c2 * (c2 + c3), x03))
    curl(71, sc(c1 * c2 * (c3 - c2), x03))
    curl(72, sc(c1 * c2 * (c1 - c3), x03))
    curl(73, sc(2.0 * c1 * c2 * c3, x03))
    curl(74, sc(c1 * c2 * c2, x03))

    # Quartic volume curls.
    curl(75, sc(c1 * c2 * c3, x01) + sc(c0 * c2 * c3, x01))
    curl(76, sc(c1 * c2 * c3, x01) + sc(c0 * c2 * c3, x01))
    curl(77, -sc(c0 * c2 * c3, x01) + sc(c0 * c1 * c2, x03))
    curl(78, sc(c1 * c2 * c3, x12) + sc(c0 * c1 * c3, x02))
    curl(79, sc(c0 * c2 * c3, x02) + sc(c0 * c1 * c2, x13))
    curl(80, sc(c1 * c2 * c3, x02) + sc(c0 * c1 * c3, x13))
    curl(81, sc(c0 * c1 * c3, x02) + sc(c1 * c2 * c3, x02))
    curl(82, sc(c0 * c1 * c2, x03) + sc(c0 * c1 * c3, x03))
    curl(83, sc(c0 * c2 * c3, x02) + sc(c1 * c2 * c3, x01))

    return Nv, dNv


def shape_3d(
    order: int,
    points: np.ndarray,
    inv_j: np.ndarray | None = None,
) -> Shape3D:
    """Evaluate the v0.1 C++ scalar and H(curl) bases on a tetrahedron.

    Parameters
    ----------
    order : int
        Polynomial order, 1-4.
    points : ndarray, shape (nq, 3)
        Points in reference coordinates ``(x, y, z)``.
    inv_j : ndarray, shape (3, 3), optional
        Inverse of the C++ Jacobian ``J = dNs_ref @ geo``.  When omitted the
        reference tetrahedron is used (identity mapping).

    Returns
    -------
    Shape3D
        ``Ns`` (nq, nums), ``dNs`` (nq, nums, 3), ``Nv`` (nq, numv, 3),
        ``dNv`` (nq, numv, 3).

    Raises
    ------
    ValueError
        If ``order`` is outside 1-4.
    """
    if order not in (1, 2, 3, 4):
        raise ValueError(
            f"3D shape functions are implemented for orders 1-4, got {order}"
        )

    l, dlam_ref = _barycentric3(points)
    if inv_j is None:
        inv_j = np.eye(3)
    inv_j = np.asarray(inv_j, dtype=float)
    if inv_j.shape != (3, 3):
        raise ValueError("inv_j must have shape (3, 3)")

    # Physical barycentric gradients (rows), matching dNs = invJ * dNs_ref.
    dl = dlam_ref @ inv_j.T

    Ns, dNs_ref = _scalar_basis3(order, l, dlam_ref)
    dNs = dNs_ref @ inv_j.T
    Nv, dNv = _vector_basis3(order, l, dl)
    return Shape3D(Ns=Ns, dNs=dNs, Nv=Nv, dNv=dNv)
