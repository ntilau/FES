"""Reference shape functions on the tetrahedron in the cFES ordering.

This module transcribes ``cFES/src/Shape.cpp`` (``case 3: // TETRAHEDRON``)
exactly, including the column ordering and the places where the reference C++
leaves derivative columns at zero.

Conventions:

* Reference tetrahedron: ``v0=(0,0,0)``, ``v1=(1,0,0)``, ``v2=(0,1,0)``,
  ``v3=(0,0,1)``.
* The C++ code writes ``cPos = (x, y, z)`` and then appends
  ``cPos(3) = 1-x-y-z``.  In this module's ``(l0, l1, l2, l3)`` ordering that
  is ``cPos(0..3) = l1, l2, l3, l0``.
* Scalar (H1) basis ordering follows the reference C++ and is *not* uniform
  across orders.  Orders 1-3 use ``[l0, l1, l2, l3]`` for the vertices; order
  4 uses the ``cPos`` order ``[l1, l2, l3, l0]`` and groups its edge nodes by
  edge, while :mod:`fes.core.dof` groups them by type.
* H(curl) basis: edge functions first, then face and volume bubbles.  The
  reference C++ only fills the derivative columns it needs; the remaining
  columns are zero.  The zero columns are reproduced rather than corrected.
* The p=4 face functions come from :func:`fes.core.shape._face_p4`, shared
  with the 2D port element.

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

from .shape import _face_p4

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


#: H(curl) DOF counts per tetrahedron, exactly as ``Nv.resize`` in Shape.cpp.
_VECTOR_DOF_COUNTS = {1: 6, 2: 20, 3: 45, 4: 84}

#: Tetrahedron edges in the C++ vertex order (0,1),(0,2),(0,3),(1,2),...
_EDGES3 = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))

#: Tetrahedron faces in ``tetFaces`` order.
_FACES3 = ((1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2))


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
    order: int, l: np.ndarray, dlam_ref: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Build the tetrahedral H1 basis in C++ order.

    ``dlam_ref`` holds the *reference* barycentric gradients; the caller maps
    the result to physical coordinates with ``dNs @ invJ.T``, matching
    ``dNs = cJac->invJ * dNs`` in the oracle.
    """
    nq = l.shape[0]
    nums = scalar_dof_count_3d(order)
    Ns = np.zeros((nq, nums))
    dNs = np.zeros((nq, nums, 3))

    if order == 1:
        # Shape.cpp:428-441: Ns = (l0, l1, l2, l3).
        for idx in range(4):
            Ns[:, idx] = l[:, idx]
            dNs[:, idx, :] = np.broadcast_to(dlam_ref[idx], (nq, 3))
        return Ns, dNs

    def place(idx: int, values, grads) -> None:
        """Place ``prod(values)`` and its reference gradient."""
        value = np.ones(nq)
        for item in values:
            value = value * item
        Ns[:, idx] = value
        grad = np.zeros((nq, 3))
        for i, gi in enumerate(grads):
            term = np.array(gi, dtype=float, copy=True)
            for j, item in enumerate(values):
                if j != i:
                    term = term * item[:, None]
            grad = grad + term
        dNs[:, idx, :] = grad

    if order == 2:
        # Shape.cpp:443-482: vertices l0..l3, edges in _EDGES3 order.
        for idx in range(4):
            t = l[:, idx]
            Ns[:, idx] = t * (2.0 * t - 1.0)
            dNs[:, idx, :] = (4.0 * t - 1.0)[:, None] * dlam_ref[idx]
        for k, (a, b) in enumerate(_EDGES3):
            place(4 + k, (l[:, a], l[:, b]), (dlam_ref[a], dlam_ref[b]))
            Ns[:, 4 + k] *= 4.0
            dNs[:, 4 + k, :] *= 4.0
        return Ns, dNs

    if order == 3:
        # Shape.cpp:483-567.
        for idx in range(4):
            t = l[:, idx]
            Ns[:, idx] = t * (3.0 * t - 1.0) * (3.0 * t - 2.0) / 2.0
            dNs[:, idx, :] = ((27.0 * t * t - 18.0 * t + 2.0) / 2.0)[:, None] \
                * dlam_ref[idx]
        for k, (a, b) in enumerate(_EDGES3):
            la, lb = l[:, a], l[:, b]
            # type 0: 9/2 la lb (3 la - 1)
            place(4 + k,
                  (la, lb, 3.0 * la - 1.0),
                  (dlam_ref[a], dlam_ref[b], 3.0 * dlam_ref[a]))
            Ns[:, 4 + k] *= 4.5
            dNs[:, 4 + k, :] *= 4.5
            # type 1: 9/2 la lb (3 lb - 1)
            place(10 + k,
                  (la, lb, 3.0 * lb - 1.0),
                  (dlam_ref[a], dlam_ref[b], 3.0 * dlam_ref[b]))
            Ns[:, 10 + k] *= 4.5
            dNs[:, 10 + k, :] *= 4.5
        for k, (a, b, c) in enumerate(_FACES3):
            place(16 + k,
                  (l[:, a], l[:, b], l[:, c]),
                  (dlam_ref[a], dlam_ref[b], dlam_ref[c]))
            Ns[:, 16 + k] *= 27.0
            dNs[:, 16 + k, :] *= 27.0
        return Ns, dNs

    # order == 4 (Shape.cpp:568-693).  cPos order: c = (l1, l2, l3, l0).
    c = [l[:, 1], l[:, 2], l[:, 3], l[:, 0]]

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

    vertices = ((4, 0, 0, 0), (0, 4, 0, 0), (0, 0, 4, 0), (0, 0, 0, 4))
    ek = (
        (3, 1, 0, 0), (2, 2, 0, 0), (1, 3, 0, 0),
        (3, 0, 1, 0), (2, 0, 2, 0), (1, 0, 3, 0),
        (3, 0, 0, 1), (2, 0, 0, 2), (1, 0, 0, 3),
        (0, 3, 1, 0), (0, 2, 2, 0), (0, 1, 3, 0),
        (0, 3, 0, 1), (0, 2, 0, 2), (0, 1, 0, 3),
        (0, 0, 3, 1), (0, 0, 2, 2), (0, 0, 1, 3),
    )
    fk = (
        (2, 1, 1, 0), (1, 2, 1, 0), (1, 1, 2, 0),
        (2, 1, 0, 1), (1, 2, 0, 1), (1, 1, 0, 2),
        (2, 0, 1, 1), (1, 0, 2, 1), (1, 0, 1, 2),
        (0, 2, 1, 1), (0, 1, 2, 1), (0, 1, 1, 2),
    )
    tables = (vertices, ek, fk, ((1, 1, 1, 1),))

    for idx in range(35):
        if idx < 4:
            k = tables[0][idx]
        elif idx < 22:
            k = ek[idx - 4]
        elif idx < 34:
            k = fk[idx - 22]
        else:
            k = tables[3][0]
        t = [4.0 * c[j] for j in range(4)]
        Lv = [Lf[k[j]](t[j]) for j in range(4)]
        dLv = [dLf[k[j]](t[j]) for j in range(4)]
        Ns[:, idx] = Lv[0] * Lv[1] * Lv[2] * Lv[3]
        dli = [4.0 * dLv[j] * Lv[(j + 1) % 4] * Lv[(j + 2) % 4] * Lv[(j + 3) % 4]
               for j in range(4)]
        # cPos(3) = 1 - cPos(0) - cPos(1) - cPos(2).
        dNs[:, idx, 0] = dli[0] - dli[3]
        dNs[:, idx, 1] = dli[1] - dli[3]
        dNs[:, idx, 2] = dli[2] - dli[3]
    return Ns, dNs


def _vector_basis3(
    order: int, l: np.ndarray, dl: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Build the tetrahedral H(curl) basis in C++ order.

    ``dl`` holds the *physical* barycentric gradients, because the oracle maps
    ``dNs`` through ``invJ`` before evaluating the vector functions.
    """
    n0, n1, n2, n3 = l[:, 0], l[:, 1], l[:, 2], l[:, 3]
    G = [dl[0], dl[1], dl[2], dl[3]]
    nq = l.shape[0]
    numv = vector_dof_count_3d(order)
    Nv = np.zeros((nq, numv, 3))
    dNv = np.zeros((nq, numv, 3))

    def X(i: int, j: int) -> np.ndarray:
        return np.cross(G[i], G[j])

    def sc(value: np.ndarray, vector: np.ndarray) -> np.ndarray:
        return value[:, None] * vector

    def col(idx: int, value: np.ndarray) -> None:
        Nv[:, idx, :] = value

    def curl(idx: int, value: np.ndarray) -> None:
        dNv[:, idx, :] = value

    # Lowest-order edge functions (Shape.cpp:125-137 / 180-191).
    for k, (i, j) in enumerate(_EDGES3):
        col(k, sc(l[:, i], G[j]) - sc(l[:, j], G[i]))
        curl(k, 2.0 * X(i, j))

    if order == 1:
        return Nv, dNv

    # Second-order edge functions: 4*grad(l_i*l_j).
    for k, (i, j) in enumerate(_EDGES3):
        col(6 + k, 4.0 * (sc(l[:, i], G[j]) + sc(l[:, j], G[i])))

    # Second-order face functions (Shape.cpp:138-145).
    col(12, sc(n1 * n2, G[3]) - sc(n1 * n3, G[2]))
    col(13, sc(n1 * n2, G[3]) - sc(n2 * n3, G[1]))
    col(14, sc(n0 * n2, G[3]) - sc(n0 * n3, G[2]))
    col(15, sc(n0 * n2, G[3]) - sc(n2 * n3, G[0]))
    col(16, sc(n0 * n1, G[3]) - sc(n0 * n3, G[1]))
    col(17, sc(n0 * n1, G[3]) - sc(n1 * n3, G[0]))
    col(18, sc(n0 * n1, G[2]) - sc(n0 * n2, G[1]))
    col(19, sc(n0 * n1, G[2]) - sc(n1 * n2, G[0]))

    curl(12, sc(2.0 * n1, X(2, 3)) + sc(n2, X(1, 3)) - sc(n3, X(1, 2)))
    curl(13, sc(n1, X(2, 3)) + sc(2.0 * n2, X(1, 3)) + sc(n3, X(1, 2)))
    curl(14, sc(2.0 * n0, X(2, 3)) + sc(n2, X(0, 3)) - sc(n3, X(0, 2)))
    curl(15, sc(n0, X(2, 3)) + sc(2.0 * n2, X(0, 3)) + sc(n3, X(0, 2)))
    curl(16, sc(2.0 * n0, X(1, 3)) + sc(n1, X(0, 3)) - sc(n3, X(0, 1)))
    curl(17, sc(n0, X(1, 3)) + sc(2.0 * n1, X(0, 3)) + sc(n3, X(0, 1)))
    curl(18, sc(2.0 * n0, X(1, 2)) + sc(n1, X(0, 2)) - sc(n2, X(0, 1)))
    curl(19, sc(n0, X(1, 2)) + sc(2.0 * n1, X(0, 2)) + sc(n2, X(0, 1)))

    if order == 2:
        return Nv, dNv

    # Cubic edge functions (Shape.cpp:201-212).
    col(20, sc(n0 * (n0 - 2.0 * n1), G[1]) + sc(n1 * (-n1 + 2.0 * n0), G[0]))
    col(21, sc(n0 * (n0 - 2.0 * n2), G[2]) + sc(n2 * (-n2 + 2.0 * n0), G[0]))
    col(22, sc(n0 * (n0 - 2.0 * n3), G[3]) + sc(n3 * (-n3 + 2.0 * n0), G[0]))
    col(23, sc(n1 * (n1 - 2.0 * n2), G[2]) + sc(n2 * (-n2 + 2.0 * n1), G[1]))
    col(24, sc(n1 * (n1 - 2.0 * n3), G[3]) + sc(n3 * (-n3 + 2.0 * n1), G[1]))
    col(25, sc(n2 * (n2 - 2.0 * n3), G[3]) + sc(n3 * (-n3 + 2.0 * n2), G[2]))

    # Face 1-2-3 (Shape.cpp:213-224).
    col(26, sc(n1 * n2, G[3]) + sc(n1 * n3, G[2]) + sc(n2 * n3, G[1]))
    col(27, -sc(n1 * n2 * (n2 - 2.0 * n3), G[3])
            - sc(n1 * n3 * (-n3 + 2.0 * n2), G[2])
            + sc(3.0 * n2 * n3 * (n2 - n3), G[1]))
    col(28, -sc(n1 * n2 * (n1 - 2.0 * n3), G[3])
            - sc(n2 * n3 * (-n3 + 2.0 * n1), G[1])
            + sc(3.0 * n1 * n3 * (n1 - n3), G[2]))
    col(29, -sc(n1 * n3 * (n1 - 2.0 * n2), G[2])
            - sc(n2 * n3 * (-n2 + 2.0 * n1), G[1])
            + sc(3.0 * n1 * n2 * (n1 - n2), G[3]))
    # Face 0-2-3 (Shape.cpp:225-236).
    col(30, sc(n0 * n2, G[3]) + sc(n0 * n3, G[2]) + sc(n2 * n3, G[0]))
    col(31, -sc(n0 * n2 * (n2 - 2.0 * n3), G[3])
            - sc(n0 * n3 * (-n3 + 2.0 * n2), G[2])
            + sc(3.0 * n2 * n3 * (n2 - n3), G[0]))
    col(32, -sc(n0 * n2 * (n0 - 2.0 * n3), G[3])
            - sc(n2 * n3 * (-n3 + 2.0 * n0), G[0])
            + sc(3.0 * n0 * n3 * (n0 - n3), G[2]))
    col(33, -sc(n0 * n3 * (n0 - 2.0 * n2), G[2])
            - sc(n2 * n3 * (-n2 + 2.0 * n0), G[0])
            + sc(3.0 * n0 * n2 * (n0 - n2), G[3]))
    # Face 0-1-3 (Shape.cpp:237-248).
    col(34, sc(n0 * n1, G[3]) + sc(n0 * n3, G[1]) + sc(n1 * n3, G[0]))
    col(35, -sc(n0 * n1 * (n1 - 2.0 * n3), G[3])
            - sc(n0 * n3 * (-n3 + 2.0 * n1), G[1])
            + sc(3.0 * n1 * n3 * (n1 - n3), G[0]))
    col(36, -sc(n0 * n1 * (n0 - 2.0 * n3), G[3])
            - sc(n1 * n3 * (-n3 + 2.0 * n0), G[0])
            + sc(3.0 * n0 * n3 * (n0 - n3), G[1]))
    col(37, -sc(n0 * n3 * (n0 - 2.0 * n1), G[1])
            - sc(n1 * n3 * (-n1 + 2.0 * n0), G[0])
            + sc(3.0 * n0 * n1 * (n0 - n1), G[3]))
    # Face 0-1-2 (Shape.cpp:249-260).
    col(38, sc(n0 * n1, G[2]) + sc(n0 * n2, G[1]) + sc(n1 * n2, G[0]))
    col(39, -sc(n0 * n1 * (n1 - 2.0 * n2), G[2])
            - sc(n0 * n2 * (-n2 + 2.0 * n1), G[1])
            + sc(3.0 * n1 * n2 * (n1 - n2), G[0]))
    col(40, -sc(n0 * n1 * (n0 - 2.0 * n2), G[2])
            - sc(n1 * n2 * (-n2 + 2.0 * n0), G[0])
            + sc(3.0 * n0 * n2 * (n0 - n2), G[1]))
    col(41, -sc(n0 * n2 * (n0 - 2.0 * n1), G[1])
            - sc(n1 * n2 * (-n1 + 2.0 * n0), G[0])
            + sc(3.0 * n0 * n1 * (n0 - n1), G[2]))
    # Volume functions (Shape.cpp:261-272).
    col(42, -sc(n0 * n1 * n2, G[3]) - sc(n1 * n0 * n3, G[2])
            - sc(n0 * n2 * n3, G[1]) + sc(3.0 * n1 * n2 * n3, G[0]))
    col(43, -sc(n0 * n1 * n2, G[3]) - sc(n1 * n0 * n3, G[2])
            - sc(n1 * n2 * n3, G[0]) + sc(3.0 * n0 * n2 * n3, G[1]))
    col(44, -sc(n0 * n1 * n2, G[3]) - sc(n0 * n2 * n3, G[1])
            - sc(n1 * n2 * n3, G[0]) + sc(3.0 * n1 * n0 * n3, G[2]))

    # Cubic curls (only the columns the oracle fills, Shape.cpp:303-335).
    curl(27, -sc(4.0 * n2 * (n2 - 2.0 * n3), X(1, 3))
             - sc(4.0 * n3 * (2.0 * n2 - n3), X(1, 2)))
    curl(28, -sc(4.0 * n1 * (n1 - 2.0 * n3), X(2, 3))
             + sc(4.0 * n3 * (2.0 * n1 - n3), X(1, 2)))
    curl(29, sc(4.0 * n1 * (n1 - 2.0 * n2), X(2, 3))
            + sc(4.0 * n2 * (2.0 * n1 - n2), X(1, 3)))
    curl(31, -sc(4.0 * n2 * (n2 - 2.0 * n3), X(0, 3))
             - sc(4.0 * n3 * (2.0 * n2 - n3), X(0, 2)))
    curl(32, -sc(4.0 * n0 * (n0 - 2.0 * n3), X(2, 3))
             + sc(4.0 * n3 * (2.0 * n0 - n3), X(0, 2)))
    curl(33, sc(4.0 * n0 * (n0 - 2.0 * n2), X(2, 3))
            + sc(4.0 * n2 * (2.0 * n0 - n2), X(0, 3)))
    curl(35, -sc(4.0 * n1 * (n1 - 2.0 * n3), X(0, 3))
             - sc(4.0 * n3 * (2.0 * n1 - n3), X(0, 1)))
    curl(36, -sc(4.0 * n0 * (n0 - 2.0 * n3), X(1, 3))
             + sc(4.0 * n3 * (2.0 * n0 - n3), X(0, 1)))
    curl(37, sc(4.0 * n0 * (n0 - 2.0 * n1), X(1, 3))
            + sc(4.0 * n1 * (2.0 * n0 - n1), X(0, 3)))
    curl(39, -sc(4.0 * n1 * (n1 - 2.0 * n2), X(0, 2))
             - sc(4.0 * n2 * (2.0 * n1 - n2), X(0, 1)))
    curl(40, -sc(4.0 * n0 * (n0 - 2.0 * n2), X(1, 2))
             + sc(4.0 * n2 * (2.0 * n0 - n2), X(0, 1)))
    curl(41, sc(4.0 * n0 * (n0 - 2.0 * n1), X(1, 2))
            + sc(4.0 * n1 * (2.0 * n0 - n1), X(0, 2)))
    curl(42, -sc(4.0 * n1 * n2, X(0, 3)) - sc(4.0 * n1 * n3, X(0, 2))
             - sc(4.0 * n2 * n3, X(0, 1)))
    curl(43, -sc(4.0 * n0 * n2, X(1, 3)) - sc(4.0 * n0 * n3, X(1, 2))
             + sc(4.0 * n2 * n3, X(0, 1)))
    curl(44, -sc(4.0 * n1 * n0, X(2, 3)) + sc(4.0 * n0 * n3, X(1, 2))
             + sc(4.0 * n1 * n3, X(0, 2)))

    if order == 3:
        return Nv, dNv

    # p=4 (Shape.cpp:337-424).  Shape.cpp zeroes Nv/dNv before filling.
    Nv = np.zeros_like(Nv)
    dNv = np.zeros_like(dNv)
    tet_edge_p4 = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    for e, (i, j) in enumerate(tet_edge_p4):
        li, lj = l[:, i], l[:, j]
        col(e, sc(li, G[j]) - sc(lj, G[i]))
        curl(e, 2.0 * X(i, j))
        col(6 + e, 4.0 * (sc(li, G[j]) + sc(lj, G[i])))
        dfi = 2.0 * li * lj - lj * lj
        dfj = li * li - 2.0 * li * lj
        col(28 + e, sc(dfi, G[i]) + sc(dfj, G[j]))
        dhi = 3.0 * li * li * lj - 4.0 * li * lj * lj + lj * lj * lj
        dhj = li * li * li - 4.0 * li * li * lj + 3.0 * li * lj * lj
        col(34 + e, sc(dhi, G[i]) + sc(dhj, G[j]))

    tet_face_vert_p4 = ((1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2))
    for f, (a, b, c) in enumerate(tet_face_vert_p4):
        for k in range(12):
            N3, C3 = _face_p4(k, l[:, a], l[:, b], l[:, c],
                              G[a], G[b], G[c])
            out = (12 + 4 * f + k) if k < 4 else (40 + 8 * f + (k - 4))
            col(out, N3)
            curl(out, C3)

    b4 = n0 * n1 * n2 * n3
    gb4 = (sc(n1 * n2 * n3, G[0]) + sc(n0 * n2 * n3, G[1])
           + sc(n0 * n1 * n3, G[2]) + sc(n0 * n1 * n2, G[3]))
    for a in range(3):
        other = [t for t in range(4) if t != a]
        o0, o1, o2 = other
        d = (sc(l[:, o1] * l[:, o2], G[o0])
             + sc(l[:, o0] * l[:, o2], G[o1])
             + sc(l[:, o0] * l[:, o1], G[o2]))
        col(72 + a, sc(l[:, a], d))
        curl(72 + a, np.cross(G[a], d))
    for a in range(3):
        col(75 + a, sc(b4, G[a]))
        curl(75 + a, np.cross(gb4, G[a]))
    col(78, gb4)
    for a in range(3):
        col(79 + a, sc(l[:, a], gb4) + sc(b4, G[a]))
    W01 = sc(n0, G[1]) - sc(n1, G[0])
    W02 = sc(n0, G[2]) - sc(n2, G[0])
    col(82, sc(b4, W01))
    curl(82, np.cross(gb4, W01) + sc(2.0 * b4, X(0, 1)))
    col(83, sc(b4, W02))
    curl(83, np.cross(gb4, W02) + sc(2.0 * b4, X(0, 2)))

    return Nv, dNv


def shape_3d(
    order: int,
    points: np.ndarray,
    inv_j: np.ndarray | None = None,
) -> Shape3D:
    """Evaluate the cFES scalar and H(curl) bases on a tetrahedron.

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
