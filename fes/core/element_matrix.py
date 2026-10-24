"""Per-element matrices in the v0.1 C++ conventions.

Line-by-line port of the first two ``ele_mat`` constructors in
``cpp/src/element_matrix.cpp``:

* the three-dimensional H(curl) bulk element, which produces the stiffness
  ``S``, the mass ``T`` and the conductivity term ``Z``;
* the *embedded* two-dimensional element used on boundary faces, which builds
  a local 2D coordinate system from the face normal and returns ``St``, ``Tt``,
  ``Sz``, ``Tz``, ``G`` and ``STt``;
* the incident-plane-wave face vector (``+einc``), which integrates the
  tangential incident field against the H(curl) basis.

Both constructors run their quadrature through ``quad(p_ord + 1)``; v0.1 only
implements orders 1-4, so orders 1-3 are reachable and order 4 throws before
assembly starts.  That ceiling is reproduced rather than worked around.

Conventions carried over verbatim from the reference C++:

* ``J = dNs_ref @ geo`` and ``detJ = |det J|``; the quadrature weights already
  carry the simplex volume factor.
* The bulk matrices are accumulated over real H(curl) products and then
  scaled by the material, in the same order as the C++ source:
  ``Z = detJ * T_raw * (i z0 sigma)`` uses the *unscaled* mass matrix.
* ``epsr2 = -tand * epsr``; it does not depend on frequency in v0.1.
* On a face, ``n`` is the outward normal of the embedded 2D frame: it is built
  from ``cross(v1, v2)`` and flipped to point away from ``int_node``.
* The face ``G`` and ``STt`` blocks keep the oracle's orientation: ``G`` is
  ``Nv^T dNs`` (vector rows against scalar columns) and ``STt`` is the
  diagonal of the vector mass matrix.

The committed fixture pins every matrix for orders 1-3 on a skewed tetrahedron and its first face.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Optional

import numpy as np

from ..constants import Z0
from .quadrature import quadrature_2d, quadrature_3d
from .shape import shape_2d
from .shape3d import shape_3d

__all__ = [
    "MAX_ASSEMBLY_ORDER",
    "Element3D",
    "FaceElement2D",
    "Material",
    "MixedElement2D",
    "einc_face_vector",
    "element_matrices_2d_mixed",
    "element_matrices_3d",
    "element_matrices_face",
    "face_axes",
    "face_frame",
    "tetrahedron_jacobian",
    "triangle_jacobian",
]

#: Highest order the assembled system can reach: ``quad(p_ord + 1)`` needs a
#: rule of order ``p_ord + 1``, and v0.1 has rules 1-4 only.
MAX_ASSEMBLY_ORDER = 3


@dataclass(frozen=True)
class Material:
    """The ``mtrl`` fields the element matrices read.

    ``Solid`` records ``tand``, not ``epsr2``; the oracle derives
    ``epsr2 = -tand * epsr`` once, when the material is created.
    """

    epsr: float = 1.0
    mur: float = 1.0
    sigma: float = 0.0
    tand: float = 0.0

    @property
    def epsr2(self) -> float:
        return -self.tand * self.epsr

    @classmethod
    def from_solid(cls, solid) -> "Material":
        """Build from a :class:`fes.mesh.poly_header.Solid` (or ``None``)."""
        if solid is None:
            return cls()
        return cls(
            epsr=float(solid.epsr),
            mur=float(solid.mur),
            sigma=float(solid.sigma),
            tand=float(solid.tand),
        )


class Element3D(NamedTuple):
    """Bulk tetrahedral matrices, all ``(numv, numv)`` and complex."""

    S: np.ndarray
    T: np.ndarray
    Z: np.ndarray


class FaceElement2D(NamedTuple):
    """Embedded-face matrices, all complex as the oracle stores them."""

    St: np.ndarray
    Tt: np.ndarray
    Sz: np.ndarray
    Tz: np.ndarray
    G: np.ndarray
    STt: np.ndarray


class MixedElement2D(NamedTuple):
    """Triangle matrices from the ``ele_mat`` ``cDim == 2`` branch.

    ``St``/``Tt`` are the H(curl) stiffness and mass, ``Sz``/``Tz`` the H1
    counterparts, and ``G`` couples the vector trial functions to the scalar
    ones (``Nv^T dNs``).  Rows/columns follow the ``shape_2d`` basis order.
    """

    St: np.ndarray
    Tt: np.ndarray
    Sz: np.ndarray
    Tz: np.ndarray
    G: np.ndarray


#: Reference gradients of the barycentric coordinates, matching the ``dNs``
#: matrices the C++ ``jacobian`` constructor hard-codes.
_DNS_REF_3D = np.array(
    [
        [-1.0, 1.0, 0.0, 0.0],
        [-1.0, 0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0, 1.0],
    ]
)
_DNS_REF_2D = np.array(
    [
        [-1.0, 1.0, 0.0],
        [-1.0, 0.0, 1.0],
    ]
)


def _checked_order(order: int) -> int:
    order = int(order)
    if order < 1 or order > MAX_ASSEMBLY_ORDER:
        raise ValueError(
            f"element assembly supports orders 1-{MAX_ASSEMBLY_ORDER}; "
            f"quad(order + 1) is not implemented for {order}"
        )
    return order


def _jacobian(dns_ref: np.ndarray, geo: np.ndarray):
    """Return ``(|detJ|, invJ)`` for ``J = dNs_ref @ geo``."""
    jac = dns_ref @ np.asarray(geo, dtype=float)
    return abs(float(np.linalg.det(jac))), np.linalg.inv(jac)


def triangle_jacobian(geo: np.ndarray) -> tuple:
    """Return the C++ ``jacobian(2, geo)`` pair ``(detJ, invJ)``."""
    geo = np.asarray(geo, dtype=float)
    if geo.shape != (3, 2):
        raise ValueError("a triangle needs three nodes with two coordinates")
    return _jacobian(_DNS_REF_2D, geo)


def tetrahedron_jacobian(geo: np.ndarray) -> tuple:
    """Return the C++ ``jacobian(3, geo)`` pair ``(detJ, invJ)``."""
    geo = np.asarray(geo, dtype=float)
    if geo.shape != (4, 3):
        raise ValueError("a tetrahedron needs four nodes with three coordinates")
    return _jacobian(_DNS_REF_3D, geo)


def face_axes(geo: np.ndarray, int_node: np.ndarray) -> tuple:
    """Return the embedded-face frame ``(u, v, n, c_geo2)``.

    Reproduces the first ``ele_mat`` face constructor: ``u`` follows the first
    edge, ``v`` completes an orthonormal frame in the face plane, and the
    normal is flipped towards ``int_node`` so that ``detJ`` is always the
    outward-side area.  ``c_geo2`` is the ``(3, 2)`` geometry whose first two
    rows are zero; ``n`` is the flipped unit normal.

    The radiation post-processor needs the same normal, so the frame is built
    once here for both the face matrices and ``radiation.cpp``.
    """
    geo = np.asarray(geo, dtype=float)
    if geo.shape != (3, 3):
        raise ValueError("a face needs three nodes with three coordinates")
    v0, v1, v2 = geo[0].copy(), geo[1].copy(), geo[2].copy()
    v_in = (v0 + v1 + v2) / 3.0 - np.asarray(int_node, dtype=float)
    v_in = v_in / np.linalg.norm(v_in)
    v1 = v1 - v0
    v2 = v2 - v0
    u = v1 / np.linalg.norm(v1)
    n = np.cross(v1, v2)
    n = n * np.dot(n, v_in)
    n = n / np.linalg.norm(n)
    v = np.cross(n, u)

    c_geo2 = np.zeros((3, 2))
    c_geo2[1, 0] = np.dot(v1, u)
    c_geo2[2, 0] = np.dot(v2, u)
    c_geo2[2, 1] = np.dot(v2, v)
    return u, v, n, c_geo2


def face_frame(geo: np.ndarray, int_node: np.ndarray) -> np.ndarray:
    """Return the ``(3, 2)`` embedded geometry the oracle builds for a face."""
    return face_axes(geo, int_node)[3]


def element_matrices_3d(
    order: int,
    geo: np.ndarray,
    material: Optional[Material] = None,
) -> Element3D:
    """Assemble ``S``, ``T`` and ``Z`` for one tetrahedron.

    ``geo`` is ``(4, 3)``; *material* defaults to vacuum.  The returned
    matrices are already scaled by ``detJ`` and the material, exactly as
    ``ele_mat(p, 3, geo, quad(p+1), mtrl, shape::hcurl)`` does.
    """
    order = _checked_order(order)
    material = material or Material()
    geo = np.asarray(geo, dtype=float)

    det_j, inv_j = tetrahedron_jacobian(geo)
    points, weights = quadrature_3d(order + 1)
    numv = shape_3d(order, points[:1], inv_j).Nv.shape[1]

    s_raw = np.zeros((numv, numv))
    t_raw = np.zeros((numv, numv))
    for iq in range(points.shape[0]):
        basis = shape_3d(order, points[iq : iq + 1], inv_j)
        nv = basis.Nv[0]  # (numv, 3)
        dnv = basis.dNv[0]  # (numv, 3)
        s_raw += (dnv @ dnv.T) * weights[iq]
        t_raw += (nv @ nv.T) * weights[iq]

    # Z is built from the unscaled mass matrix, then T is scaled.
    z = (det_j * t_raw) * complex(0.0, Z0 * material.sigma)
    s = s_raw * complex(det_j / material.mur)
    t = t_raw * complex(det_j * material.epsr, det_j * material.epsr2)
    return Element3D(S=s, T=t, Z=z)


def element_matrices_face(
    order: int,
    geo: np.ndarray,
    int_node: np.ndarray,
) -> FaceElement2D:
    """Assemble the embedded-face matrices for one triangular boundary face.

    ``geo`` is ``(3, 3)`` and *int_node* is a point on the tetrahedron side of
    the face (the oracle passes the opposite vertex of the adjacent
    tetrahedron).
    """
    order = _checked_order(order)
    c_geo2 = face_frame(geo, int_node)
    det_j, inv_j = _jacobian(_DNS_REF_2D, c_geo2)
    points, weights = quadrature_2d(order + 1)

    nums = shape_2d(order, points[:1], inv_j).Ns.shape[1]
    numv = shape_2d(order, points[:1], inv_j).Nv.shape[1]
    st = np.zeros((numv, numv))
    tt = np.zeros((numv, numv))
    sz = np.zeros((nums, nums))
    tz = np.zeros((nums, nums))
    g = np.zeros((numv, nums))
    stt = np.zeros((numv, numv))
    for iq in range(points.shape[0]):
        basis = shape_2d(order, points[iq : iq + 1], inv_j)
        nv = basis.Nv[0]  # (numv, 2)
        dnv = basis.dNv[0]  # (numv,)
        ns = basis.Ns[0]  # (nums,)
        dns = basis.dNs[0]  # (nums, 2)
        weight = weights[iq]
        st += np.outer(dnv, dnv) * weight
        tt += (nv @ nv.T) * weight
        sz += (dns @ dns.T) * weight
        tz += np.outer(ns, ns) * weight
        g += (nv @ dns.T) * weight
        # arma::diagmat keeps only the diagonal of the vector mass matrix.
        stt += np.diag(np.sum(nv * nv, axis=1)) * weight

    scale = complex(det_j)
    st = st * scale
    tt = tt * scale
    sz = sz * scale
    tz = tz * scale
    g = g * scale
    stt = stt * scale
    return FaceElement2D(St=st, Tt=tt, Sz=sz, Tz=tz, G=g, STt=stt)


def einc_face_vector(
    order: int,
    geo: np.ndarray,
    int_node: np.ndarray,
    k_einc: np.ndarray,
    pol_einc: np.ndarray,
) -> np.ndarray:
    """Incident-plane-wave right-hand-side vector for one radiation face.

    Line-by-line port of the third ``ele_mat`` constructor in
    ``cpp/src/element_matrix.cpp``.  The returned ``(numv,)`` vector is the
    face integral ``Nv^T E_inc`` that ``assemble_em_e_fd`` adds to ``B`` with
    an extra ``i k0`` factor.

    Conventions carried over verbatim:

    * ``int_node`` is *not* normalized before it orients the normal (unlike
      the first face constructor); only the sign of ``dot(n, vIn)`` survives
      the later normalization, so the two are equivalent here.
    * ``k_einc`` is normalized but the phase uses the original vector; the
      caller has already multiplied it by ``k0``.
    * ``rho = v0 + xi0*(v1-v0) + xi1*(v2-v0)`` uses the edge vectors, not the
      vertices.
    * The rule is ``quad(order + 1)``, so orders 1-3 are reachable.
    """
    order = _checked_order(order)
    geo = np.asarray(geo, dtype=float)
    if geo.shape != (3, 3):
        raise ValueError("a face needs three nodes with three coordinates")
    k_einc = np.asarray(k_einc, dtype=float)
    pol_einc = np.asarray(pol_einc, dtype=float)

    v0 = geo[0].copy()
    v1 = geo[1].copy()
    v2 = geo[2].copy()
    v_in = (v0 + v1 + v2) / 3.0 - np.asarray(int_node, dtype=float)

    v1 = v1 - v0
    v2 = v2 - v0
    u = v1 / np.linalg.norm(v1)
    n = np.cross(v1, v2)
    n = n * np.dot(n, v_in)
    n = n / np.linalg.norm(n)
    v = np.cross(n, u)
    k = k_einc / np.linalg.norm(k_einc)

    c_geo2 = np.zeros((3, 2))
    c_geo2[1, 0] = np.dot(v1, u)
    c_geo2[2, 0] = np.dot(v2, u)
    c_geo2[2, 1] = np.dot(v2, v)
    det_j, inv_j = _jacobian(_DNS_REF_2D, c_geo2)

    points, weights = quadrature_2d(order + 1)
    numv = shape_2d(order, points[:1], inv_j).Nv.shape[1]

    # ``incPol`` does not depend on the quadrature point; the oracle rebuilds
    # it inside the loop, which is the same value.
    inc_pol = pol_einc - np.cross(-n, np.cross(k, pol_einc))
    pol_u = np.dot(inc_pol, u)
    pol_v = np.dot(inc_pol, v)

    f = np.zeros(numv, dtype=complex)
    for iq in range(points.shape[0]):
        basis = shape_2d(order, points[iq : iq + 1], inv_j)
        rho = v0 + points[iq, 0] * v1 + points[iq, 1] * v2
        phase = np.exp(complex(0.0, -float(np.dot(k_einc, rho))))
        v_einc = phase * np.array([pol_u, pol_v], dtype=complex)
        # C++ ``Nv`` is stored as ``(2, numv)``; the Python basis stores the
        # transpose, so ``Nv^T * vEinc`` is a plain matrix-vector product.
        f += (basis.Nv[0] @ v_einc) * weights[iq]
    return f * det_j


def element_matrices_2d_mixed(order: int, geo: np.ndarray) -> MixedElement2D:
    """Assemble the mixed H(curl)/H1 triangle matrices.

    Line-by-line port of the ``cDim == 2`` branch of the first ``ele_mat``
    constructor: ``St``/``Tt`` from the H(curl) basis, ``Sz``/``Tz`` from the
    scalar basis, ``G = Nv^T dNs``, all scaled by ``|detJ|`` afterwards.  The
    material scaling stays at the call site, as in ``assemble_em_e_tl_eig``,
    because the same matrices are used with both ``epsr`` and ``1/mur``.

    ``geo`` is ``(3, 2)``.  The rule is ``quad(order + 1)``, so orders 1-3 are
    reachable; order 4 raises :class:`ValueError` exactly like the oracle's
    ``quadrature order not yet implemented``.
    """
    order = _checked_order(order)
    geo = np.asarray(geo, dtype=float)
    det_j, inv_j = triangle_jacobian(geo)
    points, weights = quadrature_2d(order + 1)

    nums = shape_2d(order, points[:1], inv_j).Ns.shape[1]
    numv = shape_2d(order, points[:1], inv_j).Nv.shape[1]
    st = np.zeros((numv, numv))
    tt = np.zeros((numv, numv))
    sz = np.zeros((nums, nums))
    tz = np.zeros((nums, nums))
    g = np.zeros((numv, nums))
    for iq in range(points.shape[0]):
        basis = shape_2d(order, points[iq : iq + 1], inv_j)
        nv = basis.Nv[0]  # (numv, 2)
        dnv = basis.dNv[0]  # (numv,) -- the 2D curl is a scalar
        ns = basis.Ns[0]  # (nums,)
        dns = basis.dNs[0]  # (nums, 2)
        weight = weights[iq]
        st += np.outer(dnv, dnv) * weight
        tt += (nv @ nv.T) * weight
        sz += (dns @ dns.T) * weight
        tz += np.outer(ns, ns) * weight
        g += (nv @ dns.T) * weight

    scale = complex(det_j)
    return MixedElement2D(
        St=st * scale,
        Tt=tt * scale,
        Sz=sz * scale,
        Tz=tz * scale,
        G=g * scale,
    )
