"""Tests for :mod:`fes.core.element_matrix`.

The tests use a skewed tetrahedron and one of its faces: on the reference
simplex several transform steps are invisible, so omitting ``invJ`` or the
embedded face frame would still pass.  Full 3x3 Jacobians are exercised.
"""

from __future__ import annotations


import numpy as np
import pytest

from fes.core.element_matrix import (
    MAX_ASSEMBLY_ORDER,
    Element3D,
    Material,
    element_matrices_3d,
    element_matrices_face,
    face_frame,
    tetrahedron_jacobian,
)


# ── bulk tetrahedron ────────────────────────────────────────────────────────


@pytest.mark.parametrize("order", [1, 2, 3])
def test_bulk_matrices_have_their_expected_structure(order: int) -> None:
    geo = np.array(
        [
            [0.10, 0.00, 0.00],
            [2.00, 0.30, 0.10],
            [0.40, 1.70, 0.20],
            [0.20, 0.50, 2.20],
        ]
    )
    result = element_matrices_3d(order, geo, Material(2.5, 1.3, 0.02, 0.001))

    assert isinstance(result, Element3D)
    # Both Gram matrices are symmetric; Z is pure imaginary.
    np.testing.assert_allclose(result.S, result.S.T, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(result.T, result.T.T, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(result.Z.real, 0.0, rtol=0.0, atol=1e-15)
    # Z is built from the *unscaled* mass matrix, so it is T divided by the
    # complex permittivity and multiplied by i z0 sigma.
    ratio = 1j * 376.730313461 * 0.02 / complex(2.5, -0.0025)
    np.testing.assert_allclose(result.Z, result.T * ratio, rtol=1e-12, atol=1e-12)


def test_vacuum_material_removes_the_conductivity_term() -> None:
    geo = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.3, 0.1],
            [0.4, 1.7, 0.2],
            [0.2, 0.5, 2.2],
        ]
    )
    result = element_matrices_3d(1, geo)
    np.testing.assert_array_equal(result.Z, np.zeros((6, 6)))


@pytest.mark.parametrize("order", [1, 2])
def test_material_scaling_follows_the_reference_formulas(order: int) -> None:
    geo = np.array(
        [
            [0.1, 0.0, 0.0],
            [2.0, 0.3, 0.1],
            [0.4, 1.7, 0.2],
            [0.2, 0.5, 2.2],
        ]
    )
    unit = element_matrices_3d(order, geo)
    scaled = element_matrices_3d(order, geo, Material(2.5, 1.3, 0.0, 0.001))

    # S carries detJ / mur; T carries detJ * (epsr + i epsr2).
    np.testing.assert_allclose(scaled.S, unit.S / 1.3, rtol=1e-13, atol=1e-15)
    np.testing.assert_allclose(
        scaled.T, unit.T * complex(2.5, -0.0025), rtol=1e-13, atol=1e-15
    )


def test_reference_tetrahedron_jacobian() -> None:
    geo = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    det_j, inv_j = tetrahedron_jacobian(geo)
    assert det_j == pytest.approx(1.0)
    np.testing.assert_allclose(inv_j, np.eye(3), rtol=0.0, atol=0.0)


# ── embedded boundary face ──────────────────────────────────────────────────


def test_face_frame_is_the_reference_embedded_coordinate_system() -> None:
    geo = np.array(
        [
            [0.10, 0.00, 0.00],
            [2.00, 0.30, 0.10],
            [0.40, 1.70, 0.20],
        ]
    )
    frame = face_frame(geo, np.array([0.20, 0.50, 2.20]))

    # The origin of the 2D frame is the first vertex.
    np.testing.assert_array_equal(frame[0], [0.0, 0.0])
    assert frame[1, 1] == 0.0
    # detJ of the embedded map is twice the triangle area, i.e. |v1 x v2|.
    det_j = abs(float(np.linalg.det(frame[1:])))
    v1 = geo[1] - geo[0]
    v2 = geo[2] - geo[0]
    assert det_j == pytest.approx(np.linalg.norm(np.cross(v1, v2)), rel=1e-14)


def test_face_frame_follows_the_int_node_side() -> None:
    geo = np.array(
        [
            [0.10, 0.00, 0.00],
            [2.00, 0.30, 0.10],
            [0.40, 1.70, 0.20],
        ]
    )
    above = face_frame(geo, np.array([0.20, 0.50, 2.20]))
    below = face_frame(geo, np.array([0.20, 0.50, -2.20]))
    # u stays anchored to the first edge; flipping the interior side flips v,
    # which is the second column of the embedded geometry.
    np.testing.assert_allclose(above[1:], below[1:] * [1.0, -1.0], rtol=0.0, atol=0.0)


# ── validation ──────────────────────────────────────────────────────────────


def test_assembly_order_ceiling_matches_the_reference() -> None:
    # The bulk constructor builds quad(p+1); v0.1 has rules only up to 4.
    assert MAX_ASSEMBLY_ORDER == 3
    geo = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    with pytest.raises(ValueError, match="orders 1-3"):
        element_matrices_3d(4, geo)
    with pytest.raises(ValueError, match="orders 1-3"):
        element_matrices_face(4, geo[:3], geo[3])
    with pytest.raises(ValueError):
        element_matrices_3d(0, geo)


def test_bad_geometry_is_rejected() -> None:
    with pytest.raises(ValueError):
        tetrahedron_jacobian(np.zeros((3, 3)))
    with pytest.raises(ValueError):
        face_frame(np.zeros((4, 3)), np.zeros(3))
