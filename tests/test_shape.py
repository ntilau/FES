"""Tests for :mod:`fes.core.shape`.

These exercise the basis invariants on the reference triangle and on a skewed
triangle, so both the reference basis and the ``invJ`` transform are checked.
"""

from __future__ import annotations


import numpy as np
import pytest

from fes.core.shape import (
    scalar_dof_count,
    shape_2d,
    vector_dof_count,
)


def test_dof_counts() -> None:
    assert [scalar_dof_count(p) for p in (1, 2, 3, 4)] == [3, 6, 10, 15]
    assert [vector_dof_count(p) for p in (1, 2, 3, 4)] == [3, 8, 15, 24]


@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_scalar_gradient_matches_finite_difference(order: int) -> None:
    """``dNs`` must be the physical gradient of ``Ns`` under a general map."""
    inv_j = np.array([[2.0, 0.3], [-0.5, 1.5]])
    j = np.linalg.inv(inv_j)
    point = np.array([0.2, 0.3])
    h = 1e-6

    base = shape_2d(order, point[None, :], inv_j)
    for axis in (0, 1):
        # C++ invJ = J^{-1} with J = d(ref)->d(phys) transposed, so the
        # reference-space displacement for a unit physical step e_axis is
        # invJ.T[:, axis].
        step = h * inv_j.T[:, axis]
        plus = shape_2d(order, (point + step)[None, :], inv_j)
        minus = shape_2d(order, (point - step)[None, :], inv_j)
        fd = (plus.Ns[0] - minus.Ns[0]) / (2.0 * h)
        np.testing.assert_allclose(
            base.dNs[0, :, axis], fd, rtol=1e-7, atol=1e-8,
            err_msg=f"dNs order={order} axis={axis}",
        )

    # Also make sure the mapping was actually non-trivial.
    assert not np.allclose(j, np.eye(2))


@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_vector_curl_matches_finite_difference(order: int) -> None:
    """``dNv`` must equal the curl of ``Nv``.

    The basis functions are now ordered as in the C++ oracle (Shape.cpp),
    so ``dNv`` matches the numerical curl for every column without any sign
    quirk.
    """
    inv_j = np.array([[1.3, -0.4], [0.2, 0.9]])
    point = np.array([0.15, 0.25])
    h = 1e-6

    base = shape_2d(order, point[None, :], inv_j)
    steps = [h * inv_j.T[:, axis] for axis in (0, 1)]
    plus = [shape_2d(order, (point + s)[None, :], inv_j) for s in steps]
    minus = [shape_2d(order, (point - s)[None, :], inv_j) for s in steps]

    dny_dx = (plus[0].Nv[0, :, 1] - minus[0].Nv[0, :, 1]) / (2.0 * h)
    dnx_dy = (plus[1].Nv[0, :, 0] - minus[1].Nv[0, :, 0]) / (2.0 * h)
    numerical = dny_dx - dnx_dy

    np.testing.assert_allclose(
        base.dNv[0], numerical, rtol=1e-6, atol=1e-7,
        err_msg=f"curl order={order}",
    )


@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_nonidentity_mapping_follows_the_reference_transform(order: int) -> None:
    """The reference applies ``invJ`` to reference gradients and scales curls."""
    inv_j = np.array([[1.3, -0.4], [0.2, 0.9]])
    points = np.array([[0.2, 0.3], [0.1, 0.6], [0.05, 0.05]])

    reference = shape_2d(order, points)
    mapped = shape_2d(order, points, inv_j)

    np.testing.assert_allclose(mapped.Ns, reference.Ns, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        mapped.dNs, reference.dNs @ inv_j.T, rtol=1e-13, atol=1e-13,
    )
    np.testing.assert_allclose(
        mapped.Nv, reference.Nv @ inv_j.T, rtol=1e-13, atol=1e-13,
    )
    np.testing.assert_allclose(
        mapped.dNv, reference.dNv * np.linalg.det(inv_j), rtol=1e-13, atol=1e-13,
    )


@pytest.mark.parametrize("order", [2, 3, 4])
def test_scalar_edge_functions_vanish_on_opposite_edge(order: int) -> None:
    """Each edge bubble must vanish on the two edges it does not belong to."""
    # Edge 0-1 (y=0): l2=0.  Edge 0-2 (x=0): l1=0.  Edge 1-2 (x+y=1): l0=0.
    samples = {
        0: np.array([0.25, 0.0]),   # on edge 0-1
        1: np.array([0.0, 0.25]),   # on edge 0-2
        2: np.array([0.25, 0.75]),  # on edge 1-2
    }
    if order == 2:
        edge_columns = {0: [3], 1: [4], 2: [5]}
    elif order == 3:
        edge_columns = {0: [3, 6], 1: [4, 7], 2: [5, 8]}
    else:
        edge_columns = {0: [3, 6, 9], 1: [4, 7, 10], 2: [5, 8, 11]}

    for own_edge, point in samples.items():
        shape = shape_2d(order, point[None, :])
        for edge, cols in edge_columns.items():
            if edge == own_edge:
                continue
            np.testing.assert_allclose(
                shape.Ns[0, cols], 0.0, atol=1e-12,
                err_msg=f"order={order} edge {edge} did not vanish on edge {own_edge}",
            )


def test_order_four_vertex_functions_are_quartic() -> None:
    """At a vertex the p=4 vertex function is 1 and the others vanish."""
    vertices = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    shape = shape_2d(4, vertices)
    np.testing.assert_allclose(shape.Ns[:, :3], np.eye(3), atol=1e-12)
    np.testing.assert_allclose(shape.Ns[:, 3:], 0.0, atol=1e-12)


def test_rejects_unsupported_order() -> None:
    with pytest.raises(ValueError):
        shape_2d(5, np.array([[0.2, 0.2]]))
