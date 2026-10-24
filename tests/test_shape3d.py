"""Tests for :mod:`fes.core.shape3d`.

The tetrahedral bases are checked for orders 1-4 at interior points on the
reference tetrahedron.  The H1 ordering changes between orders (p=1/p=4 put
``l3`` first, p=2/p=3 put it last), and several H(curl) derivative columns are
left at zero; both quirks are part of the reference behaviour and are
reproduced rather than corrected.
"""

from __future__ import annotations


import numpy as np
import pytest

from fes.core.shape3d import (
    scalar_dof_count_3d,
    shape_3d,
    vector_dof_count_3d,
)


def test_dof_counts() -> None:
    assert [scalar_dof_count_3d(p) for p in (1, 2, 3, 4)] == [4, 10, 20, 35]
    assert [vector_dof_count_3d(p) for p in (1, 2, 3, 4)] == [6, 20, 45, 84]


@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_nonidentity_mapping_follows_the_reference_transform(order: int) -> None:
    """The reference maps gradients by ``invJ`` and scales curls by ``detJ``."""
    inv_j = np.array([[1.3, -0.4, 0.2], [0.2, 0.9, -0.1], [-0.3, 0.25, 1.1]])
    points = np.array([[0.2, 0.3, 0.1], [0.1, 0.6, 0.2], [0.05, 0.05, 0.4]])

    reference = shape_3d(order, points)
    mapped = shape_3d(order, points, inv_j)

    np.testing.assert_allclose(mapped.Ns, reference.Ns, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        mapped.dNs, reference.dNs @ inv_j.T, rtol=1e-13, atol=1e-13,
    )
    np.testing.assert_allclose(
        mapped.Nv, reference.Nv @ inv_j.T, rtol=1e-13, atol=1e-13,
    )
    # Cross products transform as det(J) * J^{-T} (columns), i.e. as
    # det(J) * J^{-1} on the right when the vectors are stored as rows.
    np.testing.assert_allclose(
        mapped.dNv,
        np.linalg.det(inv_j) * (reference.dNv @ np.linalg.inv(inv_j)),
        rtol=1e-13, atol=1e-13,
    )


@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_scalar_gradient_matches_finite_difference(order: int) -> None:
    """``dNs`` must be the physical gradient of ``Ns``, except for one quirk."""
    inv_j = np.array([[2.0, 0.3, -0.1], [-0.5, 1.5, 0.2], [0.1, -0.2, 1.4]])
    point = np.array([0.2, 0.3, 0.1])
    h = 1e-6

    base = shape_3d(order, point[None, :], inv_j)
    for axis in (0, 1, 2):
        step = h * inv_j.T[:, axis]
        plus = shape_3d(order, (point + step)[None, :], inv_j)
        minus = shape_3d(order, (point - step)[None, :], inv_j)
        fd = (plus.Ns[0] - minus.Ns[0]) / (2.0 * h)

        got = base.dNs[0, :, axis].copy()
        if order == 4:
            # shape.cpp's dNs(0,26) carries a spurious l3 factor.
            got[26] = fd[26]
        np.testing.assert_allclose(
            got, fd, rtol=1e-7, atol=1e-8,
            err_msg=f"dNs order={order} axis={axis}",
        )


@pytest.mark.parametrize("order", [1, 2, 3])
def test_vector_curl_matches_finite_difference(order: int) -> None:
    """For p<=3 every H(curl) column's curl matches the numerical curl.

    The p=4 quartic block is not a consistent Nedelec set (for
    example the quartic edge functions have a non-zero numerical curl while
    the reference leaves those columns at zero), so it is excluded here.
    """
    point = np.array([0.17, 0.23, 0.19])
    h = 1e-6

    base = shape_3d(order, point[None, :])
    plus = [shape_3d(order, (point + h * np.eye(3)[a])[None, :]) for a in range(3)]
    minus = [shape_3d(order, (point - h * np.eye(3)[a])[None, :]) for a in range(3)]
    d = [(plus[a].Nv[0] - minus[a].Nv[0]) / (2.0 * h) for a in range(3)]

    numerical = np.stack(
        [
            d[1][:, 2] - d[2][:, 1],
            d[2][:, 0] - d[0][:, 2],
            d[0][:, 1] - d[1][:, 0],
        ],
        axis=1,
    )
    np.testing.assert_allclose(base.dNv[0], numerical, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("order", [1, 2, 3])
def test_scalar_basis_partitions_unity(order: int) -> None:
    points = np.array([[0.1, 0.2, 0.3], [0.25, 0.25, 0.25], [0.05, 0.1, 0.7]])
    shape = shape_3d(order, points)
    np.testing.assert_allclose(shape.Ns.sum(axis=1), 1.0, atol=1e-12)


def test_order_four_face_derivative_quirk() -> None:
    """``dNs(0,26)`` reproduces the reference's extra ``l3`` factor verbatim."""
    point = np.array([0.1, 0.2, 0.3])
    l0 = 1.0 - point.sum()
    # In the reference's c-ordering bases 26 is l1*l2*l3*(l2-l1) with
    # (l1, l2, l3) = (y, z, 1-x-y-z); the derivative adds a spurious l3.
    l1, l2, l3 = point[1], point[2], l0
    shape = shape_3d(4, point[None, :])
    assert shape.dNs[0, 26, 0] == pytest.approx(
        -l1 * l2 * l3 * (l2 - l1), rel=1e-15, abs=1e-15,
    )


def test_zero_curl_columns_follow_the_reference() -> None:
    """The reference fills only a subset of the H(curl) derivative columns."""
    # These rows are zero in the reference output, transcribing the source.
    expected_zero = {
        1: [],
        2: list(range(6, 12)),
        3: [6, 7, 8, 9, 10, 11, 20, 21, 22, 23, 24, 25, 26, 30, 34, 38],
    }
    points = np.array([[0.1, 0.2, 0.3], [0.25, 0.25, 0.25], [0.05, 0.1, 0.7]])
    for order, rows in expected_zero.items():
        shape = shape_3d(order, points)
        zero = np.where(np.all(shape.dNv == 0.0, axis=(0, 2)))[0].tolist()
        assert zero == rows


def test_rejects_unsupported_order() -> None:
    with pytest.raises(ValueError):
        shape_3d(5, np.array([[0.2, 0.2, 0.2]]))
    with pytest.raises(ValueError):
        vector_dof_count_3d(5)
