"""Tests for :mod:`fes.core.quadrature`.

The reference simplex rules use ``quad(order)`` → ``quad(order+1)`` points.
The weights integrate the unit simplex, so they sum to 1/2 for triangles and
1/6 for tetrahedra.
"""

from __future__ import annotations


import pytest

from fes.core.quadrature import MAX_ORDER, quadrature_2d, quadrature_3d


EXPECTED_POINTS = {1: 3, 2: 6, 3: 12, 4: 64}
EXPECTED_TET_POINTS = {1: 4, 2: 11, 3: 24, 4: 64}


def test_max_order_is_four() -> None:
    assert MAX_ORDER == 4


@pytest.mark.parametrize("order", sorted(EXPECTED_POINTS))
def test_weights_integrate_the_unit_simplex(order: int) -> None:
    _, weights = quadrature_2d(order)
    assert weights.sum() == pytest.approx(0.5, rel=0, abs=1e-14)


@pytest.mark.parametrize("order", [0, -1, 5])
def test_unsupported_orders_raise(order: int) -> None:
    with pytest.raises(ValueError):
        quadrature_2d(order)


def test_returns_copies() -> None:
    """Mutating the result must not corrupt later calls."""
    points, weights = quadrature_2d(1)
    points[0, 0] = 123.0
    weights[0] = -1.0
    fresh_points, fresh_weights = quadrature_2d(1)
    assert fresh_points[0, 0] != 123.0
    assert fresh_weights[0] != -1.0


@pytest.mark.parametrize("order", sorted(EXPECTED_TET_POINTS))
def test_tet_weights_integrate_the_reference_tetrahedron(order: int) -> None:
    _, weights = quadrature_3d(order)
    assert weights.sum() == pytest.approx(1.0 / 6.0, rel=0, abs=1e-14)


@pytest.mark.parametrize("order", [0, -1, 5])
def test_unsupported_tet_orders_raise(order: int) -> None:
    with pytest.raises(ValueError):
        quadrature_3d(order)


def test_tet_returns_copies() -> None:
    points, weights = quadrature_3d(1)
    points[0, 0] = 123.0
    weights[0] = -1.0
    fresh_points, fresh_weights = quadrature_3d(1)
    assert fresh_points[0, 0] != 123.0
    assert fresh_weights[0] != -1.0
