"""Tests for core pyFES utilities."""

import numpy as np
from pyfes.constants import get_constants
from pyfes.fem.shape_functions import (
    get_shape_functions, calc_order_mat_size
)
from pyfes.fem.jacobian import jacobian_2d
from pyfes.fem.quadrature import simplex_quad
from pyfes.fem.dof import calc_dofs_number, calc_glob_index, calc_order_mat_size as dof_mat_size


class TestConstants:
    def test_physical_values(self):
        c = get_constants()
        assert abs(c["c0"] - 299792458) < 1
        assert abs(c["z0"] - 120 * np.pi) < 1e-10
        assert abs(c["eps0"] * c["mu0"] * c["c0"] ** 2 - 1) < 1e-15
        assert abs(c["db"](10) - 20) < 1e-10


class TestShapeFunctions:
    def test_1d_linear(self):
        sf, df = get_shape_functions(1, 1)
        # At x=0: shape(0) = [1, 0]
        np.testing.assert_array_almost_equal(sf(0), [1, 0])
        # At x=1: shape(1) = [0, 1]
        np.testing.assert_array_almost_equal(sf(1), [0, 1])
        # Partition of unity at midpoint
        np.testing.assert_array_almost_equal(sf(0.5), [0.5, 0.5])
        # Derivative
        np.testing.assert_array_almost_equal(df(0.5), [-1, 1])

    def test_1d_quadratic(self):
        sf, df = get_shape_functions(1, 2)
        # Partition of unity
        np.testing.assert_array_almost_equal(np.sum(sf(0)), 1)
        np.testing.assert_array_almost_equal(np.sum(sf(0.5)), 1)
        np.testing.assert_array_almost_equal(np.sum(sf(1)), 1)

    def test_2d_linear(self):
        sf, dx, dy = get_shape_functions(2, 1)
        # At vertex (0,0): shape = [1, 0, 0]
        np.testing.assert_array_almost_equal(sf(0, 0), [1, 0, 0])
        # At vertex (1,0): shape = [0, 1, 0]
        np.testing.assert_array_almost_equal(sf(1, 0), [0, 1, 0])
        # At vertex (0,1): shape = [0, 0, 1]
        np.testing.assert_array_almost_equal(sf(0, 1), [0, 0, 1])
        # Partition of unity at centroid
        np.testing.assert_array_almost_equal(np.sum(sf(1/3, 1/3)), 1)
        # Centroid
        np.testing.assert_array_almost_equal(sf(1/3, 1/3), [1/3, 1/3, 1/3])

    def test_2d_quadratic(self):
        sf, dx, dy = get_shape_functions(2, 2)
        for x in [0, 0.2, 0.5]:
            for y in [0, 0.2, 0.5]:
                if x + y <= 1:
                    np.testing.assert_array_almost_equal(
                        np.sum(sf(x, y)), 1, err_msg=f"Failed at ({x},{y})"
                    )

    def test_2d_cubic(self):
        sf, dx, dy = get_shape_functions(2, 3)
        for x in [0, 0.2, 0.5]:
            for y in [0, 0.2, 0.5]:
                if x + y <= 1:
                    np.testing.assert_array_almost_equal(
                        np.sum(sf(x, y)), 1, err_msg=f"Failed at ({x},{y})"
                    )

    def test_2d_quartic(self):
        sf, dx, dy = get_shape_functions(2, 4)
        for x in [0, 0.2, 0.5]:
            for y in [0, 0.2, 0.5]:
                if x + y <= 1:
                    np.testing.assert_array_almost_equal(
                        np.sum(sf(x, y)), 1, err_msg=f"Failed at ({x},{y})"
                    )


class TestOrderMatSize:
    def test_values(self):
        # p=1: nums=3,  numv=3
        n_s, n_v = calc_order_mat_size(1)
        assert n_s == 3 and n_v == 3
        # p=2: nums=6,  numv=8
        n_s, n_v = calc_order_mat_size(2)
        assert n_s == 6 and n_v == 8
        # p=3: nums=10, numv=15
        n_s, n_v = calc_order_mat_size(3)
        assert n_s == 10 and n_v == 15
        # p=4: nums=15, numv=24
        n_s, n_v = calc_order_mat_size(4)
        assert n_s == 15 and n_v == 24


class TestJacobian:
    def test_unit_triangle(self):
        xy = np.array([[0, 0], [1, 0], [0, 1]])
        detJ, invJt = jacobian_2d(xy)
        # J = [[1,0],[0,1]] => det=1
        assert abs(detJ - 1.0) < 1e-14
        np.testing.assert_array_almost_equal(invJt, np.eye(2))

    def test_scaled_triangle(self):
        xy = np.array([[0, 0], [2, 0], [0, 2]])
        detJ, invJt = jacobian_2d(xy)
        # J = [[2,0],[0,2]] => det=4
        assert abs(detJ - 4.0) < 1e-14


class TestQuadrature:
    def test_1d(self):
        X, W = simplex_quad(2, 1)
        assert len(X) == 2
        assert len(W) == 2
        # Integral of 1 over [0,1] should be 1
        assert abs(np.sum(W) - 1) < 1e-14
        # Integral of x over [0,1] should be 1/2
        assert abs(np.sum(X.ravel() * W) - 0.5) < 1e-14

    def test_2d(self):
        X, W = simplex_quad(2, 2)
        assert X.shape[1] == 2
        assert len(W) == 4
        # Integral of 1 over reference triangle should be 0.5
        area = np.sum(W)
        assert abs(area - 0.5) < 1e-14
        # Integral of x over reference triangle should be 1/6
        mx = np.sum(X[:, 0] * W)
        assert abs(mx - 1/6) < 1e-14


class TestMesh:
    def test_regular_square_mesh(self):
        from pyfes.mesh.build import build_regular_square
        mesh = build_regular_square(4, 4)
        assert mesh["NNODE"] == 16
        assert mesh["NELE"] == 18  # 2 * 3 * 3
        assert mesh["NSPIG"] > 0
        assert mesh["node"].shape == (16, 2)
        assert mesh["ele"].shape == (18, 3)
        assert mesh["spig"].shape == (18, 3)
        assert mesh["spig2"].shape[1] == 2

    def test_build_mesh_connectivity(self):
        from pyfes.mesh.build import build_regular_square
        mesh = build_regular_square(3, 3)
        # All elements should be triangles (3 nodes each)
        assert np.all(mesh["ele"].shape[1] == 3)
        # spig values should be non-zero (edges exist)
        assert np.all(np.abs(mesh["spig"]) > 0)
        # spig2 should be sorted
        assert np.all(mesh["spig2"][:, 0] < mesh["spig2"][:, 1])


class TestDof:
    def test_glob_index_2d_p1(self):
        from pyfes.mesh.build import build_regular_square
        mesh = build_regular_square(3, 3)
        gIs, gIv = calc_glob_index(2, 1, mesh, 0)
        assert len(gIs) == 3  # 3 nodes per element
        assert len(gIv) == 3  # 3 edges per element
        # 1-based indices (matching MATLAB convention)
        assert np.all(gIs >= 1) and np.all(gIs <= mesh["NNODE"])
        assert np.all(gIv >= 1) and np.all(gIv <= mesh["NSPIG"])
        assert np.all(gIv >= 0)
