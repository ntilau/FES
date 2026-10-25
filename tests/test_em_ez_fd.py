"""Tests for the 2D TMz assembler (:mod:`fes.core.assemblers.em_ez_fd`).

The tests check the building blocks directly, then exercise the assembler on
``BilatFilter``, so a regression localises quickly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fes.core.assemblers import assemble_em_ez_fd
from fes.core.assemblers.em_ez_fd import (
    C0,
    PI,
    _lagrange_1d,
    _triangle_jacobian,
    _bc_type_map,
    _dirichlet_dofs,
)
from fes.mesh.mesh2d import Mesh2D, mesh_from_poly
from fes.mesh.poly_header import Boundary

PY_DIR = Path(__file__).resolve().parents[1]
REPO = PY_DIR


# ── building blocks ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("ne", [2, 3, 4, 5, 6])
def test_lagrange_partitions_unity(ne: int) -> None:
    for t in np.linspace(-1.0, 1.0, 9):
        values, derivs = _lagrange_1d(ne, t)
        assert values.sum() == pytest.approx(1.0, abs=1e-13)
        assert derivs.sum() == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("ne", [2, 5])
def test_lagrange_is_interpolatory(ne: int) -> None:
    nodes = -1.0 + 2.0 * np.arange(ne) / (ne - 1)
    for i, t in enumerate(nodes):
        values, _ = _lagrange_1d(ne, t)
        np.testing.assert_allclose(
            values, np.eye(ne)[i], rtol=0, atol=1e-14
        )


def test_triangle_jacobian() -> None:
    det, inv_j = _triangle_jacobian(
        np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 3.0]])
    )
    assert det == pytest.approx(6.0)
    np.testing.assert_allclose(inv_j, np.diag([0.5, 1.0 / 3.0]), atol=1e-15)
    # A flip must not change |det| (the reference implementation takes the absolute value).
    flipped, _ = _triangle_jacobian(
        np.array([[0.0, 0.0], [0.0, 3.0], [2.0, 0.0]])
    )
    assert flipped == pytest.approx(6.0)


def _square_mesh() -> Mesh2D:
    nodes = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]
    )
    mesh = Mesh2D(
        nodes=nodes,
        faces=np.array([[0, 1, 2], [1, 2, 3]]),
        face_labels=np.ones(2, dtype=np.int64),
        edges=np.array([[0, 1], [1, 3], [2, 3], [0, 2]]),
        edge_labels=np.array([1, 1, 0, 0], dtype=np.uint64),
        n_boundary_edges=4,
    )
    mesh.normalize_order()
    mesh.build_2d_edge_connectivity()
    return mesh


def test_bc_type_map_last_label_wins() -> None:
    types = _bc_type_map(
        [
            Boundary("a", 1, "PerfectE"),
            Boundary("b", 2, "PerfectH"),
            Boundary("c", 1, "WavePort"),
        ]
    )
    assert types == {1: "WavePort", 2: "PerfectH"}


@pytest.mark.parametrize(
    "order, expected",
    [
        (1, {0, 1, 3}),
        (2, {0, 1, 3, 4, 5}),  # plus nN + edge for the two PerfectE edges
    ],
)
def test_dirichlet_dofs(order: int, expected: set) -> None:
    mesh = _square_mesh()
    types = {1: "PerfectE"}
    result = _dirichlet_dofs(mesh, order, types)
    # Edge 0 is (0, 1), edge 1 is (1, 3): both endpoints plus interior DOFs.
    bottom_and_right = set(result.tolist())
    assert bottom_and_right == expected


def test_dirichlet_ignores_other_boundaries() -> None:
    mesh = _square_mesh()
    assert _dirichlet_dofs(mesh, 1, {2: "PerfectE"}).size == 0


# ── end to end ──────────────────────────────────────────────────────────────


@pytest.mark.triangle
@pytest.mark.parametrize("order", [1, 2, 3])
def test_s_and_betas_are_physical(order: int) -> None:
    """Every port is matched in a lossless model: S must be unitary."""
    mesh = mesh_from_poly(REPO / "data" / "BilatFilter.poly")
    system = assemble_em_ez_fd(mesh, 150e9, order)
    assert system.S is not None
    np.testing.assert_allclose(
        system.S.conj().T @ system.S, np.eye(system.n_modes), rtol=0, atol=1e-12
    )
    assert system.A.shape[0] == system.A.shape[1]
    assert system.X.shape[0] == system.A.shape[0]
    assert system.dof >= system.A.shape[0] - system.n_modes
    assert np.all(np.isfinite(system.S))
    for port in system.ports:
        assert port.mode_beta.shape == (port.n_modes,)
        assert np.all(np.isfinite(port.mode_beta))


@pytest.mark.triangle
def test_legacy_connectivity_makes_the_system_singular() -> None:
    """The v0.1 edge-connectivity bug collapses DOFs; the factorisation fails."""
    mesh = mesh_from_poly(REPO / "data" / "BilatFilter.poly")
    with pytest.raises(RuntimeError):
        assemble_em_ez_fd(mesh, 150e9, 2, legacy_edge_connectivity=True)


@pytest.mark.triangle
def test_connectivity_default_is_corrected() -> None:
    mesh = mesh_from_poly(REPO / "data" / "BilatFilter.poly")
    system = assemble_em_ez_fd(mesh, 150e9, 2)
    assert system.legacy_edge_connectivity is False
    assert mesh.edge_connectivity_legacy is False
    # Exact edge count varies by platform (Triangle predicate differences);
    # the structural property (corrected connectivity) is what matters.


def test_wavenumber_matches_the_v01_constants() -> None:
    assert C0 == 299792458.0
    assert PI == 3.14159265358979323846
