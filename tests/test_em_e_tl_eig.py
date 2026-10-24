"""Tests for the 2D cross-section eigenmode assembler.

These pin the bookkeeping -- DOF counts, the shift, the free set, the guards --
and the equivalence of the dense and sparse operator paths.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fes.core.assemblers import assemble_em_e_tl_eig
from fes.core.assemblers import em_e_tl_eig as module
from fes.core.assemblers.em_e_tl_eig import (
    _bc_type_map,
    _dirichlet_dofs,
    _finalize_modes,
)
from fes.mesh.mesh2d import Mesh2D, mesh_from_poly
from fes.mesh.poly_header import Boundary

PY_DIR = Path(__file__).resolve().parents[1]
REPO = PY_DIR


def _square_mesh() -> Mesh2D:
    """A unit square split into two triangles, both vacuum."""
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


# ── bookkeeping ─────────────────────────────────────────────────────────────


@pytest.mark.triangle
def test_wr90_tl_dof_bookkeeping() -> None:
    mesh = mesh_from_poly(REPO / "data" / "WR90_TL.poly")
    system = assemble_em_e_tl_eig(mesh, 10e9, 2, solve=False)
    # p=2 on this cross-section: 2(nEns+nF) vector DOFs of which the two PEC
    # boundaries (6 mesh edges) remove 12; nN+nE scalar DOFs, unconstrained.
    assert system.n_edge_dofs == 26
    assert system.free_edges.size == 14
    assert system.n_node_dofs == 15
    assert system.dof == 41
    # k0 = 2 pi f / c0 with the reference constants.
    assert system.k0 == pytest.approx(2 * np.pi * 10e9 / 299792458.0)
    assert system.k0 == pytest.approx(209.5845, rel=1e-6)
    # Vacuum: shift = -k0^2 epsr_max mur_max.
    assert system.shift == pytest.approx(-system.k0**2)


def test_bc_type_map_last_label_wins() -> None:
    types = _bc_type_map(
        [
            Boundary("a", 1, "PerfectE"),
            Boundary("b", 2, "PerfectH"),
            Boundary("c", 1, "WavePort"),
        ]
    )
    assert types == {1: "WavePort", 2: "PerfectH"}


def test_dirichlet_dofs_use_the_reference_level_numbering() -> None:
    mesh = _square_mesh()
    # Edges 0 (0,1) and 1 (1,3) are PerfectE; ``order`` copies of each are
    # constrained at ``lev * nMeshEdges + edge``.
    assert set(_dirichlet_dofs(mesh, 1, {1: "PerfectE"}).tolist()) == {0, 1}
    assert set(_dirichlet_dofs(mesh, 2, {1: "PerfectE"}).tolist()) == {
        0,
        1,
        mesh.n_edges,
        mesh.n_edges + 1,
    }
    assert _dirichlet_dofs(mesh, 1, {2: "PerfectE"}).size == 0


# ── guards ──────────────────────────────────────────────────────────────────


def test_rejects_3d_meshes_with_the_reference_message() -> None:
    mesh = SimpleNamespace(n_tetras=4)
    with pytest.raises(RuntimeError) as caught:
        assemble_em_e_tl_eig(mesh, 1e9)
    assert str(caught.value) == (
        "em_e_tl_eig requires a 2D cross-section mesh "
        "(use +formula em_e_fd for 3D)"
    )


def test_rejects_waveport_models_with_the_reference_message() -> None:
    mesh = _square_mesh()
    mesh.boundaries = [Boundary("Port", 11, "WavePort", 5)]
    with pytest.raises(RuntimeError) as caught:
        assemble_em_e_tl_eig(mesh, 1e9)
    assert str(caught.value) == (
        "em_e_tl_eig does not support models with waveport boundaries; "
        "use +formula em_ez_fd for driven 2D waveport simulation"
    )


def test_order_4_is_beyond_the_quadrature_ceiling() -> None:
    mesh = _square_mesh()
    with pytest.raises(ValueError, match="orders 1-3"):
        assemble_em_e_tl_eig(mesh, 1e9, 4)


def test_solve_false_returns_the_unsolved_system() -> None:
    mesh = _square_mesh()
    system = assemble_em_e_tl_eig(mesh, 1e9, 2, solve=False)
    assert system.n_modes == 0
    assert system.mode_beta.size == 0
    assert system.St.shape == (system.n_edge_dofs, system.n_edge_dofs)
    assert system.G.shape == (system.n_edge_dofs, system.n_node_dofs)
    assert system.S is None


# ── ordering ────────────────────────────────────────────────────────────────


def test_finalize_modes_applies_the_two_reference_sorts() -> None:
    shift = -100.0
    # ``d = shift + 1/theta``; with a negative shift, larger theta means a
    # larger, less imaginary d.  The reference ends with a stable |beta|-descending
    # sort, so the modes come back [sqrt(1900), 30, 20].
    theta = np.array([2e-3, 0.5e-3, 1e-3])
    modes = _finalize_modes(theta, shift, num_modes=3)
    assert modes.imag.min() == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(
        np.abs(modes), [np.sqrt(1900.0), 30.0, 20.0], rtol=1e-12
    )


def test_finalize_modes_truncates_by_largest_theta() -> None:
    theta = np.array([2e-3, 0.5e-3, 1e-3])
    modes = _finalize_modes(theta, -100.0, num_modes=2)
    # The two largest |theta| survive; 0.5e-3 (beta = sqrt(1900)) is dropped.
    np.testing.assert_allclose(np.abs(modes), [30.0, 20.0], rtol=1e-12)


# ── dense vs sparse operator ────────────────────────────────────────────────


@pytest.mark.triangle
def test_sparse_path_matches_the_dense_operator(monkeypatch) -> None:
    """ARPACK on the SuperLU operator reproduces the dense spectrum.

    ``DENSE_LIMIT`` selects the path; forcing it to zero exercises the ARPACK
    branch that the ``Coaxial`` cross-section needs.
    """
    mesh = mesh_from_poly(REPO / "data" / "WR90_TL.poly")
    dense = assemble_em_e_tl_eig(mesh, 10e9, 2)
    monkeypatch.setattr(module, "DENSE_LIMIT", 0)
    sparse = assemble_em_e_tl_eig(mesh, 10e9, 2)
    assert dense.solver == "dense"
    assert sparse.solver == "arpack"
    np.testing.assert_allclose(
        sparse.mode_beta, dense.mode_beta, rtol=1e-9, atol=0
    )

