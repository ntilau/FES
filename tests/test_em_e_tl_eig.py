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
    # Vacuum: shift = -0.5 * k0^2 * epsr_max * mur_max.
    assert system.shift == pytest.approx(-0.5 * system.k0**2)


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


# ── user-facing controls ──


@pytest.mark.triangle
def test_n_modes_overrides_the_default_count() -> None:
    """``n_modes`` overrides ``min(5, n/2 + 1)``.

    The shift-invert solver returns the ``n_modes`` modes closest to the shift
    (largest ``|\theta|``), reported in stable ``|\beta|`` descending order.
    With the default negative shift these are the modes nearest the shift, not
    necessarily the first ``n_modes`` entries of the default 5-mode result.
    """
    mesh = mesh_from_poly(REPO / "data" / "WR90_TL.poly")
    default = assemble_em_e_tl_eig(mesh, 10e9, 2)
    limited = assemble_em_e_tl_eig(mesh, 10e9, 2, n_modes=2)
    assert default.n_modes == 5
    assert limited.n_modes == 2
    # Top 2 modes closest to the shift, sorted by |beta| descending.
    np.testing.assert_allclose(
        limited.mode_beta,
        np.array([0 + 82.57241796j, 0 + 61.09184387j]),
        rtol=1e-9,
        atol=0,
    )


@pytest.mark.triangle
def test_n_modes_is_clamped_to_valid_range() -> None:
    """``n_modes`` above the maximum is clamped; zero or negative falls back to default."""
    mesh = mesh_from_poly(REPO / "data" / "WR90_TL.poly")
    default = assemble_em_e_tl_eig(mesh, 10e9, 2)
    assert default.n_modes == min(5, (default.free_edges.size + default.n_node_dofs) // 2 + 1)
    # n_modes=0 means "use default".
    fallback = assemble_em_e_tl_eig(mesh, 10e9, 2, n_modes=0)
    assert fallback.n_modes == default.n_modes
    # n_modes larger than numtot//2+1 is clamped to numtot//2+1.
    oversized = assemble_em_e_tl_eig(mesh, 10e9, 2, n_modes=1000)
    assert oversized.n_modes == (default.free_edges.size + default.n_node_dofs) // 2 + 1


@pytest.mark.triangle
def test_custom_shift_changes_the_spectrum() -> None:
    """A user-specified shift overrides ``-k0^2 * maxepsr * maxmur``."""
    mesh = mesh_from_poly(REPO / "data" / "WR90_TL.poly")
    default = assemble_em_e_tl_eig(mesh, 10e9, 2)
    assert default.shift == pytest.approx(-0.5 * default.k0**2)
    shifted = assemble_em_e_tl_eig(mesh, 10e9, 2, shift=-50.0)
    assert shifted.shift == pytest.approx(-50.0)
    assert shifted.shift != default.shift
    # Modes should differ when the shift changes.
    assert not np.allclose(default.mode_beta, shifted.mode_beta, atol=1e-6)


def test_ortho_false_skips_the_projection() -> None:
    """``ortho=False`` skips the OO = diag(I, 0) projection."""
    mesh = mesh_from_poly(REPO / "data" / "WR90_TL.poly")
    ortho_on = assemble_em_e_tl_eig(mesh, 10e9, 2, ortho=True)
    ortho_off = assemble_em_e_tl_eig(mesh, 10e9, 2, ortho=False)
    assert ortho_on.n_modes == ortho_off.n_modes
    # Results should differ when orthogonalisation is disabled.
    assert not np.allclose(ortho_on.mode_beta, ortho_off.mode_beta, atol=1e-6)


@pytest.mark.triangle
def test_eig_tol_forwarded_to_arpack(monkeypatch) -> None:
    """``eig_tol`` is passed to the ARPACK sparse solver."""
    mesh = mesh_from_poly(REPO / "data" / "WR90_TL.poly")
    monkeypatch.setattr(module, "DENSE_LIMIT", 0)
    system = assemble_em_e_tl_eig(mesh, 10e9, 2, eig_tol=1e-8)
    assert system.solver == "arpack"
    assert system.n_modes > 0


# ── arnoldi shift-invert path ───────────────────────────────────────────────


@pytest.mark.triangle
def test_arnoldi_solver_matches_dense(monkeypatch) -> None:
    """The hand-written Arnoldi shift-invert reproduces the dense spectrum."""
    mesh = mesh_from_poly(REPO / "data" / "WR90_TL.poly")
    dense = assemble_em_e_tl_eig(mesh, 10e9, 2)
    arnoldi = assemble_em_e_tl_eig(mesh, 10e9, 2, solver="arnoldi")
    assert arnoldi.solver == "arnoldi"
    np.testing.assert_allclose(
        arnoldi.mode_beta, dense.mode_beta, rtol=1e-6, atol=1e-9
    )


@pytest.mark.triangle
def test_arnoldi_solver_respects_n_modes() -> None:
    """``solver='arnoldi'`` respects ``n_modes``."""
    mesh = mesh_from_poly(REPO / "data" / "WR90_TL.poly")
    system = assemble_em_e_tl_eig(mesh, 10e9, 2, solver="arnoldi", n_modes=2)
    assert system.solver == "arnoldi"
    assert system.n_modes == 2


@pytest.mark.triangle
def test_arnoldi_solver_auto_falls_back_to_dense(monkeypatch) -> None:
    """``solver='auto'`` with small systems uses dense, not arnoldi."""
    mesh = mesh_from_poly(REPO / "data" / "WR90_TL.poly")
    system = assemble_em_e_tl_eig(mesh, 10e9, 2, solver="auto")
    assert system.solver == "dense"


def test_unknown_solver_raises() -> None:
    """An unknown ``solver`` value raises ``ValueError``."""
    mesh = _square_mesh()
    with pytest.raises(ValueError, match="unknown solver"):
        assemble_em_e_tl_eig(mesh, 1e9, 2, solver="bogus")

