"""Tests for the 2D cross-section eigenmode assembler.

These pin the bookkeeping -- DOF counts, the shift, the free sets, the guards --
and the parity with the ``em_e_fd`` waveport formulation: the Zhu-Cangellaris
Schur complement, the ascending-cutoff selection, the B-orthonormalized mode
vectors and the sweep tracking.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.linalg

from fes.core.assemblers import assemble_em_e_tl_eig
from fes.core.assemblers.em_e_tl_eig import (
    _bc_type_map,
    _dirichlet_dofs,
    _dirichlet_node_dofs,
)
from fes.core.assemblers.port_pencil import (
    _port_reduced_pencil,
    postprocess_modes,
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


def _reduced_pencil(system):
    """Rebuild the edge-only pencil ``(A_tt, B_schur)`` from a solved system."""
    fv, fs = system.free_edges, system.free_nodes
    numt, numz = fv.size, fs.size
    tmpA = np.zeros((numt + numz, numt + numz), dtype=complex)
    tmpB = np.zeros_like(tmpA)
    tmpA[:numt, :numt] = (
        system.St[fv][:, fv] - system.kk * system.Tt[fv][:, fv]
    ).toarray()
    tmpB[:numt, :numt] = system.Tt2[fv][:, fv].toarray()
    if numz:
        Gf = system.G[fv][:, fs].toarray()
        tmpB[:numt, numt:] = Gf
        tmpB[numt:, :numt] = Gf.T
        tmpB[numt:, numt:] = (
            system.Sz[fs][:, fs] - system.kk * system.Tz[fs][:, fs]
        ).toarray()
    return _port_reduced_pencil(tmpA, tmpB, numt, numz)


# ── bookkeeping ─────────────────────────────────────────────────────────────


@pytest.mark.triangle
def test_wr90_tl_dof_bookkeeping() -> None:
    mesh = mesh_from_poly(REPO / "data" / "WR90_TL.poly")
    system = assemble_em_e_tl_eig(mesh, 10e9, 2, solve=False)
    # p=2 on this cross-section: 26 H(curl) DOFs, of which the PEC boundary
    # (6 mesh edges, 2 levels each) removes 12.  Scalar: 15 DOFs, of which the
    # PEC edges constrain both vertices and the edge-interior level (12).
    assert system.n_edge_dofs == 26
    assert system.free_edges.size == 14
    assert system.n_node_dofs == 15
    assert system.free_nodes.size == 3
    assert system.dirichlet_nodes.size == 12
    assert system.dof == 41
    # k0 = 2 pi f / c0 with the reference constants.
    assert system.k0 == pytest.approx(2 * np.pi * 10e9 / 299792458.0)
    assert system.k0 == pytest.approx(209.5845, rel=1e-6)
    # Vacuum and the waveport default shift: -k0^2 * epsr_max * mur_max.
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


def test_dirichlet_node_dofs_cover_vertices_and_edge_interiors() -> None:
    mesh = _square_mesh()
    # Order 1 consists of the vertex DOFs only.
    assert set(_dirichlet_node_dofs(mesh, 1, {1: "PerfectE"}).tolist()) == {
        0,
        1,
        3,
    }
    # Order 2 adds one edge-interior level: n_nodes + edge_id.
    assert set(_dirichlet_node_dofs(mesh, 2, {1: "PerfectE"}).tolist()) == {
        0,
        1,
        3,
        mesh.n_nodes,
        mesh.n_nodes + 1,
    }
    # Order 3 adds a second level, but never the face-interior DOFs.
    order3 = set(_dirichlet_node_dofs(mesh, 3, {1: "PerfectE"}).tolist())
    face_dofs = {
        mesh.n_nodes + 2 * mesh.n_edges + face_id
        for face_id in range(mesh.n_faces)
    }
    assert order3 == {
        0,
        1,
        3,
        mesh.n_nodes,
        mesh.n_nodes + 1,
        mesh.n_nodes + mesh.n_edges,
        mesh.n_nodes + mesh.n_edges + 1,
    }
    assert order3.isdisjoint(face_dofs)
    assert _dirichlet_node_dofs(mesh, 2, {2: "PerfectE"}).size == 0


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
    assert system.mode_vec.size == 0
    assert system.St.shape == (system.n_edge_dofs, system.n_edge_dofs)
    assert system.G.shape == (system.n_edge_dofs, system.n_node_dofs)
    assert system.S is None


# ── post-processing (shared with the waveport) ──────────────────────────────


def test_postprocess_keeps_physical_modes_in_ascending_cutoff_order() -> None:
    kk = 100.0
    # kc^2 = kk + lam: 50, 80, 140 are physical; -50 is not.
    lam = np.array([-50.0, -20.0, 40.0, -150.0], dtype=complex)
    vectors = np.eye(4, dtype=complex)
    beta, _, _ = postprocess_modes(
        kk, 10.0, lam, vectors, lambda x: x, nmodes=5
    )
    np.testing.assert_allclose(np.real(beta), [np.sqrt(50.0), np.sqrt(20.0), 0.0])
    assert beta[2] == pytest.approx(1j * np.sqrt(40.0))
    assert beta.size == 3


def test_postprocess_truncates_to_n_modes() -> None:
    kk = 100.0
    lam = np.array([-50.0, -20.0, 40.0], dtype=complex)
    beta, mode_vec, _ = postprocess_modes(
        kk, 10.0, lam, np.eye(3, dtype=complex), lambda x: x, nmodes=2
    )
    np.testing.assert_allclose(np.real(beta), [np.sqrt(50.0), np.sqrt(20.0)])
    assert mode_vec.shape == (3, 2)


def test_postprocess_b_orthonormalizes_and_pivots_signs() -> None:
    kk = 100.0
    lam = np.array([-50.0, -20.0], dtype=complex)
    B = np.diag([2.0, 3.0, 4.0]).astype(complex)
    # A non-trivial combination, so the Gram matrix is non-diagonal.
    vectors = np.array(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=complex
    )
    beta, mode_vec, mode_vecf = postprocess_modes(
        kk, 10.0, lam, vectors, lambda x: B @ x, nmodes=2
    )
    np.testing.assert_allclose(
        mode_vec.T @ B @ mode_vec, np.eye(2), atol=1e-12
    )
    assert mode_vec.shape == mode_vecf.shape == (3, 2)
    for m in range(2):
        pivot = int(np.argmax(np.abs(mode_vec[:, m].real)))
        assert mode_vec[pivot, m].real >= 0.0
    assert beta.size == 2


def test_postprocess_tracks_previous_modes_by_nearest_beta() -> None:
    kk = 100.0
    lam = np.array([-50.0, -20.0, 40.0], dtype=complex)
    vectors = np.eye(3, dtype=complex)
    base, _, _ = postprocess_modes(kk, 10.0, lam, vectors, lambda x: x, nmodes=3)
    prev = base[[2, 0, 1]]
    tracked, _, _ = postprocess_modes(
        kk, 10.0, lam, vectors, lambda x: x, nmodes=3, prev_beta=prev
    )
    np.testing.assert_allclose(tracked, prev)


# ── parity with the waveport formulation ────────────────────────────────────


@pytest.mark.triangle
def test_mode_vectors_are_b_orthonormal() -> None:
    mesh = mesh_from_poly(REPO / "data" / "WR90_TL.poly")
    system = assemble_em_e_tl_eig(mesh, 10e9, 2)
    _, B_schur = _reduced_pencil(system)
    np.testing.assert_allclose(
        system.mode_vec.T @ B_schur @ system.mode_vec,
        np.eye(system.n_modes),
        atol=1e-10,
    )


@pytest.mark.triangle
def test_schur_complement_matches_the_full_block_pencil() -> None:
    """The reduced pencil's spectrum equals the full pencil's finite part."""
    mesh = mesh_from_poly(REPO / "data" / "WR90_TL.poly")
    system = assemble_em_e_tl_eig(mesh, 10e9, 2)
    fv, fs = system.free_edges, system.free_nodes
    numt, numz = fv.size, fs.size
    A_tt, B_schur = _reduced_pencil(system)
    Gf = system.G[fv][:, fs].toarray()
    D = (
        system.Sz[fs][:, fs] - system.kk * system.Tz[fs][:, fs]
    ).toarray()
    # [[A_tt, 0], [G^T, D]] x = lam [[Tt2, G], [0, 0]] x  ->  A_tt v = lam B_schur v
    left = np.block(
        [[A_tt, np.zeros((numt, numz), complex)], [Gf.T, D]]
    )
    right = np.block(
        [[B_schur + Gf @ np.linalg.solve(D, Gf.T), Gf],
         [np.zeros((numz, numt + numz), complex)]]
    )
    alpha, beta = scipy.linalg.eig(
        left, right, homogeneous_eigvals=True
    )[0]
    finite = alpha[np.abs(beta) > 1e-10 * np.max(np.abs(alpha))] / beta[
        np.abs(beta) > 1e-10 * np.max(np.abs(alpha))
    ]
    reduced = scipy.linalg.eigvals(A_tt, B_schur)
    # Every reduced eigenvalue is a finite eigenvalue of the full pencil.
    for value in reduced:
        assert np.min(np.abs(finite - value)) < 1e-6 * max(1.0, abs(value))


@pytest.mark.triangle
def test_modes_satisfy_the_propagation_relation() -> None:
    mesh = mesh_from_poly(REPO / "data" / "WR90_TL.poly")
    system = assemble_em_e_tl_eig(mesh, 10e9, 2)
    np.testing.assert_allclose(
        system.kc**2,
        np.abs(system.k0**2 - system.mode_beta**2),
        rtol=1e-10,
    )
    kc2 = np.real(system.k0**2 + (-system.mode_beta**2))
    assert np.all(np.diff(kc2) >= -1e-9)


# ── solver paths ────────────────────────────────────────────────────────────


@pytest.mark.triangle
def test_arpack_path_matches_dense() -> None:
    """ARPACK and the dense generalized solve agree."""
    mesh = mesh_from_poly(REPO / "data" / "WR90_TL.poly")
    dense = assemble_em_e_tl_eig(mesh, 10e9, 3, solver="dense")
    arpack = assemble_em_e_tl_eig(mesh, 10e9, 3, solver="arpack")
    assert dense.solver == "dense"
    assert arpack.solver == "arpack"
    np.testing.assert_allclose(
        arpack.mode_beta, dense.mode_beta, rtol=1e-8, atol=1e-8
    )


@pytest.mark.triangle
def test_arnoldi_is_a_deprecated_arpack_alias() -> None:
    mesh = mesh_from_poly(REPO / "data" / "WR90_TL.poly")
    arpack = assemble_em_e_tl_eig(mesh, 10e9, 3, solver="arpack")
    arnoldi = assemble_em_e_tl_eig(mesh, 10e9, 3, solver="arnoldi")
    assert arnoldi.solver == arpack.solver == "arpack"
    np.testing.assert_allclose(arnoldi.mode_beta, arpack.mode_beta, atol=0)


# ── user-facing controls ────────────────────────────────────────────────────


@pytest.mark.triangle
def test_n_modes_overrides_the_default_count() -> None:
    mesh = mesh_from_poly(REPO / "data" / "WR90_TL.poly")
    default = assemble_em_e_tl_eig(mesh, 10e9, 2)
    limited = assemble_em_e_tl_eig(mesh, 10e9, 2, n_modes=2)
    assert default.n_modes == 5
    assert limited.n_modes == 2
    # The limited result is the TE10-first prefix of the default selection.
    np.testing.assert_allclose(limited.mode_beta, default.mode_beta[:2])


@pytest.mark.triangle
def test_n_modes_is_clamped_to_valid_range() -> None:
    """``n_modes`` above ``numtot`` is clamped; zero falls back to the default."""
    mesh = mesh_from_poly(REPO / "data" / "WR90_TL.poly")
    default = assemble_em_e_tl_eig(mesh, 10e9, 2)
    numtot = default.free_edges.size + default.free_nodes.size
    assert default.n_modes == min(5, numtot)
    fallback = assemble_em_e_tl_eig(mesh, 10e9, 2, n_modes=0)
    assert fallback.n_modes == default.n_modes
    oversized = assemble_em_e_tl_eig(mesh, 10e9, 2, n_modes=1000)
    # Clamped to every physical mode the pencil has (kc^2 > 0), never more
    # than numtot.
    A_tt, B_schur = _reduced_pencil(default)
    lam = scipy.linalg.eigvals(A_tt, B_schur)
    n_physical = int(np.count_nonzero(np.real(default.kk + lam) > 0.0))
    assert oversized.n_modes == n_physical <= numtot


@pytest.mark.triangle
def test_custom_shift_overrides_the_waveport_default() -> None:
    mesh = mesh_from_poly(REPO / "data" / "WR90_TL.poly")
    default = assemble_em_e_tl_eig(mesh, 10e9, 2)
    assert default.shift == pytest.approx(-default.k0**2)
    shifted = assemble_em_e_tl_eig(mesh, 10e9, 2, shift=-50.0)
    assert shifted.shift == pytest.approx(-50.0)


def test_zero_shift_is_rejected() -> None:
    mesh = _square_mesh()
    with pytest.raises(ValueError, match="shift must be non-zero"):
        assemble_em_e_tl_eig(mesh, 1e9, 2, shift=0.0)


@pytest.mark.triangle
def test_prev_beta_tracks_a_sweep() -> None:
    """``prev_beta`` preserves the previous frequency's mode order."""
    mesh = mesh_from_poly(REPO / "data" / "WR90_TL.poly")
    base = assemble_em_e_tl_eig(mesh, 10e9, 2)
    permutation = [3, 1, 4, 0, 2]
    prev = base.mode_beta[permutation]
    tracked = assemble_em_e_tl_eig(mesh, 10e9, 2, prev_beta=prev)
    np.testing.assert_allclose(tracked.mode_beta, prev, atol=1e-10)


@pytest.mark.triangle
def test_ortho_flag_is_a_no_op() -> None:
    mesh = mesh_from_poly(REPO / "data" / "WR90_TL.poly")
    ortho_on = assemble_em_e_tl_eig(mesh, 10e9, 2, ortho=True)
    ortho_off = assemble_em_e_tl_eig(mesh, 10e9, 2, ortho=False)
    np.testing.assert_array_equal(ortho_on.mode_beta, ortho_off.mode_beta)


def test_unknown_solver_raises() -> None:
    """An unknown ``solver`` value raises ``ValueError``."""
    mesh = _square_mesh()
    with pytest.raises(ValueError, match="unknown solver"):
        assemble_em_e_tl_eig(mesh, 1e9, 2, solver="bogus")
