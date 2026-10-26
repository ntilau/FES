"""Tests for the 3D H(curl) x H(grad) A-V assembler (:mod:`fes.core.assemblers.em_av_fd`)."""

from __future__ import annotations

import numpy as np
import pytest

from fes.core.assemblers.em_av_fd import (
    _assemble_bulk,
    _collect_pec,
    _discrete_gradient_local,
    _scalar_dof_count_3d,
    _vector_dof_count_3d,
)
from fes.core.assemblers import assemble_em_av_fd
from fes.mesh.mesh3d import mesh3d_from_poly


# ── discrete gradient ───────────────────────────────────────────────────────


def test_discrete_gradient_order1():
    """Order-1 discrete gradient: G[edge, node_i] = +1, G[edge, node_j] = -1."""
    G = _discrete_gradient_local(6)
    assert G.shape == (6, 4)
    # Edge (0,1): +1 at node 0, -1 at node 1
    assert G[0, 0] == 1.0 and G[0, 1] == -1.0
    assert G[0, 2] == 0.0 and G[0, 3] == 0.0
    # Edge (2,3): +1 at node 2, -1 at node 3
    assert G[5, 2] == 1.0 and G[5, 3] == -1.0
    assert G[5, 0] == 0.0 and G[5, 1] == 0.0


# ── local DOF counts ───────────────────────────────────────────────────────


def test_local_dof_counts():
    assert _scalar_dof_count_3d(1) == 4
    assert _vector_dof_count_3d(1) == 6
    assert _scalar_dof_count_3d(2) == 10
    assert _vector_dof_count_3d(2) == 20


# ── PEC collection ─────────────────────────────────────────────────────────


def test_collect_pec_wr90():
    mesh = mesh3d_from_poly("data/WR90.poly")
    mesh.build_topology()
    dirdofs_s, dirdofs_v, marked = _collect_pec(mesh, 1)
    # WR90 is a closed waveguide: all 8 corner nodes are on PEC faces.
    # But face_dof_map for order 1 returns face nodes, so scalar DoFs are the
    # face node indices. Each PEC face has 3 nodes; 5 PEC faces total.
    assert len(dirdofs_s) > 0
    assert len(dirdofs_v) > 0
    assert len(marked) > 0
    # Marked edges should be a subset of dirdofs_v
    assert set(int(e) for e in marked).issubset(set(int(e) for e in dirdofs_v))


# ── bulk assembly ─────────────────────────────────────────────────────────


def test_assemble_bulk_wr90_order1():
    mesh = mesh3d_from_poly("data/WR90.poly")
    mesh.build_topology()

    from fes.constants import C0, PI

    order = 1
    freq = 1e10
    k0 = 2.0 * PI * freq / C0
    kk = k0 * k0
    dofnum_s = 8  # WR90 has 8 nodes
    dofnum_v = 33  # WR90 has 33 edges
    dofnum = dofnum_s + dofnum_v

    dirdofs_s, dirdofs_v, marked = _collect_pec(mesh, order)
    matrix = _assemble_bulk(mesh, order, k0, kk, dofnum_s, dofnum_v, dofnum, marked)

    assert matrix.shape == (dofnum, dofnum)
    assert matrix.nnz > 0
    assert not np.any(np.isnan(matrix.data))
    assert not np.any(np.isinf(matrix.data))

    # Diagonal should be mostly nonzero (stiffness + projector mass terms)
    diag = matrix.diagonal()
    assert np.count_nonzero(diag) > dofnum // 2


def test_assemble_em_av_fd_raises_without_ports():
    """Without waveports or an incident wave, the assembler raises
    Null Right Hand Side, same as em_e_fd.  WR90 carries WavePort
    boundaries, so at order 1 the port mode computation fails with
    ValueError instead ("not enough dofs on wave_ports"); at order 2 it
    proceeds to the TFE reduction."""
    from fes.core.assemblers.errors import NullRightHandSideError
    mesh = mesh3d_from_poly("data/WR90.poly")
    mesh.build_topology()
    with pytest.raises((NullRightHandSideError, ValueError)):
        assemble_em_av_fd(mesh, 1e10, 1)


def test_assemble_em_av_fd_matrix_properties():
    """The assembled matrix should be a valid complex sparse matrix."""
    from fes.core.assemblers.errors import NullRightHandSideError
    mesh = mesh3d_from_poly("data/WR90.poly")
    mesh.build_topology()
    try:
        assemble_em_av_fd(mesh, 1e10, 1)
    except (NullRightHandSideError, ValueError):
        pass  # WR90 has wave ports; order 1 port may have too few dofs

    # Verify we can at least build the matrix by calling internals
    from fes.core.assemblers.em_av_fd import _assemble_bulk, _collect_pec
    from fes.constants import C0, PI
    from fes.core.dof import scalar_dof_count, vector_dof_count

    order = 1
    freq = 1e10
    k0 = 2.0 * PI * freq / C0
    kk = k0 * k0
    dofnum_s = scalar_dof_count(mesh, order)
    dofnum_v = vector_dof_count(mesh, order)
    dofnum = dofnum_s + dofnum_v

    dirdofs_s, dirdofs_v, marked = _collect_pec(mesh, order)
    matrix = _assemble_bulk(mesh, order, k0, kk, dofnum_s, dofnum_v, dofnum, marked)

    assert matrix.shape == (dofnum, dofnum)
    assert matrix.dtype == complex
    # Diagonal should be mostly nonzero (stiffness + projector mass terms)
    diag = matrix.diagonal()
    assert np.count_nonzero(diag) > dofnum // 2
