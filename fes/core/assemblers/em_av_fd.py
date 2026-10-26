"""3D H(curl) x H(grad) A-V (magnetic vector potential + electric scalar
potential) assembly with a gauge-fixing projector.

Line-by-line port of the reference C++ ``assemble3d`` path together
with the projector and the wave-port TFE reduction from ``compute_wave_port_modes``.

The unknowns are ``A`` (magnetic vector potential, H(curl) edge DOFs) and
``V`` (electric scalar potential, H(grad) nodal DOFs).  With

``B = curl(A)``, ``E = -grad(V) - i*omega*A``, ``H = (1/mur) * curl(A)``

the weak form on one tetrahedron gives

``S = H(curl) stiffness weighted by 1/mur``   (on edge/A DOFs)
``T = H(curl) mass    weighted by (eps + i*sigma/omega)``  (on edge DOFs)

and the AV projector ``P`` (n_edge x (n_node + n_edge)):

``P[edge, node0] = +dir,  P[edge, node1] = -dir``        (discrete gradient)
``P[edge, n_node + edge] = -1.0`` (marked/gauge DoF) or ``k0`` (free DoF)

The element system on the combined [V, A] DoFs is::

    K = S_on_A_block - kk * P^T * T * P

Boundary conditions:

* ``PerfectE``: homogeneous Dirichlet on all AV DoFs of the face
  (both the scalar/V and vector/A parts).  Edge DoFs on PEC faces are
  *marked* in the projector, so their diagonal entry is -1 instead of k0.
* ``Radiation``: first-order ABC ``i*k0*Tt*sqrt(epsr/mur)`` on the A block.
* ``WavePort``/``LumpedPort``: transfinite-element modal reduction on the A
  block; the scalar/V DoFs of port faces remain regular DoFs.
* ``PerfectH``: treated like Radiation (zero tangential H, i.e. no A-block term).

With no waveports and no incident wave the reference throws
``Null Right Hand Side``; so does this port.

Deliberate difference from the oracle: the linear solve uses SuperLU on the
full symmetric expansion of the one-triangle matrix, same as em_e_fd.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg
import scipy.sparse as sp
import scipy.sparse.linalg

from ...constants import C0, PI, Z0
from ..dof import (
    MAX_ORDER,
    face_dof_map,
    scalar_dof_count,
    scalar_dof_map,
    vector_dof_count,
    vector_dof_map,
)
from ..element_matrix import (
    Material,
    _DNS_REF_2D,
    _checked_order,
    _jacobian,
    einc_face_vector,
    element_matrices_3d,
    element_matrices_face,
    quadrature_2d,
    shape_2d,
)
from .errors import NullRightHandSideError
from .em_e_fd import _full_symmetric, _dense_to_sparse

__all__ = ["AVSystem", "assemble_em_av_fd"]

#: ``DBL_MIN``, the reference's starting value for ``maxepsr``/``maxmur``.
_DBL_MIN = float.fromhex("0x0.0000000000001p-1022")

#: Local edge order of ``tetEdges`` (shared with mesh3d).
_LOCAL_EDGES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))

#: ``gamtol`` from ``eigen.cpp``: components below this become exactly zero.
_GAMMA_TOL = 1e-9


@dataclass
class AVPort:
    """One modal port after ``compute_wave_port_modes``.

    ``mode_vec`` is ``(n_port_vector_dofs, num_modes)`` and
    ``mode_vecdof`` holds the global A-block DOF numbers of the free port
    vector DoFs, both in the reference's order.
    """

    name: str
    label: int
    num_modes: int
    mode_beta: np.ndarray
    mode_vec: np.ndarray
    mode_vecf: np.ndarray
    mode_vecdof: np.ndarray
    edges: np.ndarray
    free: np.ndarray
    shift: float
    impedance: float = 0.0

    @property
    def n_modes(self) -> int:
        return int(self.mode_beta.size)

    @property
    def is_lumped(self) -> bool:
        return self.impedance > 0.0


@dataclass
class AVSystem:
    """Everything the 3D ``em_av_fd`` assembler produced.

    The global DoF layout is ``[V, A]``: the first ``dofnum_s`` DoFs are the
    scalar (H(grad)) unknowns, the next ``dofnum_v`` DoFs are the vector
    (H(curl)) unknowns.
    """

    mesh: object
    freq: float
    order: int
    k0: float
    kk: float
    dofnum_s: int
    dofnum_v: int
    dofnum: int
    A: sp.csr_matrix
    B: np.ndarray
    dirichlet_s: np.ndarray
    dirichlet_v: np.ndarray
    power: float = 1.0
    ports: List[AVPort] = field(default_factory=list)
    wave_ports_num: int = 0
    wave_ports_dofnum: int = 0
    nonwave_ids: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    Sol: Optional[np.ndarray] = None
    S: Optional[np.ndarray] = None

    # TFE reduction outputs (populated when wave ports are present)
    Meig: Optional[np.ndarray] = None
    MAcoeff: Optional[np.ndarray] = None
    B_exc: Optional[np.ndarray] = None
    A_wp: Optional[np.ndarray] = None
    A_wp_nn: Optional[np.ndarray] = None
    A_nn_wp: Optional[np.ndarray] = None
    Ad_wp: Optional[np.ndarray] = None
    Attred: Optional[np.ndarray] = None
    Atired: Optional[np.ndarray] = None
    top_left: Optional[np.ndarray] = None
    reduced_solution: Optional[np.ndarray] = None
    Sp: Optional[np.ndarray] = None

    # ConstE: Dirichlet DoFs and their prescribed E values.
    # ``dofnum_s + dirdofs_v`` entries with nonzero prescribed values need the
    # coupling to nonwave DoFs dropped (they are excluded from the system).
    constE_dirdofs_s: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    constE_dirdofs_v_free: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    constE_dirdofs_v_marked: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    constE_e_values: List[Tuple[int, float, float, float]] = field(default_factory=list)

    # ConstH: Neumann DoFs, specified H values, and outward normals.
    constH_dofs_v: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    constH_h_values: List[Tuple[int, float, float, float]] = field(default_factory=list)
    constH_normal_vecs: List[Tuple[int, float, float, float]] = field(default_factory=list)


# ── helpers ─────────────────────────────────────────────────────────────────


def _ensure_topology(mesh) -> None:
    if getattr(mesh, "tet_edges", None) is None:
        mesh.build_topology()
    if getattr(mesh, "face_adj_tet", None) is None:
        mesh.build_topology()


def _material_for_tet(mesh, tet_id: int) -> Material:
    return Material.from_solid(mesh.material_for_tet(int(tet_id)))


def _scalar_dof_count_3d(order: int) -> int:
    """Number of scalar (H(grad)) DOFs per tetrahedron at *order*."""
    return (order + 1) * (order + 2) * (order + 3) // 6


def _vector_dof_count_3d(order: int) -> int:
    """Number of vector (H(curl)) DOFs per tetrahedron at *order*."""
    return {1: 6, 2: 20, 3: 45, 4: 84}[int(order)]


# ── discrete gradient ───────────────────────────────────────────────────────


def _discrete_gradient_local(n_vector: int) -> np.ndarray:
    """Local discrete gradient ``G`` (n_vector x n_scalar_local).

    For each local edge ``(i, j)`` with ``i < j``::

        G[edge, node_i] = +1
        G[edge, node_j] = -1

    Matching ``AV_ProjectorMatrix`` in the oracle.  The edge direction is +1
    because both local and global edges are ascending.
    """
    n_scalar = _scalar_dof_count_3d(1)  # 4 nodal DOFs for the gradient block
    G = np.zeros((n_vector, n_scalar), dtype=float)
    for edge_idx, (i, j) in enumerate(_LOCAL_EDGES):
        G[edge_idx, i] = 1.0
        G[edge_idx, j] = -1.0
    return G


# ── PEC marking ─────────────────────────────────────────────────────────────


def _collect_pec(mesh, order: int) -> tuple:
    """Collect PEC boundary information.

    Returns ``(dirdofs_s, dirdofs_v, marked_edges)`` where:

    * ``dirdofs_s`` — scalar DoFs on PEC faces (Dirichlet for V)
    * ``dirdofs_v`` — vector DoFs on PEC faces (Dirichlet for A)
    * ``marked_edges`` — global vector DoFs on PEC faces (projector D = -1)
    """
    scalar_parts: List[np.ndarray] = []
    vector_parts: List[np.ndarray] = []
    marked_parts: List[np.ndarray] = []

    for bc in mesh.boundaries:
        if bc.type != "PerfectE":
            continue
        for fid in mesh.faces_for_boundary(bc):
            cdof = face_dof_map(mesh, order, int(fid))
            scalar_parts.append(cdof.s)
            vector_parts.append(cdof.v)
            marked_parts.append(cdof.v)

    if scalar_parts:
        dirdofs_s = np.unique(np.concatenate(scalar_parts))
        dirdofs_v = np.unique(np.concatenate(vector_parts))
        marked_edges = np.unique(np.concatenate(marked_parts))
    else:
        dirdofs_s = np.zeros(0, dtype=np.int64)
        dirdofs_v = np.zeros(0, dtype=np.int64)
        marked_edges = np.zeros(0, dtype=np.int64)
    return dirdofs_s, dirdofs_v, marked_edges


# ── ConstE / ConstH collection ──────────────────────────────────────────────


def _collect_constE(mesh, order: int) -> tuple:
    """Collect ConstE boundary information.

    Returns ``(dirdofs_s, dirdofs_v_free, dirdofs_v_marked, e_values)`` where:

    * ``dirdofs_s`` — scalar DoFs on ConstE faces (Dirichlet for V, value 0)
    * ``dirdofs_v_free`` — free vector DoFs on ConstE faces (Dirichlet for A,
      value ``E_specified / k0``)
    * ``dirdofs_v_marked`` — PEC-marked vector DoFs on ConstE faces
      (Dirichlet for A, value ``-E_specified``)
    * ``e_values`` — list of ``(face_id, Ex, Ey, Ez)`` for each ConstE face
    """
    scalar_parts: List[np.ndarray] = []
    vector_free_parts: List[np.ndarray] = []
    vector_marked_parts: List[np.ndarray] = []
    e_values: List[Tuple[int, float, float, float]] = []
    marked_set: set = set()

    # Need PEC marks to distinguish free vs marked A DoFs on ConstE faces.
    for bc in mesh.boundaries:
        if bc.type != "PerfectE":
            continue
        for fid in mesh.faces_for_boundary(bc):
            cdof = face_dof_map(mesh, order, int(fid))
            marked_set.update(int(v) for v in cdof.v)

    for bc in mesh.boundaries:
        if bc.type != "ConstE":
            continue
        e_val = bc.e_value or (1.0, 0.0, 0.0)
        for fid in mesh.faces_for_boundary(bc):
            fid = int(fid)
            cdof = face_dof_map(mesh, order, fid)
            scalar_parts.append(cdof.s)
            for v in cdof.v:
                vi = int(v)
                if vi in marked_set:
                    vector_marked_parts.append(np.array([vi]))
                else:
                    vector_free_parts.append(np.array([vi]))
            e_values.append((fid, float(e_val[0]), float(e_val[1]), float(e_val[2])))

    if scalar_parts:
        dirdofs_s = np.unique(np.concatenate(scalar_parts))
    else:
        dirdofs_s = np.zeros(0, dtype=np.int64)
    if vector_free_parts:
        dirdofs_v_free = np.unique(np.concatenate(vector_free_parts))
    else:
        dirdofs_v_free = np.zeros(0, dtype=np.int64)
    if vector_marked_parts:
        dirdofs_v_marked = np.unique(np.concatenate(vector_marked_parts))
    else:
        dirdofs_v_marked = np.zeros(0, dtype=np.int64)
    return dirdofs_s, dirdofs_v_free, dirdofs_v_marked, e_values


def _collect_constH(mesh, order: int) -> tuple:
    """Collect ConstH boundary information.

    Returns ``(dofs_v, h_values, normal_vecs)`` where:

    * ``dofs_v`` — vector DoFs on ConstH faces (Neumann for A)
    * ``h_values`` — list of ``(face_id, Hx, Hy, Hz)`` for each ConstH face
    * ``normal_vecs`` — list of ``(face_id, nx, ny, nz)`` for each ConstH face
    """
    vector_parts: List[np.ndarray] = []
    h_values: List[Tuple[int, float, float, float]] = []
    normal_vecs: List[Tuple[int, float, float, float]] = []

    for bc in mesh.boundaries:
        if bc.type != "ConstH":
            continue
        h_val = bc.h_value or (0.0, 0.0, 1.0)
        n_val = bc.normal_vec or (0.0, 0.0, 1.0)
        for fid in mesh.faces_for_boundary(bc):
            fid = int(fid)
            cdof = face_dof_map(mesh, order, fid)
            vector_parts.append(cdof.v)
            h_values.append((fid, float(h_val[0]), float(h_val[1]), float(h_val[2])))
            normal_vecs.append((fid, float(n_val[0]), float(n_val[1]), float(n_val[2])))

    if vector_parts:
        dofs_v = np.unique(np.concatenate(vector_parts))
    else:
        dofs_v = np.zeros(0, dtype=np.int64)
    return dofs_v, h_values, normal_vecs


def _const_face_vector_load(order, geo, int_node, C_vec) -> np.ndarray:
    """∫_face C · N_i dS for a constant tangential 3-vector C on a face.

    Returns a ``(numv,)`` complex vector indexed by the face's H(curl) DoFs.
    *C* must be tangential (``C · n = 0``); only its in-face components matter.
    """
    order = _checked_order(order)
    geo = np.asarray(geo, dtype=float)
    if geo.shape != (3, 3):
        raise ValueError("a face needs three nodes with three coordinates")
    C_vec = np.asarray(C_vec, dtype=float)

    v0 = geo[0].copy()
    v1 = geo[1].copy()
    v2 = geo[2].copy()
    v_in = (v0 + v1 + v2) / 3.0 - np.asarray(int_node, dtype=float)

    v1 = v1 - v0
    v2 = v2 - v0
    u = v1 / np.linalg.norm(v1)
    n = np.cross(v1, v2)
    n = n * np.dot(n, v_in)
    n = n / np.linalg.norm(n)
    v = np.cross(n, u)

    C_uv = np.array([np.dot(C_vec, u), np.dot(C_vec, v)], dtype=complex)

    c_geo2 = np.zeros((3, 2))
    c_geo2[1, 0] = np.dot(v1, u)
    c_geo2[2, 0] = np.dot(v2, u)
    c_geo2[2, 1] = np.dot(v2, v)
    det_j, inv_j = _jacobian(_DNS_REF_2D, c_geo2)

    points, weights = quadrature_2d(order + 1)
    numv = shape_2d(order, points[:1], inv_j).Nv.shape[1]

    f = np.zeros(numv, dtype=complex)
    for iq in range(points.shape[0]):
        basis = shape_2d(order, points[iq : iq + 1], inv_j)
        f += (basis.Nv[0] @ C_uv) * weights[iq]
    return f * det_j


# ── bulk assembly ───────────────────────────────────────────────────────────


def _assemble_bulk(
    mesh,
    order: int,
    k0: float,
    kk: float,
    dofnum_s: int,
    dofnum_v: int,
    dofnum: int,
    marked_edges: np.ndarray,
) -> sp.csr_matrix:
    """``S_on_A - kk * P^T * T * P`` over every tetrahedron.

    ``marked_edges`` is the set of global vector DoFs on PEC faces; their
    projector diagonal entry is -1 instead of k0.
    """
    marked_set = set(int(e) for e in marked_edges)
    parts = []
    n_vector_local = _vector_dof_count_3d(order)
    G_local = _discrete_gradient_local(n_vector_local)

    for tet_id in range(mesh.n_tetras):
        material = _material_for_tet(mesh, tet_id)
        lmat = element_matrices_3d(order, mesh.tet_geo(tet_id), material)
        s_block = lmat.S   # (n_vector_local, n_vector_local)
        t_block = lmat.T   # (n_vector_local, n_vector_local)

        s_dofs = scalar_dof_map(mesh, order, tet_id)     # (n_scalar_tet,)
        v_dofs = vector_dof_map(mesh, order, tet_id)     # (n_vector_tet,)

        n_scalar_tet = s_dofs.size
        n_vector_tet = v_dofs.size

        # Diagonal scaling D (n_vector_tet): -1 for marked PEC edges, k0 else.
        D = np.full(n_vector_tet, k0, dtype=float)
        for edge_idx in range(n_vector_tet):
            if int(v_dofs[edge_idx]) in marked_set:
                D[edge_idx] = -1.0

        # Projector P = [G, D]  (n_vector_tet x (n_scalar_tet + n_vector_tet))
        P = np.zeros((n_vector_tet, n_scalar_tet + n_vector_tet), dtype=complex)
        # The discrete gradient only relates the 4 nodal scalar DoFs to the
        # edges; higher-order scalar DoFs (edge-/face-interior) have no
        # gradient coupling and stay zero in P.
        n_scalar_nodal = _scalar_dof_count_3d(1)
        P[:, :n_scalar_nodal] = G_local[:n_vector_tet, :n_scalar_nodal]
        P[:, n_scalar_tet:] = np.diag(D)

        # TT = P^T * T * P on the combined [V, A] DoFs
        tt = P.conj().T @ (t_block @ P)

        # Element matrix on [V, A] x [V, A]:
        #   K[V, V] = -kk * TT[V, V]
        #   K[V, A] = -kk * TT[V, A]
        #   K[A, V] = -kk * TT[A, V]
        #   K[A, A] = S - kk * TT[A, A]
        kelem = np.zeros((n_scalar_tet + n_vector_tet, n_scalar_tet + n_vector_tet), dtype=complex)
        kelem[:n_scalar_tet, :n_scalar_tet] = -kk * tt[:n_scalar_tet, :n_scalar_tet]
        kelem[:n_scalar_tet, n_scalar_tet:] = -kk * tt[:n_scalar_tet, n_scalar_tet:]
        kelem[n_scalar_tet:, :n_scalar_tet] = -kk * tt[n_scalar_tet:, :n_scalar_tet]
        kelem[n_scalar_tet:, n_scalar_tet:] = s_block - kk * tt[n_scalar_tet:, n_scalar_tet:]

        # Higher-order scalar DoFs (edge-/face-interior at order >= 2) have no
        # gradient coupling in P, leaving them with zero rows/columns.  Add a
        # diagonal gauge term so they are constrained (V = 0 for these DoFs).
        for i in range(n_scalar_nodal, n_scalar_tet):
            kelem[i, i] -= kk

        # Combined DoF map: [scalar_dofs, dofnum_s + vector_dofs] -> global [V, A]
        cdof = np.concatenate([s_dofs, dofnum_s + v_dofs])

        # Upper triangle only, matching the reference's map accumulator.
        rows, cols = np.nonzero(cdof[:, None] <= cdof[None, :])
        parts.append((cdof[rows], cdof[cols], kelem[rows, cols]))

    return _dense_to_sparse(parts, dofnum)


# ── radiation ───────────────────────────────────────────────────────────────


def _add_radiation(
    mesh, matrix: sp.csr_matrix, order: int, k0: float, dofnum_s: int, dofnum: int
) -> sp.csr_matrix:
    """Add the first-order ABC on every ``Radiation`` face (A block only)."""
    parts = []
    for bc in mesh.boundaries:
        if bc.type not in ("Radiation", "PerfectH"):
            continue
        for fid in mesh.faces_for_boundary(bc):
            fid = int(fid)
            adjacent = mesh.face_adj_tet[fid]
            material = _material_for_tet(mesh, int(adjacent[0]))
            epsr = complex(material.epsr, -material.tand * material.epsr)
            mur = material.mur
            lmat = element_matrices_face(order, mesh.fac_geo(fid), mesh.int_node(fid))
            factor = complex(0.0, k0) * np.sqrt(epsr / mur)
            cdof = face_dof_map(mesh, order, fid)
            v_dofs = dofnum_s + cdof.v
            block = lmat.Tt * factor
            rows, cols = np.nonzero(v_dofs[:, None] <= v_dofs[None, :])
            parts.append((v_dofs[rows], v_dofs[cols], block[rows, cols]))
    if not parts:
        return matrix
    return (matrix + _dense_to_sparse(parts, dofnum)).tocsr()


# ── port eigenproblem ───────────────────────────────────────────────────────


def _port_eigenpairs(tmpA: np.ndarray, tmpB: np.ndarray, numt: int,
                     shift: float, nmodes: int) -> tuple:
    """Resolve ``num_modes`` modes of the generalized pencil ``tmpA x = l tmpB x``.

    Same approach as em_e_fd: LAPACK's ``ggev`` resolves the full pencil,
    modes whose H(curl) block is empty are skipped.
    """
    values, vectors = scipy.linalg.eig(tmpA, tmpB)
    distance = np.abs(values - shift)
    content = np.linalg.norm(vectors[:numt, :], axis=0)
    total = np.linalg.norm(vectors, axis=0)
    has_vector_part = content > 1e-8 * np.maximum(total, np.finfo(float).tiny)
    ranked = np.where(has_vector_part, distance, np.inf)
    selected = np.argsort(ranked, kind="stable")[:nmodes]
    return values[selected], vectors[:, selected]


def _port_modes_av(
    mesh,
    bc,
    order: int,
    k0: float,
    kk: float,
    freq: float,
    dirichlet_s: np.ndarray,
    dirichlet_v: np.ndarray,
    dofnum_s: int,
    dofnum_v: int,
    dofnum: int,
) -> AVPort:
    """``compute_wave_port_modes`` for a single boundary.

    Computes port modes on the A (vector/edge) DoFs of the port face, using
    the face matrices ``St``, ``Tt``, ``Tt2``, ``Sz``, ``Tz`` and ``G`` from
    the embedded-face element.  The scalar (V) DoFs on the port face remain
    regular DoFs in the global system.
    """
    faces = [int(fid) for fid in mesh.faces_for_boundary(bc)]
    cdofs = [face_dof_map(mesh, order, fid) for fid in faces]

    edge_parts = [cdof.v for cdof in cdofs]
    node_parts = [cdof.s for cdof in cdofs]
    tmp_edges = (
        np.unique(np.concatenate(edge_parts)) if edge_parts else np.zeros(0, np.int64)
    )
    tmp_nodes = (
        np.unique(np.concatenate(node_parts)) if node_parts else np.zeros(0, np.int64)
    )

    # Global DoF -> index inside the port's local list.
    tmpdofv = np.zeros(dofnum_v, dtype=np.int64)
    tmpdofv[tmp_edges - dofnum_s] = np.arange(tmp_edges.size)
    tmpdofs = np.zeros(max(int(tmp_nodes.max()) + 1 if tmp_nodes.size else 1, 1),
                       dtype=np.int64)
    tmpdofs[tmp_nodes] = np.arange(tmp_nodes.size)

    tmpDirdofv = np.flatnonzero(~np.isin(tmp_edges, dirichlet_v))
    tmpDirdofs = np.flatnonzero(~np.isin(tmp_nodes, dirichlet_s))

    nport = tmp_edges.size
    nnodes = tmp_nodes.size
    tmpSt = np.zeros((nport, nport), dtype=complex)
    tmpTt = np.zeros((nport, nport), dtype=complex)
    tmpTt2 = np.zeros((nport, nport), dtype=complex)
    tmpSz = np.zeros((nnodes, nnodes), dtype=complex)
    tmpTz = np.zeros((nnodes, nnodes), dtype=complex)
    tmpG = np.zeros((nport, nnodes), dtype=complex)
    maxepsr, maxmur = _DBL_MIN, _DBL_MIN

    for fid, cdof in zip(faces, cdofs):
        adjacent = mesh.face_adj_tet[fid]
        material = _material_for_tet(mesh, int(adjacent[0]))
        epsr = complex(material.epsr, -material.tand * material.epsr)
        mur = material.mur
        maxepsr = max(maxepsr, epsr.real)
        maxmur = max(maxmur, mur)
        lmat = element_matrices_face(order, mesh.fac_geo(fid), mesh.int_node(fid))

        vi = tmpdofv[cdof.v - dofnum_s]
        si = tmpdofs[cdof.s]
        block = np.ix_(vi, vi)
        tmpSt[block] += lmat.St / mur
        tmpTt[block] += lmat.Tt * epsr
        tmpTt2[block] += lmat.Tt / mur
        sb = np.ix_(si, si)
        tmpSz[sb] += lmat.Sz / mur
        tmpTz[sb] += lmat.Tz * epsr
        tmpG[np.ix_(vi, si)] += lmat.G / mur

    tmpSt = tmpSt[np.ix_(tmpDirdofv, tmpDirdofv)]
    tmpTt = tmpTt[np.ix_(tmpDirdofv, tmpDirdofv)]
    tmpTt2 = tmpTt2[np.ix_(tmpDirdofv, tmpDirdofv)]
    tmpSz = tmpSz[np.ix_(tmpDirdofs, tmpDirdofs)]
    tmpTz = tmpTz[np.ix_(tmpDirdofs, tmpDirdofs)]
    tmpG = tmpG[np.ix_(tmpDirdofv, tmpDirdofs)]

    numt, numz = tmpSt.shape[0], tmpSz.shape[0]
    numtot = numt + numz

    tmpA = np.zeros((numtot, numtot), dtype=complex)
    tmpB = np.zeros((numtot, numtot), dtype=complex)
    tmpA[:numt, :numt] = tmpSt - kk * tmpTt
    tmpB[:numt, :numt] = tmpTt2
    if numz > 0:
        tmpB[numt:, numt:] = tmpSz - kk * tmpTz
        tmpB[:numt, numt:] = tmpG
        tmpB[numt:, :numt] = tmpG.T

    if numtot < 5:
        raise ValueError("Not enough dofs on wave_ports")

    shift = -kk * maxepsr * maxmur
    nmodes = max(int(bc.num_modes), 1)

    values, vectors = _port_eigenpairs(tmpA, tmpB, numt, shift, nmodes)

    beta = np.sqrt(np.asarray(values, dtype=complex))
    real = np.abs(beta.real)
    imag = np.abs(beta.imag)
    real[real < _GAMMA_TOL] = 0.0
    imag[imag < _GAMMA_TOL] = 0.0
    beta = real + 1j * imag

    order_idx = np.argsort(-np.abs(beta), kind="stable")
    beta = beta[order_idx]
    vectors = np.asarray(vectors, dtype=complex)[:, order_idx]

    gram = vectors.T @ tmpB @ vectors
    sqrt_gram = np.sqrt(gram)
    scale = np.linalg.inv(sqrt_gram)
    mode_vec = vectors[:numt, :] @ scale

    coeff = np.zeros((beta.size, beta.size), dtype=complex)
    np.fill_diagonal(coeff, np.sqrt(complex(0.0, k0 * Z0) / beta))
    mode_vecf = tmpB @ (vectors @ coeff @ scale)
    mode_vecf = mode_vecf[:numt, :]

    for m in range(mode_vec.shape[1]):
        pivot = int(np.argmax(np.abs(mode_vec[:, m].real)))
        if mode_vec[pivot, m].real < 0:
            mode_vec[:, m] *= -1.0
            mode_vecf[:, m] *= -1.0

    mode_vecdof = tmp_edges[tmpDirdofv]
    impedance = 0.0
    if bc.type == "LumpedPort":
        impedance = float(getattr(bc, "impedance", 50.0))

    return AVPort(
        name=bc.name,
        label=int(bc.label),
        num_modes=nmodes,
        mode_beta=beta,
        mode_vec=mode_vec,
        mode_vecf=mode_vecf,
        mode_vecdof=mode_vecdof,
        edges=tmp_edges,
        free=tmpDirdofv,
        shift=shift,
        impedance=impedance,
    )


# ── driver ──────────────────────────────────────────────────────────────────


def assemble_em_av_fd(
    mesh,
    freq: float,
    order: int = 2,
    *,
    power: float = 1.0,
    solve: bool = True,
    e_inc: Optional[Sequence[float]] = None,
    k_inc: Optional[Sequence[float]] = None,
) -> AVSystem:
    """Assemble and (by default) solve the 3D H(curl) x H(grad) A-V system.

    ``e_inc``/``k_inc`` are the incident plane-wave parameters.  As in
    ``assemble_em_e_fd`` the incident-wave branch runs only when the model has
    no waveports.
    """
    order = int(order)
    if order < 1 or order > MAX_ORDER:
        raise ValueError(f"polynomial order {order} not yet implemented")
    freq = float(freq)
    _ensure_topology(mesh)

    k0 = 2.0 * PI * freq / C0
    kk = k0 * k0
    dofnum_s = scalar_dof_count(mesh, order)
    dofnum_v = vector_dof_count(mesh, order)
    dofnum = dofnum_s + dofnum_v

    # Determine PEC marking before assembly so the projector diagonal is correct.
    dirdofs_s, dirdofs_v, marked_edges = _collect_pec(mesh, order)

    matrix = _assemble_bulk(
        mesh, order, k0, kk, dofnum_s, dofnum_v, dofnum, marked_edges
    )
    matrix = _add_radiation(mesh, matrix, order, k0, dofnum_s, dofnum)

    # ConstE / ConstH boundary conditions.
    constE_dirdofs_s, constE_dirdofs_v_free, constE_dirdofs_v_marked, constE_e_values = (
        _collect_constE(mesh, order)
    )
    constH_dofs_v, constH_h_values, constH_normal_vecs = _collect_constH(mesh, order)

    # Dirichlet mask over the full combined [V, A] DoF space.
    doftoLeave = np.ones(dofnum, dtype=bool)
    doftoLeave[:dofnum_s][dirdofs_s] = False
    doftoLeave[dofnum_s:][dirdofs_v] = False
    # ConstE: V = 0 on ConstE faces; A is prescribed (E/k0 free, -E marked).
    doftoLeave[:dofnum_s][constE_dirdofs_s] = False
    doftoLeave[dofnum_s:][constE_dirdofs_v_free] = False
    doftoLeave[dofnum_s:][constE_dirdofs_v_marked] = False

    ports: List[AVPort] = []
    for bc in mesh.boundaries:
        if bc.type not in ("WavePort", "LumpedPort"):
            continue
        port = _port_modes_av(
            mesh, bc, order, k0, kk, freq,
            dirdofs_s, dirdofs_v, dofnum_s, dofnum_v, dofnum,
        )
        # Mark port A DoFs as not nonwave: they will be replaced by modal DoFs.
        port_a_dofs = dofnum_s + port.edges
        doftoLeave[port_a_dofs] = False
        ports.append(port)

    wave_ports_num = sum(port.n_modes for port in ports)
    wave_ports_dofnum = sum(port.mode_vec.shape[0] for port in ports)
    nonwave_ids = np.flatnonzero(doftoLeave)
    dofreal = int(nonwave_ids.size + wave_ports_num)

    result = AVSystem(
        mesh=mesh,
        freq=freq,
        order=order,
        k0=k0,
        kk=kk,
        dofnum_s=dofnum_s,
        dofnum_v=dofnum_v,
        dofnum=dofnum,
        A=matrix,
        B=np.zeros((dofreal, wave_ports_num), dtype=complex),
        dirichlet_s=dirdofs_s,
        dirichlet_v=dirdofs_v,
        wave_ports_num=wave_ports_num,
        wave_ports_dofnum=wave_ports_dofnum,
        nonwave_ids=nonwave_ids,
        ports=ports,
        power=float(power),
        constE_dirdofs_s=constE_dirdofs_s,
        constE_dirdofs_v_free=constE_dirdofs_v_free,
        constE_dirdofs_v_marked=constE_dirdofs_v_marked,
        constE_e_values=constE_e_values,
        constH_dofs_v=constH_dofs_v,
        constH_h_values=constH_h_values,
        constH_normal_vecs=constH_normal_vecs,
    )

    if wave_ports_num > 0:
        return _reduce_and_solve(
            result, matrix, wave_ports_num, wave_ports_dofnum,
            dirdofs_v, nonwave_ids, solve=solve,
            constH_dofs_v=constH_dofs_v,
            constH_h_values=constH_h_values,
            constH_normal_vecs=constH_normal_vecs,
        )
    if e_inc is not None and k_inc is not None:
        return _incident_solve(
            result, matrix, dofnum_s, dirdofs_s, dirdofs_v, e_inc, k_inc, solve=solve
        )
    raise NullRightHandSideError("Null Right Hand Side")


def _incident_solve(
    result: AVSystem,
    matrix: sp.csr_matrix,
    dofnum_s: int,
    dirdofs_s: np.ndarray,
    dirdofs_v: np.ndarray,
    e_inc: Sequence[float],
    k_inc: Sequence[float],
    *,
    solve: bool,
) -> AVSystem:
    """Incident-plane-wave right-hand side on ``Radiation`` faces.

    Zeroes the Dirichlet rows and columns, sets their diagonal to one, and
    solves the full combined [V, A] system.
    """
    mesh = result.mesh
    order = result.order
    k0 = result.k0
    k_einc = np.asarray(k_inc, dtype=float) * k0
    pol_einc = np.asarray(e_inc, dtype=float) / np.sqrt(2.0)

    rhs = np.zeros((result.dofnum, 1), dtype=complex)
    for bc in mesh.boundaries:
        if bc.type != "Radiation":
            continue
        for fid in mesh.faces_for_boundary(bc):
            fid = int(fid)
            vector = einc_face_vector(
                order,
                mesh.fac_geo(fid),
                mesh.int_node(fid),
                k_einc,
                pol_einc,
            )
            cdof = face_dof_map(mesh, order, fid)
            # Face vector DoFs map to the A block: dofnum_s + cdof.v
            rhs[dofnum_s + cdof.v, 0] += vector * complex(0.0, k0)

    # ConstH Neumann RHS on A DoFs (full-system incident-wave path).
    if result.constH_dofs_v is not None and result.constH_dofs_v.size > 0:
        eta0_over_k0 = Z0 / k0
        marked_set = set(int(v) for v in dirdofs_v)
        for face_idx in range(len(result.constH_h_values)):
            fid, Hx, Hy, Hz = result.constH_h_values[face_idx]
            _, _, nx, ny, nz = result.constH_normal_vecs[face_idx]
            H_vec = np.array([Hx, Hy, Hz])
            n_vec = np.array([nx, ny, nz])
            C = np.cross(n_vec, H_vec)
            cdof = face_dof_map(mesh, order, fid)
            load_face = _const_face_vector_load(
                order, mesh.fac_geo(fid), mesh.int_node(fid), C
            )
            for local_idx in range(cdof.v.size):
                global_v = int(cdof.v[local_idx])
                D_val = -1.0 if global_v in marked_set else k0
                rhs[dofnum_s + global_v, 0] += (
                    (-eta0_over_k0) * D_val * load_face[local_idx]
                )

    result.B = rhs
    result.A = matrix
    result.dofreal = result.dofnum  # incident-wave keeps every DoF

    if not np.any(rhs != 0):
        raise NullRightHandSideError("Null Right Hand Side")
    if not solve:
        return result

    # ConstE DoFs are also Dirichlet in the full-system path.
    all_dirichlet = np.concatenate([dirdofs_s, dofnum_s + dirdofs_v])
    if result.constE_dirdofs_s.size:
        all_dirichlet = np.concatenate([all_dirichlet, result.constE_dirdofs_s])
    if result.constE_dirdofs_v_free.size:
        all_dirichlet = np.concatenate(
            [all_dirichlet, dofnum_s + result.constE_dirdofs_v_free]
        )
    if result.constE_dirdofs_v_marked.size:
        all_dirichlet = np.concatenate(
            [all_dirichlet, dofnum_s + result.constE_dirdofs_v_marked]
        )
    all_dirichlet = np.unique(all_dirichlet)
    full = _full_symmetric(matrix).tolil()
    full[all_dirichlet, :] = 0.0
    full[:, all_dirichlet] = 0.0
    full[all_dirichlet, all_dirichlet] = 1.0
    full = full.tocsr()

    lu = scipy.sparse.linalg.splu(full.tocsc())
    result.Sol = lu.solve(rhs)

    # ConstE prescribed A values after solve (V = 0 is already correct).
    for fid, Ex, Ey, Ez in result.constE_e_values:
        E_vec = np.array([Ex, Ey, Ez], dtype=complex)
        cdof = face_dof_map(mesh, order, fid)
        geo = mesh.fac_geo(fid)
        t_edge = geo[1] - geo[0]
        t_edge = t_edge / np.linalg.norm(t_edge)
        E_tan = float(np.dot(E_vec.real, t_edge)) + 1j * float(
            np.dot(E_vec.imag, t_edge)
        )
        free_set = set(int(v) for v in result.constE_dirdofs_v_free)
        marked_set_e = set(int(v) for v in result.constE_dirdofs_v_marked)
        for v in cdof.v:
            dof = int(v)
            if dof in free_set:
                result.Sol[dofnum_s + dof, 0] = E_tan / k0
            elif dof in marked_set_e:
                result.Sol[dofnum_s + dof, 0] = -E_tan
    return result


def _reduce_and_solve(
    result: AVSystem,
    matrix: sp.csr_matrix,
    wave_ports_num: int,
    wave_ports_dofnum: int,
    dirdofs_v: np.ndarray,
    nonwave_ids: np.ndarray,
    *,
    solve: bool,
    constH_dofs_v: Optional[np.ndarray] = None,
    constH_h_values: Optional[List[Tuple[int, float, float, float]]] = None,
    constH_normal_vecs: Optional[List[Tuple[int, float, float, float]]] = None,
) -> AVSystem:
    """Build the TFE-reduced AV system, its right-hand side, and optionally solve.

    The reduced system has size ``(wave_ports_num + n_nonwave)`` where the
    nonwave DoFs include all scalar (V) DoFs plus the non-port vector (A)
    DoFs.  Port A DoFs are replaced by modal DoFs.
    """
    k0, power = result.k0, result.power
    wpn, wpdofnum = wave_ports_num, wave_ports_dofnum
    nnum = int(nonwave_ids.size)
    dofnum_s = result.dofnum_s
    dofnum = result.dofnum
    dofreal = wpn + nnum
    mesh = result.mesh

    # Marked (PEC) A DoFs: projector diagonal is -1 instead of k0.
    marked_set = set(int(v) for v in dirdofs_v)

    # ── build meig: modal amplitudes -> port A DoFs ──────────────────────
    meig = np.zeros((wpn, wpdofnum), dtype=complex)
    macoeff = np.zeros((wpn, wpn), dtype=complex)
    b_exc = np.zeros((wpn, wpn), dtype=complex)
    idx = jdx = 0
    for port in result.ports:
        impedance = port.impedance if port.is_lumped else Z0
        for m in range(port.n_modes):
            sqrt_beta = np.sqrt(
                complex(0.0, k0 * impedance * power) / port.mode_beta[m]
            )
            macoeff[idx, idx] = complex(0.0, k0 * impedance * power)
            b_exc[idx, idx] = complex(0.0, 2.0 * k0 * impedance * power)
            meig[idx, jdx:jdx + port.mode_vec.shape[0]] = (
                sqrt_beta * port.mode_vec[:, m]
            )
            idx += 1
        jdx += port.mode_vec.shape[0]

    # ── collect port A DoF numbers ───────────────────────────────────────
    wp_ids = np.zeros(0, dtype=np.int64)
    for port in result.ports:
        wp_ids = np.concatenate([wp_ids, np.asarray(port.mode_vecdof, dtype=np.int64)])

    # ── Step 1: interior block, upper triangle (before diagonal clearing) ──
    parts = []
    for wi in range(nnum):
        rows = nonwave_ids[wi]
        for wj in range(wi, nnum):
            val = matrix[rows, nonwave_ids[wj]]
            if val != 0:
                parts.append(
                    (np.array([wpn + wi]), np.array([wpn + wj]), np.array([val]))
                )

    # ── Step 2: save and clear the diagonal of the assembled matrix ──────
    diagonal = np.asarray(matrix.diagonal(), dtype=complex)
    zeroed = matrix.tolil()
    zeroed.setdiag(0.0)
    zeroed = zeroed.tocsr()
    zeroed.eliminate_zeros()

    # ── Step 3: extract the port blocks from the zeroed matrix ───────────
    wnum = int(wp_ids.size)
    a_wp = np.zeros((wnum, wnum), dtype=complex)
    a_wp_nn = np.zeros((wnum, nnum), dtype=complex)
    a_nn_wp = np.zeros((nnum, wnum), dtype=complex)
    for wi in range(wnum):
        for wj in range(wi, wnum):
            a_wp[wi, wj] = zeroed[dofnum_s + wp_ids[wi], dofnum_s + wp_ids[wj]]
        for nj in range(nnum):
            val = zeroed[dofnum_s + wp_ids[wi], nonwave_ids[nj]]
            if val != 0:
                a_wp_nn[wi, nj] = val
            val = zeroed[nonwave_ids[nj], dofnum_s + wp_ids[wi]]
            if val != 0:
                a_nn_wp[nj, wi] = val

    ad_wp = np.zeros((wnum, wnum), dtype=complex)
    for wi in range(wnum):
        ad_wp[wi, wi] = diagonal[dofnum_s + wp_ids[wi]]

    # ── TFE modal reduction ──────────────────────────────────────────────
    attred = meig @ a_wp + meig @ a_wp.conj().T + meig @ ad_wp
    top_left = attred @ meig.conj().T + macoeff
    atired = meig @ a_wp_nn + meig @ a_nn_wp.conj().T

    # Top-left (lower triangle, j <= i) and top-right coupling blocks.
    for i in range(wpn):
        for j in range(i + 1):
            parts.append((np.array([i]), np.array([j]), np.array([top_left[i, j]])))
    for i in range(wpn):
        for j in range(nnum):
            parts.append(
                (np.array([i]), np.array([wpn + j]), np.array([atired[i, j]]))
            )

    reduced = _dense_to_sparse(parts, dofreal)
    reduced.eliminate_zeros()

    result.A = reduced
    result.Meig = meig
    result.MAcoeff = macoeff
    result.B_exc = b_exc
    result.A_wp = a_wp
    result.A_wp_nn = a_wp_nn
    result.A_nn_wp = a_nn_wp
    result.Ad_wp = ad_wp
    result.Attred = attred
    result.Atired = atired
    result.top_left = top_left
    result.B = np.zeros((dofreal, wpn), dtype=complex)
    for i in range(wpn):
        result.B[i, i] = b_exc[i, i]

    # ── ConstH Neumann RHS on A DoFs ─────────────────────────────────────
    # rhs_A[e] = -(eta0/k0) * D[e] * ∫_face (n × H) · N_e dS
    # where D[e] = -1 for marked (PEC) edges, k0 for free edges.
    # Port A DoFs are projected to modal DoFs via meig.T;
    # nonwave A DoFs are added directly.
    if constH_dofs_v is not None and constH_dofs_v.size > 0:
        eta0_over_k0 = Z0 / k0
        rhs_constH = np.zeros(result.dofnum_v, dtype=complex)
        order = result.order
        for face_idx in range(len(constH_h_values)):
            fid, Hx, Hy, Hz = constH_h_values[face_idx]
            _, _, nx, ny, nz = constH_normal_vecs[face_idx]
            H_vec = np.array([Hx, Hy, Hz])
            n_vec = np.array([nx, ny, nz])
            C = np.cross(n_vec, H_vec)  # n × H, tangential to the face
            cdof = face_dof_map(mesh, order, fid)
            load_face = _const_face_vector_load(
                order, mesh.fac_geo(fid), mesh.int_node(fid), C
            )
            for local_idx in range(cdof.v.size):
                global_v = int(cdof.v[local_idx])
                D_val = -1.0 if global_v in marked_set else k0
                rhs_constH[global_v] += (-eta0_over_k0) * D_val * load_face[local_idx]

        # Port A DoF global numbers (in the A block, offset by dofnum_s).
        port_a_dofs = np.zeros(0, dtype=np.int64)
        for port in result.ports:
            port_a_dofs = np.concatenate(
                [port_a_dofs, np.asarray(port.mode_vecdof, dtype=np.int64)]
            )

        # Nonwave A DoFs: direct addition to reduced RHS.
        for wi in range(nnum):
            gid = int(nonwave_ids[wi])
            if gid >= dofnum_s:
                a_dof = gid - dofnum_s
                result.B[wpn + wi, :] += rhs_constH[a_dof]

        # Port A DoFs: project to modal DoFs via meig.T.
        if port_a_dofs.size > 0:
            port_a_load = rhs_constH[port_a_dofs.astype(int)]
            modal_load = meig.T @ port_a_load
            for i in range(wpn):
                result.B[i, :] += modal_load[i]

    if not solve:
        return result

    full = _full_symmetric(reduced)
    lu = scipy.sparse.linalg.splu(full.tocsc())
    solution = lu.solve(result.B)
    result.reduced_solution = solution
    result.Sp = solution[:wpn, :]
    result.Sol = _extract_solution_full(result, solution)
    result.S = (result.Sp - np.eye(wpn)) * np.sqrt(power)
    return result


def _extract_solution_full(
    system: AVSystem, reduced: np.ndarray
) -> np.ndarray:
    """``solver::extract_solution_full``: expand the reduced AV solution.

    ``reduced`` is the ``(wave_ports_num + Nonwave, wave_ports_num)`` solution
    of the TFE system.  The reference stores it back into a ``(dofnum, n_rhs)``
    array: the interior rows go to their global DoF numbers, the modal rows to
    ``Sp``, and every free port DoF is rebuilt from the modal amplitudes with
    the ``sqrt(j*k0*Z/beta)`` weight.  Dirichlet DoFs stay zero.
    """
    wpn = int(system.wave_ports_num)
    dofnum_s = system.dofnum_s
    k0 = system.k0
    solution = np.zeros((system.dofnum, reduced.shape[1]), dtype=complex)
    solution[system.nonwave_ids, :] = reduced[wpn:, :]

    # ConstE Dirichlet values: V = 0 on ConstE faces;
    # A = E/k0 for free edges, A = -E for marked edges (E is tangential).
    constE_free_set = set(int(v) for v in system.constE_dirdofs_v_free)
    constE_marked_set = set(int(v) for v in system.constE_dirdofs_v_marked)
    for fid, Ex, Ey, Ez in system.constE_e_values:
        E_vec = np.array([Ex, Ey, Ez], dtype=complex)
        cdof = face_dof_map(system.mesh, system.order, fid)
        # Scalar V DoFs: V = 0 (already zero from initialization).
        solution[cdof.s.astype(int), :] = 0.0
        # Vector A DoFs: project tangential E onto the face's first edge
        # direction as the local tangent for each edge DoF.
        geo = system.mesh.fac_geo(fid)
        t_edge = geo[1] - geo[0]
        t_edge = t_edge / np.linalg.norm(t_edge)
        E_tan = float(np.dot(E_vec.real, t_edge)) + 1j * float(
            np.dot(E_vec.imag, t_edge)
        )
        for local_idx in range(cdof.v.size):
            dof = int(cdof.v[local_idx])
            if dof in constE_free_set:
                solution[dofnum_s + dof, :] = E_tan / k0
            elif dof in constE_marked_set:
                solution[dofnum_s + dof, :] = -E_tan

    mode_offset = 0
    for port in system.ports:
        impedance = port.impedance if port.is_lumped else Z0
        jk0z = complex(0.0, 2.0 * PI * system.freq / C0 * impedance * system.power)
        for m in range(port.n_modes):
            weight = np.sqrt(jk0z / port.mode_beta[m])
            solution[port.mode_vecdof, :] += (
                weight * np.outer(port.mode_vec[:, m], reduced[mode_offset + m, :])
            )
        mode_offset += port.n_modes
    return solution
