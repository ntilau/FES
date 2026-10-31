"""2D transmission-line cross-section eigenmode analysis.

The formulation is the same as the ``em_e_fd`` wave-port mode computation
(``assembler::compute_waveport_modes``): the scalar (H1) block is eliminated
with the Zhu-Cangellaris Schur complement and the edge-only generalized
pencil is solved

.. math::

    A_{tt} v = \\lambda B_{schur} v, \\qquad
    A_{tt} = S_t - k_0^2 T_t, \\qquad
    B_{schur} = T_{t2} - G (S_z - k_0^2 T_z)^{-1} G^H.

The physical modes (``k_c^2 = k_0^2 + \\lambda > 0``) are ordered TE10-first by
ascending cutoff wavenumber and reported as

.. math::

    \\beta = \\operatorname{Re}\\sqrt{k_0^2 - k_c^2}, \\qquad
    \\alpha = \\operatorname{Im}\\sqrt{k_0^2 - k_c^2}, \\qquad
    k_c = \\sqrt{|k_0^2 - (\\beta + j\\alpha)^2|}.

``TLEigSystem.mode_beta`` stores ``beta + 1j*alpha`` (the waveport
convention: real part is propagation, imaginary part attenuation).  The mode
vectors are B-orthonormalized exactly as in ``AssLin.cpp``

.. math::

    \\text{ModeVec} = V \\, (V^T B_{schur} V)^{-1/2},

and the modal coefficients use ``sqrt(j k_0 Z_0 / beta)``.

Dirichlet boundaries
--------------------

``PerfectE`` mesh edges constrain both the H(curl) edge DOFs and the H1 trace
DOFs on that edge (vertices and edge-interior levels at order >= 2).  As in
the 3D waveport path, face-interior scalar DOFs are not removed.

Solver
------

``"auto"`` and ``"arpack"`` try shift-invert ARPACK on
``A_tt/shift - B_schur`` with ``which="LM"`` and fall back to the dense
generalized solve when ARPACK does not converge or returns too few physical
modes.  ``"dense"`` forces the dense solve.  ``"arnoldi"`` is a deprecated
alias for ARPACK.  The default shift is ``-k0^2 * maxepsr * maxmur``.

The Schur complement is dense (``numt x numt``), exactly as in the waveport
formulation; very large cross-sections (e.g. ``Coaxial``) need correspondingly
large memory.

Compatibility
-------------

``ortho``, ``use_t_block`` and ``t_block`` are accepted but have no effect:
the Schur-complement formulation needs neither the Cangellaris ``OO``
projection nor the extended T-matrix variant.  ``prev_beta`` still reorders
the selected modes by nearest ``|mode_beta|`` across a frequency sweep.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np
import scipy.sparse as sp

from ...constants import C0, PI
from ..dof import face_dof_map, scalar_dof_count, vector_dof_count
from ..element_matrix import element_matrices_2d_mixed
from .port_pencil import (
    _DBL_MIN,
    _GAMMA_TOL,
    _port_reduced_pencil,
    postprocess_modes,
    solve_reduced_pencil,
)

__all__ = ["TLEigSystem", "assemble_em_e_tl_eig"]


@dataclass
class TLEigSystem:
    """Everything the cross-section eigenproblem produced at one frequency."""

    mesh: object
    freq: float
    order: int
    k0: float
    kk: float
    dof: int
    n_edge_dofs: int
    n_node_dofs: int
    free_edges: np.ndarray
    dirichlet: np.ndarray
    shift: float
    #: ``beta + 1j*alpha`` per mode (waveport convention), TE10-first.
    mode_beta: np.ndarray = field(default_factory=lambda: np.zeros(0, complex))
    #: Actual solver path: ``"dense"`` or ``"arpack"``.
    solver: str = "dense"
    #: Free scalar (H1) node DOFs after PerfectE elimination.
    free_nodes: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    #: Scalar DOFs constrained by PerfectE boundaries.
    dirichlet_nodes: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    #: B-orthonormalized mode vectors ``(numt, n_modes)``.
    mode_vec: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), complex))
    #: Modal coefficient vectors ``(numt, n_modes)``.
    mode_vecf: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), complex))
    #: Global edge DOF numbers matching ``mode_vec`` rows (``free_edges``).
    mode_vecdof: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    #: Sparse blocks, kept for inspection and for the parity fixtures.
    St: Optional[sp.csr_matrix] = None
    Tt: Optional[sp.csr_matrix] = None
    Tt2: Optional[sp.csr_matrix] = None
    Sz: Optional[sp.csr_matrix] = None
    Tz: Optional[sp.csr_matrix] = None
    G: Optional[sp.csr_matrix] = None

    @property
    def n_modes(self) -> int:
        return int(self.mode_beta.size)

    @property
    def beta(self) -> np.ndarray:
        return np.real(self.mode_beta)

    @property
    def alpha(self) -> np.ndarray:
        return np.imag(self.mode_beta)

    @property
    def kc(self) -> np.ndarray:
        return np.sqrt(np.abs(self.k0 * self.k0 - self.mode_beta**2))

    @property
    def propagating(self) -> np.ndarray:
        return (np.abs(self.alpha) < _GAMMA_TOL) & (self.beta > 0.0)

    @property
    def S(self) -> None:
        """The cross-section problem produces no S-parameters."""
        return None



def _bc_type_map(boundaries: Sequence) -> Dict[int, str]:
    """``std::map<label, type>``: a later boundary with the same label wins."""
    types: Dict[int, str] = {}
    for bc in boundaries:
        types[int(bc.label)] = bc.type
    return types


def _triplets(rows: np.ndarray, cols: np.ndarray, block: np.ndarray):
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    values = np.asarray(block, dtype=complex)
    return (
        np.repeat(rows, cols.size),
        np.tile(cols, rows.size),
        values.ravel(),
    )


def _add(pieces, shape) -> sp.csr_matrix:
    rows = np.concatenate([piece[0] for piece in pieces]) if pieces else np.zeros(0, np.int64)
    cols = np.concatenate([piece[1] for piece in pieces]) if pieces else np.zeros(0, np.int64)
    vals = np.concatenate([piece[2] for piece in pieces]) if pieces else np.zeros(0, complex)
    matrix = sp.coo_matrix((vals, (rows, cols)), shape=shape, dtype=complex)
    matrix.sum_duplicates()
    return matrix.tocsr()


def _assemble_blocks(mesh, order: int):
    """Assemble the six global blocks with the oracle's material scaling."""
    n_edges = vector_dof_count(mesh, order)
    n_nodes = scalar_dof_count(mesh, order)

    st_parts = []
    tt_parts = []
    tt2_parts = []
    sz_parts = []
    tz_parts = []
    g_parts = []
    maxepsr = _DBL_MIN
    maxmur = _DBL_MIN

    for face_id in range(mesh.n_faces):
        epsr, mur = mesh.material_properties(face_id)
        maxepsr = max(maxepsr, epsr)
        maxmur = max(maxmur, mur)

        geo = np.asarray(mesh.nodes[mesh.faces[face_id], :2], dtype=float)
        element = element_matrices_2d_mixed(order, geo)
        cdof = face_dof_map(mesh, order, face_id)
        v, s = cdof.v, cdof.s

        st_parts.append(_triplets(v, v, element.St / mur))
        tt_parts.append(_triplets(v, v, element.Tt * epsr))
        tt2_parts.append(_triplets(v, v, element.Tt / mur))
        sz_parts.append(_triplets(s, s, element.Sz / mur))
        tz_parts.append(_triplets(s, s, element.Tz * epsr))
        g_parts.append(_triplets(v, s, element.G / mur))

    return (
        _add(st_parts, (n_edges, n_edges)),
        _add(tt_parts, (n_edges, n_edges)),
        _add(tt2_parts, (n_edges, n_edges)),
        _add(sz_parts, (n_nodes, n_nodes)),
        _add(tz_parts, (n_nodes, n_nodes)),
        _add(g_parts, (n_edges, n_nodes)),
        maxepsr,
        maxmur,
    )


def _dirichlet_dofs(mesh, order: int, bc_types: Dict[int, str]) -> np.ndarray:
    """PEC edge DOFs, including the higher levels.

    Edge DOFs at level ``L`` (0-based) use the same numbering as
    ``face_dof_map``'s H(curl) block:

    * level 0: ``edge_id``
    * level 1: ``n_edges + edge_id``
    * level 2: ``2*n_edges + 2*n_faces + edge_id``
    * level 3: ``3*n_edges + 6*n_faces + edge_id``

    This matches ``face_dof_map``'s ordering where face DOFs are interleaved
    between edge levels at orders 3 and 4.
    """
    n_mesh_edges = mesh.n_edges
    n_faces = mesh.n_faces
    dofs = set()
    for edge_id in range(n_mesh_edges):
        marker = int(mesh.edge_labels[edge_id])
        if bc_types.get(marker) != "PerfectE":
            continue
        for level in range(order):
            # Edge DOF at level L: L*n_edges + L*(L-1)*n_faces + edge_id
            dofs.add(level * n_mesh_edges + level * (level - 1) * n_faces + edge_id)
    return np.array(sorted(dofs), dtype=np.int64)


def _dirichlet_node_dofs(mesh, order: int, bc_types: Dict[int, str]) -> np.ndarray:
    """Scalar (H1) DOFs on PEC edges, mirroring the 3D waveport elimination.

    The scalar numbering (``fes/core/dof.py``) is

    * node vertex DOFs: ``0 .. n_nodes-1``
    * edge-interior level ``L`` (0-based): ``n_nodes + L*n_edges + edge_id``
    * face-interior DOFs: above the edge levels, never touched here.

    As in the waveport path, face-interior scalar DOFs are *not* removed: the
    Dirichlet condition only constrains the trace on the PEC edges.
    """
    n_nodes = mesh.n_nodes
    n_edges = mesh.n_edges
    dofs = set()
    for edge_id in range(n_edges):
        marker = int(mesh.edge_labels[edge_id])
        if bc_types.get(marker) != "PerfectE":
            continue
        a, b = mesh.edges[edge_id]
        dofs.add(int(a))
        dofs.add(int(b))
        for level in range(order - 1):
            dofs.add(n_nodes + level * n_edges + edge_id)
    return np.array(sorted(dofs), dtype=np.int64)


def assemble_em_e_tl_eig(
    mesh,
    freq: float,
    order: int = 1,
    *,
    solve: bool = True,
    n_modes: Optional[int] = None,
    shift: Optional[float] = None,
    ortho: bool = True,
    eig_tol: float = 0.0,
    solver: str = "auto",
    prev_beta: Optional[np.ndarray] = None,
    use_t_block: Optional[bool] = None,
    t_block: Optional[bool] = None,
) -> TLEigSystem:
    """Assemble and (by default) solve the transmission-line eigenproblem.

    Parameters
    ----------
    mesh : fes.mesh.mesh2d.Mesh2D
        A 2D cross-section mesh.  Its edge connectivity is rebuilt with the
        corrected (sorted-endpoint) convention.
    freq : float
        Frequency in Hz; ``k0`` uses the oracle's ``c0``.
    order : int
        Polynomial order, 1-3 (``quad(order + 1)`` is the v0.1 ceiling).
    solve : bool
        Run the eigen-solve; when False the system is returned unsolved.
    n_modes : int | None
        Number of eigenmodes to compute (clamped to ``[1, numtot]``);
        ``None`` or 0 uses the default of 5.
    shift : float | None
        User-specified shift; defaults to ``-k0^2 * maxepsr * maxmur``, the
        same default as the waveport mode computation.  Must be non-zero.
    ortho : bool
        Accepted for compatibility; the Schur-complement formulation needs no
        Cangellaris projection, so this has no effect.
    eig_tol : float
        ARPACK tolerance used when the ARPACK path is selected.
    solver : str
        ``"auto"`` (ARPACK with dense fallback), ``"dense"``, ``"arpack"`` or
        ``"arnoldi"`` (deprecated alias for ``"arpack"``).
    prev_beta : np.ndarray | None
        Modal betas from the previous frequency.  When given, the selected
        modes are reordered to the nearest ``|mode_beta|`` so mode indices do
        not swap between adjacent sweep points.
    use_t_block, t_block : bool | None
        Accepted for compatibility; the Schur complement does not use the
        extended T-matrix structure, so these have no effect.

    Raises
    ------
    RuntimeError
        For 3D meshes or for models with ``WavePort`` boundaries, with the
        oracle's messages.
    """
    if getattr(mesh, "n_tetras", 0) > 0:
        raise RuntimeError(
            "em_e_tl_eig requires a 2D cross-section mesh "
            "(use +formula em_e_fd for 3D)"
        )
    for bc in mesh.boundaries:
        if bc.type == "WavePort":
            raise RuntimeError(
                "em_e_tl_eig does not support models with waveport boundaries; "
                "use +formula em_ez_fd for driven 2D waveport simulation"
            )

    order = int(order)
    freq = float(freq)
    mesh.build_2d_edge_connectivity()

    k0 = 2.0 * PI * freq / C0
    kk = k0 * k0

    n_edges = vector_dof_count(mesh, order)
    n_nodes = scalar_dof_count(mesh, order)
    St, Tt, Tt2, Sz, Tz, G, maxepsr, maxmur = _assemble_blocks(mesh, order)

    bc_types = _bc_type_map(mesh.boundaries)
    dirichlet = _dirichlet_dofs(mesh, order, bc_types)
    dirichlet_set = {int(dof) for dof in dirichlet}
    free_v = np.array(
        [i for i in range(n_edges) if i not in dirichlet_set], dtype=np.int64
    )

    # Dirichlet parity with the waveport path: the scalar (H1) trace DOFs on
    # PerfectE edges are eliminated alongside the H(curl) edge DOFs.
    dirichlet_nodes = _dirichlet_node_dofs(mesh, order, bc_types)
    dirichlet_node_set = {int(dof) for dof in dirichlet_nodes}
    free_s = np.array(
        [i for i in range(n_nodes) if i not in dirichlet_node_set],
        dtype=np.int64,
    )

    numt = int(free_v.size)
    numz = int(free_s.size)
    numtot = numt + numz

    if shift is None:
        shift = -kk * maxepsr * maxmur

    system = TLEigSystem(
        mesh=mesh,
        freq=freq,
        order=order,
        k0=k0,
        kk=kk,
        dof=n_edges + n_nodes,
        n_edge_dofs=n_edges,
        n_node_dofs=n_nodes,
        free_edges=free_v,
        dirichlet=dirichlet,
        shift=shift,
        free_nodes=free_s,
        dirichlet_nodes=dirichlet_nodes,
        St=St,
        Tt=Tt,
        Tt2=Tt2,
        Sz=Sz,
        Tz=Tz,
        G=G,
    )

    # The oracle prints a message and skips the solve in these two cases.
    if numt == 0 or numtot < 2 or not solve:
        return system

    # A zero shift causes division by zero in A / shift inside the
    # shift-invert operator; reject it with a clear error.
    if system.shift == 0:
        raise ValueError(
            "shift must be non-zero for the shift-invert eigen-solver; "
            f"got shift={system.shift}"
        )

    if numtot < 5:
        raise ValueError("Not enough dofs on the cross-section")

    if n_modes is not None and n_modes > 0:
        num_modes = min(max(int(n_modes), 1), numtot)
    else:
        num_modes = min(5, numtot)

    if solver == "arnoldi":
        solver = "arpack"  # deprecated alias
    elif solver not in ("auto", "arpack", "dense"):
        raise ValueError(
            f"unknown solver {solver!r}; use 'auto', 'dense' or 'arpack'"
        )

    # Same reduced pencil as the waveport: dense port-sized blocks and the
    # Zhu-Cangellaris Schur complement of the scalar block.
    tmpA = np.zeros((numtot, numtot), dtype=complex)
    tmpB = np.zeros((numtot, numtot), dtype=complex)
    tmpA[:numt, :numt] = (
        St[free_v][:, free_v] - kk * Tt[free_v][:, free_v]
    ).toarray()
    tmpB[:numt, :numt] = Tt2[free_v][:, free_v].toarray()
    if numz > 0:
        Gf = G[free_v][:, free_s].toarray()
        tmpB[:numt, numt:] = Gf
        tmpB[numt:, :numt] = Gf.T
        tmpB[numt:, numt:] = (
            Sz[free_s][:, free_s] - kk * Tz[free_s][:, free_s]
        ).toarray()

    A_tt, B_schur = _port_reduced_pencil(tmpA, tmpB, numt, numz)

    lam, vectors, label = solve_reduced_pencil(
        A_tt, B_schur, kk, shift, num_modes, eig_tol, solver
    )

    beta, mode_vec, mode_vecf = postprocess_modes(
        kk,
        k0,
        lam,
        vectors,
        lambda x: B_schur @ x,
        num_modes,
        prev_beta=prev_beta,
    )

    system.mode_beta = beta
    system.mode_vec = mode_vec
    system.mode_vecf = mode_vecf
    system.mode_vecdof = free_v
    system.solver = label
    return system
