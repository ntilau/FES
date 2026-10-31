"""3D H(curl) curl-curl assembly with transfinite-element waveports.

Line-by-line port of ``cpp/src/assemble_em_e_fd.cpp`` together with the two
functions it delegates to:

* ``assembler::compute_waveport_modes`` (``cpp/src/factory.cpp``), which builds
  the per-port tangential operators ``St/Tt/Tt2/Sz/Tz/G``, solves the port
  eigenproblem and normalises the modes;
* ``eigen`` (``cpp/src/eigen.cpp``), the ARPACK shift-invert solve including
  the Zhu-Cangellaris orthogonaliser and the deterministic start vector.

The pipeline follows the reference implementation step for step:

1. ``k0 = 2*pi*f/c0``, ``kk = k0**2``, ``dofnum = dof(prj).dofnumv``.
2. Bulk assembly: for each tetrahedron, ``S + k0*Z - kk*T`` accumulated only on
   the upper triangle ``i <= j`` of the vector DOF map, exactly as the reference's
   ``std::map`` accumulator does.  Entry values are therefore identical to the
   reference's down to the summation order.
3. ``PerfectE`` faces append their scalar and vector DOFs to the Dirichlet
   lists; the vector DOFs also clear ``doftoLeave``.
4. ``Radiation`` faces add ``i*k0*Tt*sqrt(epsr/mur)`` on the upper triangle.
5. ``WavePort``/``LumpedPort`` boundaries run the modal eigenproblem.
6. The transfinite-element reduction replaces the porter block by
   ``num_modes`` modal DOFs and leaves the complement
   (``Nonwave_portIds``) untouched.
7. The reduced system is solved and ``S = (Sp - I)*sqrt(power)``.
8. With no waveports and ``+einc``, the incident-wave branch instead builds
   ``B`` from the ``Radiation`` faces (``i*k0`` times the face integral of the
   tangential incident field), zeroes the Dirichlet rows and columns, and
   solves the full ``dofnum`` system.  With neither ports nor an incident
   wave the reference throws ``Null Right Hand Side``; so does this port.

Deliberate differences, both outside the numerical definition of the answer:

* The port eigenproblem is solved densely instead of with ARPACK.  The reference
  asks ``znaupd`` for the ``num_modes`` eigenvalues of the shift-invert
  operator with the largest magnitude, i.e. the generalized eigenvalues of
  ``(tmpA, tmpB)`` *nearest the shift*.  On a port of this size
  ``ncv == numtot``, so the Krylov space is truncated and which Ritz value it
  drops depends on the ARPACK build: the frozen binary (ARPACK-NG 3.9.1 +
  OpenBLAS) drops one, SciPy's bundled copy drops another.  LAPACK's
  generalized eigensolver resolves the whole pencil, so the selection is
  deterministic and independent of the BLAS.  See :func:`_port_eigenpairs`.
* The linear solve uses SuperLU on the full symmetric expansion of the
  one-triangle matrix.  The reference C++ hands the same entries to MUMPS with
  ``sym=1``; both are direct LU factorisations of the same operator.

The reference's representation keeps the top-left block in its *lower* triangle,
the top-right coupling block, and the bottom-right block in its *upper*
triangle.  Each symmetric pair appears exactly once, which is what MUMPS
expects for ``sym=1`` and what :func:`_full_symmetric` reconstructs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import scipy.linalg
from scipy.linalg import solve, sqrtm, inv as _inv
import scipy.sparse as sp
import scipy.sparse.linalg

from ...constants import C0, PI, Z0
from ..dof import face_dof_map, vector_dof_count, vector_dof_map
from .errors import NullRightHandSideError
from ..element_matrix import (
    Material,
    einc_face_vector,
    element_matrices_3d,
    element_matrices_face,
)

__all__ = ["FDPort", "FDSystem", "assemble_em_e_fd"]

#: ``DBL_MIN``, the reference's starting value for ``maxepsr``/``maxmur``.
_DBL_MIN = float.fromhex("0x0.0000000000001p-1022")

#: ``gamtol`` from ``eigen.cpp``: components below this become exactly zero.
_GAMMA_TOL = 1e-9


@dataclass
class FDPort:
    """One modal port after ``compute_waveport_modes``.

    ``mode_vec`` is ``(n_port_dofs, num_modes)`` and ``mode_vecdof`` holds the
    global DOF numbers of the free (non-Dirichlet) port DOFs, both in the reference's order.  ``edges`` is ``tmpEdges`` itself, so ``mode_vecdof`` is
    ``edges[free]``.
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
class FDSystem:
    """Everything the 3D ``em_e_fd`` assembler produced."""

    mesh: object
    freq: float
    order: int
    k0: float
    kk: float
    dofnum: int
    dofreal: int
    A: sp.csr_matrix
    B: np.ndarray
    dirichlet_s: np.ndarray
    dirichlet_v: np.ndarray
    ports: List[FDPort] = field(default_factory=list)
    wave_ports_num: int = 0
    wave_ports_dofnum: int = 0
    nonwave_ids: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    Meig: Optional[np.ndarray] = None
    MAcoeff: Optional[np.ndarray] = None
    B_exc: Optional[np.ndarray] = None
    A_wp: Optional[np.ndarray] = None
    A_wp_nn: Optional[np.ndarray] = None
    Attred: Optional[np.ndarray] = None
    Atired: Optional[np.ndarray] = None
    top_left: Optional[np.ndarray] = None
    power: float = 1.0
    Sol: Optional[np.ndarray] = None
    reduced_solution: Optional[np.ndarray] = None
    Sp: Optional[np.ndarray] = None
    S: Optional[np.ndarray] = None

    @property
    def port_shift(self) -> np.ndarray:
        return np.array([port.shift for port in self.ports])


# ── helpers ─────────────────────────────────────────────────────────────────


def _ensure_topology(mesh) -> None:
    if getattr(mesh, "tet_edges", None) is None:
        mesh.build_topology()
    if getattr(mesh, "face_adj_tet", None) is None:
        mesh.build_topology()


def _material_for_tet(mesh, tet_id: int) -> Material:
    return Material.from_solid(mesh.material_for_tet(int(tet_id)))


def _upper_triplets(v: np.ndarray, block: np.ndarray):
    """Triplets the reference's ``cdof.v(i) <= cdof.v(j)`` test selects.

    The test is on the *global* DOF numbers, not on the local indices.  That
    matters because the local ``dof`` lists are not sorted (a tetrahedron
    lists its edge DOFs in local edge order), so ``np.triu_indices`` over the
    local index would put some pairs in the lower triangle.  The matrix is
    symmetric so the values are the same either way, but the stored triangle
    decides which half of each coupling the TFE reduction can read back.
    """
    rows, cols = np.nonzero(v[:, None] <= v[None, :])
    return v[rows], v[cols], block[rows, cols]


def _dense_to_sparse(parts, size: int) -> sp.csr_matrix:
    if not parts:
        return sp.csr_matrix((size, size), dtype=complex)
    rows = np.concatenate([p[0] for p in parts])
    cols = np.concatenate([p[1] for p in parts])
    vals = np.concatenate([p[2] for p in parts])
    matrix = sp.coo_matrix((vals, (rows, cols)), shape=(size, size), dtype=complex)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    return matrix.tocsr()


def _full_symmetric(matrix: sp.spmatrix) -> sp.csr_matrix:
    """Expand a one-triangle matrix to its full symmetric form.

    The reference stores each symmetric pair once (possibly in either triangle)
    and lets MUMPS ``sym=1`` mirror it.  Reconstructing the full matrix gives
    the same operator to a direct solver.
    """
    matrix = matrix.tocsr()
    diagonal = sp.diags(matrix.diagonal())
    return (matrix + matrix.T - diagonal).tocsr()


# ── bulk assembly ───────────────────────────────────────────────────────────


def _assemble_bulk(mesh, order: int, k0: float, kk: float, dofnum: int) -> sp.csr_matrix:
    """``S + k0*Z - kk*T`` over every tetrahedron, upper triangle only."""
    parts = []
    for tet_id in range(mesh.n_tetras):
        material = _material_for_tet(mesh, tet_id)
        lmat = element_matrices_3d(order, mesh.tet_geo(tet_id), material)
        block = lmat.S + k0 * lmat.Z - kk * lmat.T
        v = vector_dof_map(mesh, order, tet_id)
        parts.append(_upper_triplets(v, block))
    return _dense_to_sparse(parts, dofnum)


# ── radiation ───────────────────────────────────────────────────────────────


def _add_radiation(mesh, matrix: sp.csr_matrix, order: int, k0: float, freq: float,
                   dofnum: int) -> sp.csr_matrix:
    """Add the first-order ABC on every ``Radiation`` face."""
    parts = []
    for bc in mesh.boundaries:
        if bc.type != "Radiation":
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
            parts.append(_upper_triplets(cdof.v, lmat.Tt * factor))
    if not parts:
        return matrix
    return (matrix + _dense_to_sparse(parts, dofnum)).tocsr()


# ── Dirichlet DOFs ──────────────────────────────────────────────────────────


def _dirichlet(mesh, order: int, doftoLeave: np.ndarray):
    """Collect the ``PerfectE`` scalar/vector DOFs and clear ``doftoLeave``."""
    scalar_parts: List[np.ndarray] = []
    vector_parts: List[np.ndarray] = []
    for bc in mesh.boundaries:
        if bc.type == "PerfectE":
            for fid in mesh.faces_for_boundary(bc):
                cdof = face_dof_map(mesh, order, int(fid))
                scalar_parts.append(cdof.s)
                vector_parts.append(cdof.v)
                doftoLeave[cdof.v] = False
        elif bc.type in ("PerfectH", "WavePort", "LumpedPort", "Radiation"):
            continue
        else:
            raise ValueError(f"Wrong boundary type: {bc.type}")
    if scalar_parts:
        dirdofs = np.unique(np.concatenate(scalar_parts))
        dirdofv = np.unique(np.concatenate(vector_parts))
    else:
        dirdofs = np.zeros(0, dtype=np.int64)
        dirdofv = np.zeros(0, dtype=np.int64)
    return dirdofs, dirdofv


# ── port eigenproblem ───────────────────────────────────────────────────────


def _port_eigenpairs(tmpA: np.ndarray, tmpB: np.ndarray, numt: int,
                     shift: float, nmodes: int) -> tuple:
    """Resolve all eigenpairs of the generalized pencil ``tmpA x = l tmpB x``.

    Used when ``numz == 0`` (no scalar face DoFs).  Returns every eigenpair
    after removing static scalar modes (zero vector part); the caller
    (:func:`_port_modes`) selects the physical port modes by cutoff wavenumber
    and orders them TE10-first.  ``values`` are the pencil eigenvalues
    ``lambda`` directly (dense generalized solve), with ``kc^2 = kk + lambda``.
    """
    values, vectors = scipy.linalg.eig(tmpA, tmpB)

    # Filter modes with no vector part (static scalar mode at lambda = 0).
    content = np.linalg.norm(vectors[:numt, :], axis=0)
    total = np.linalg.norm(vectors, axis=0)
    has_vector_part = content > 1e-8 * np.maximum(total, np.finfo(float).tiny)
    values = values[has_vector_part]
    vectors = vectors[:, has_vector_part]

    return values, vectors


def _port_eigenpairs_oo(tmpA: np.ndarray, tmpB: np.ndarray, numt: int,
                        numz: int, kk: float, shift: float,
                        nmodes: int) -> tuple:
    """Resolve port modes via Schur-complement reduction (numz > 0).

    Eliminates the scalar face DOFs with the Zhu-Cangellaris Schur complement:

        B_schur = Tt2 - G @ inv(Sz - kk*Tz) @ G.T

    then solves the reduced edge pencil

        (St - kk*Tt) v = lambda * B_schur v

    directly with a dense generalized eigensolve.  This is mathematically
    equivalent to the shift-invert + Zhu-Cangellaris operator used by the
    reference, but returns true pencil eigenvectors (residual ~1e-9) instead
    of eigenvectors of the non-symmetric OPc operator (residual ~1e5).

    Returns ``(lam, vectors, B_schur)`` where ``lam`` are the pencil
    eigenvalues (``kc^2 = kk + lam``), ``vectors`` are edge-only eigenvectors
    of shape ``(numt, n)``, and ``B_schur`` is the Schur-complement matrix used
    for B-orthonormalisation by the caller.  The caller selects physical port
    modes by cutoff wavenumber ``kc^2 > 0`` and orders them TE10-first.
    """
    # Extract blocks from tmpB
    Tt2 = tmpB[:numt, :numt]
    G = tmpB[:numt, numt:]
    D = tmpB[numt:, numt:]

    # Schur complement: eliminate scalar DOFs
    B_schur = Tt2 - G @ solve(D, G.conj().T)

    # A_tt is the top-left block of tmpA (= St - kk*Tt)
    A_tt = tmpA[:numt, :numt]

    # Dense generalized eigensolve of the reduced edge pencil
    lam, vectors = scipy.linalg.eig(A_tt, B_schur)

    return lam, vectors, B_schur


def _port_modes(
    mesh,
    bc,
    order: int,
    k0: float,
    kk: float,
    freq: float,
    dirichlet_s: np.ndarray,
    dirichlet_v: np.ndarray,
    dofnum: int,
) -> FDPort:
    """``assembler::compute_waveport_modes`` for a single boundary."""
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

    # tmpdofv/tmpdofs: global DOF -> index inside the port's local list.  The
    # reference starts from zero-filled uvecs, so DOFs outside the list map to 0.
    tmpdofv = np.zeros(dofnum, dtype=np.int64)
    tmpdofv[tmp_edges] = np.arange(tmp_edges.size)
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

        vi = tmpdofv[cdof.v]
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

    if numz > 0:
        lam, vectors, B_schur = _port_eigenpairs_oo(
            tmpA, tmpB, numt, numz, kk, shift, nmodes)
        # pencil eigenvalue lambda; kc^2 = kk + lambda
        lam = np.asarray(lam, dtype=complex)
    else:
        lam, vectors = _port_eigenpairs(tmpA, tmpB, numt, shift, nmodes)
        lam = np.asarray(lam, dtype=complex)
        B_schur = tmpB[:numt, :numt]

    # Cutoff wavenumber squared: kc^2 = kk + lambda = k0^2 + (kc^2 - k0^2).
    # Keep only physical waveguide modes (kc^2 > 0); the indefinite pencil
    # (St - k0^2 Tt, Tt2) also yields spurious negative-kc^2 eigenvalues.
    kc2 = kk + lam
    physical = np.real(kc2) > 0.0
    lam = lam[physical]
    vectors = np.asarray(vectors, dtype=complex)[:, physical]
    kc2 = kc2[physical]

    # Order TE10-first: ascending cutoff wavenumber, then take nmodes.
    take = min(int(nmodes), kc2.size)
    order_idx = np.argsort(np.real(kc2), kind="stable")[:take]
    lam = lam[order_idx]
    vectors = vectors[:, order_idx]
    kc2 = kc2[order_idx]

    # Propagation constant: beta^2 = k0^2 - kc^2 = -lambda.
    beta = np.sqrt(kk - kc2)
    real = np.abs(beta.real)
    imag = np.abs(beta.imag)
    real[real < _GAMMA_TOL] = 0.0
    imag[imag < _GAMMA_TOL] = 0.0
    beta = real + 1j * imag

    # B-orthonormalisation matching C++ (AssLin.cpp line 342-343):
    #   ModeVec = V[:numt,:] @ inv(sqrt(V.st() * B * V))
    # where sqrt is the matrix square root and .st() is the plain transpose
    # (not conjugate).  Using V^T B_schur V (complex symmetric) matches the
    # oracle.  This makes the selected eigenvectors orthonormal in the B inner
    # product, as the reference requires for the TFE reduction.
    gram = vectors.T @ B_schur @ vectors
    scale = _inv(sqrtm(gram))
    mode_vec = vectors @ scale

    coeff = np.zeros((beta.size, beta.size), dtype=complex)
    np.fill_diagonal(coeff, np.sqrt(complex(0.0, k0 * Z0) / beta))
    mode_vecf = B_schur @ (vectors @ coeff @ scale)

    for m in range(mode_vec.shape[1]):
        pivot = int(np.argmax(np.abs(mode_vec[:, m].real)))
        if mode_vec[pivot, m].real < 0:
            mode_vec[:, m] *= -1.0
            mode_vecf[:, m] *= -1.0

    mode_vecdof = tmp_edges[tmpDirdofv]
    impedance = 0.0
    if bc.type == "LumpedPort":
        impedance = float(getattr(bc, "impedance", 50.0))

    return FDPort(
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


def assemble_em_e_fd(
    mesh,
    freq: float,
    order: int = 2,
    *,
    power: float = 1.0,
    solve: bool = True,
    e_inc: Optional[Sequence[float]] = None,
    k_inc: Optional[Sequence[float]] = None,
) -> FDSystem:
    """Assemble and (by default) solve the 3D H(curl) curl-curl system.

    ``e_inc``/``k_inc`` are the ``+einc`` incident plane-wave parameters in
    the reference's units (the electric-field vector and the propagation
    direction).  As in ``assemble_em_e_fd.cpp`` the incident-wave branch runs
    only when the model has no waveports; with ports the TFE reduction wins.
    """
    order = int(order)
    freq = float(freq)
    _ensure_topology(mesh)

    k0 = 2.0 * PI * freq / C0
    kk = k0 * k0
    dofnum = vector_dof_count(mesh, order)

    matrix = _assemble_bulk(mesh, order, k0, kk, dofnum)
    matrix = _add_radiation(mesh, matrix, order, k0, freq, dofnum)

    doftoLeave = np.ones(dofnum, dtype=bool)
    dirdofs, dirdofv = _dirichlet(mesh, order, doftoLeave)

    ports: List[FDPort] = []
    for bc in mesh.boundaries:
        if bc.type not in ("WavePort", "LumpedPort"):
            continue
        port = _port_modes(
            mesh, bc, order, k0, kk, freq, dirdofs, dirdofv, dofnum
        )
        doftoLeave[port.edges] = False
        ports.append(port)

    wave_ports_num = sum(port.n_modes for port in ports)
    wave_ports_dofnum = sum(port.mode_vec.shape[0] for port in ports)
    nonwave_ids = np.flatnonzero(doftoLeave)
    dofreal = int(nonwave_ids.size + wave_ports_num)

    result = FDSystem(
        mesh=mesh,
        freq=freq,
        order=order,
        k0=k0,
        kk=kk,
        dofnum=dofnum,
        dofreal=dofreal,
        A=matrix,
        B=np.zeros((dofreal, wave_ports_num), dtype=complex),
        dirichlet_s=dirdofs,
        dirichlet_v=dirdofv,
        ports=ports,
        wave_ports_num=wave_ports_num,
        wave_ports_dofnum=wave_ports_dofnum,
        nonwave_ids=nonwave_ids,
        power=float(power),
    )

    if wave_ports_num > 0:
        return _reduce_and_solve(result, matrix, wave_ports_num,
                                 wave_ports_dofnum, dirdofv, nonwave_ids,
                                 solve=solve)
    if e_inc is not None and k_inc is not None:
        return _incident_solve(result, matrix, dirdofv, e_inc, k_inc, solve=solve)
    # With neither ports nor an incident wave ``B`` stays empty and the
    # reference throws before the solver runs.
    raise NullRightHandSideError("Null Right Hand Side")


def _incident_solve(
    result: FDSystem,
    matrix: sp.csr_matrix,
    dirdofv: np.ndarray,
    e_inc: Sequence[float],
    k_inc: Sequence[float],
    *,
    solve: bool,
) -> FDSystem:
    """``assemble_em_e_fd.cpp``'s ``else if(opt->einc)`` branch.

    ``kEinc`` is scaled by ``k0`` and ``polEinc`` by ``1/sqrt(2)``; every
    ``Radiation`` face contributes ``i*k0*f`` to ``B``.  The Dirichlet rows
    *and* columns are zeroed and their diagonal set to one, then the full
    ``(dofnum, 1)`` system is solved (MUMPS ``dofreal == dofnum`` here).
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
            rhs[cdof.v, 0] += vector * complex(0.0, k0)

    result.B = rhs
    result.A = matrix
    # ``sys.set_dofreal(sys.get_dofnum())``: the incident-wave system keeps
    # every DOF, including the Dirichlet rows that the branch zeros out.
    result.dofreal = result.dofnum

    if not np.any(rhs != 0):
        raise NullRightHandSideError("Null Right Hand Side")
    if not solve:
        return result

    full = _full_symmetric(matrix).tolil()
    full[dirdofv, :] = 0.0
    full[:, dirdofv] = 0.0
    full[dirdofv, dirdofv] = 1.0
    full = full.tocsr()

    lu = scipy.sparse.linalg.splu(full.tocsc())
    result.Sol = lu.solve(rhs)
    return result


def _reduce_and_solve(
    result: FDSystem,
    matrix: sp.csr_matrix,
    wave_ports_num: int,
    wave_ports_dofnum: int,
    dirdofv: np.ndarray,
    nonwave_ids: np.ndarray,
    *,
    solve: bool,
) -> FDSystem:
    """Build the TFE-reduced system, its right-hand side, and optionally solve."""
    k0, power = result.k0, result.power
    wpn, wpdofnum = wave_ports_num, wave_ports_dofnum
    nnum = int(nonwave_ids.size)
    dofnum = result.dofnum

    meig = np.zeros((wpn, wpdofnum), dtype=complex)
    macoeff = np.zeros((wpn, wpn), dtype=complex)
    b_exc = np.zeros((wpn, wpn), dtype=complex)
    idx = jdx = 0
    for port in result.ports:
        if port.is_lumped:
            impedance = port.impedance
        else:
            impedance = Z0
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

    wp_ids = np.zeros(0, dtype=np.int64)
    for port in result.ports:
        wp_ids = np.concatenate([wp_ids, np.asarray(port.mode_vecdof, dtype=np.int64)])

    # Step 1: zero the diagonal.  The C++ reference (AssLin.cpp lines 443-446)
    # saves the diagonal to Adiag, then zeroes it in sys->A.  Attred1 (used for
    # top_left) reads from the zeroed matrix; Attred2 (which would add back the
    # diagonal and transpose) is computed but discarded.  So a_wp has ZERO
    # diagonal, matching C++.
    zeroed = matrix.tolil()
    zeroed.setdiag(0.0)
    zeroed = zeroed.tocsr()
    zeroed.eliminate_zeros()

    # Step 2: interior block, upper triangle (from the original matrix, diagonal
    # intact — matching C++ which copies the interior before zeroing).
    # ``nonwave_ids`` is ascending, so ``wpn + wi <= wpn + wj`` iff ``wi <= wj``:
    # the upper triangle of the index sub-block is exactly the reference's
    # global-DOF-number test.
    parts = []
    interior = matrix[np.ix_(nonwave_ids, nonwave_ids)]
    interior = sp.triu(interior, format="coo")
    interior.eliminate_zeros()
    parts.append((interior.row + wpn, interior.col + wpn, interior.data))

    # Step 3: extract the port blocks from the zeroed matrix.  C++ reads
    # A_zeroed[WPids, WPids] (upper triangle only, zero diagonal) for Attred1
    # which is used for top_left.  The diagonal is NOT restored (Attred2 which
    # would restore it is discarded).  For a_wp_nn, C++ reads A_zeroed[WPids,
    # nnWPids] (upper triangle only); using the full symmetric matrix here would
    # add transposed entries C++ does not have.
    wnum = int(wp_ids.size)

    # C++ (AssLin.cpp lines 448-452) computes:
    #   Attred1 = Meig * (A_zeroed[WPids,WPids] + A_zeroed[WPids,WPids]^T
    #                   + Adiag[WPids,WPids]) * Meig^T
    #   = Meig * _full_symmetric(A[WPids,WPids]) * Meig^T
    # So a_wp must be the full symmetric expansion of the original port block
    # (restoring both triangles and the diagonal), not just the zeroed upper
    # triangle.
    a_wp_orig = matrix[np.ix_(wp_ids, wp_ids)].toarray()
    a_wp = a_wp_orig + a_wp_orig.T - np.diag(np.diag(a_wp_orig))

    # C++ (AssLin.cpp lines 464-466) computes:
    #   Atired1 = Meig * A_zeroed[WPids, nnWPids]
    # Reading one triangle only.  Since wp_ids and nonwave_ids are interleaved
    # (not all port DOFs precede all interior DOFs), reading one triangle would
    # miss entries stored at the transposed position.  A is symmetric so
    # A[wp,nw] = A[nw,wp]; reading both triangles recovers the full block.
    a_wp_nn = matrix[np.ix_(wp_ids, nonwave_ids)].toarray()
    a_wp_nn += matrix[np.ix_(nonwave_ids, wp_ids)].T.toarray()

    # TFE modal reduction.  The C++ reference (AssLin.cpp, lines 448-454) computes:
    #   Attred1 = Meig * A_zeroed[WPids, WPids]          (one triangle only!)
    #   Anew[0:wpn, 0:wpn] = lower_triangle(Attred1 * Meig^T) + MAcoeff
    #   Atired1 = Meig * A_zeroed[WPids, nnWPids]
    #   Anew[0:wpn, wpn:] = Atired1   (direct copy)
    # Note: only Attred1 is used for the top-left (Attred2, which adds the
    # transpose and Adiag, is computed but discarded).  Adiag[WPids, nnWPids]
    # is zero since the index sets are disjoint.
    attred = meig @ a_wp
    # C++ uses plain transpose (gmm::transposed): Meig * A_zeroed[WPids,WPids] *
    # Meig^T with plain Meig on both sides.  The result is complex symmetric
    # (top_left[i,j] == top_left[j,i]), which is what _full_symmetric expects
    # when it mirrors the stored triangle.  Using meig.conj() on the left would
    # make top_left Hermitian (top_left[i,j] == conj(top_left[j,i])) instead,
    # so _full_symmetric would write conjugated values into the upper triangle
    # and corrupt the operator — breaking energy conservation.
    top_left = (meig @ a_wp) @ meig.T + macoeff
    atired = meig @ a_wp_nn

    # Top-left (lower triangle, j <= i) and top-right coupling blocks.
    top_left_coo = sp.coo_matrix(np.tril(top_left))
    parts.append((top_left_coo.row, top_left_coo.col, top_left_coo.data))
    atired_coo = sp.coo_matrix(atired)
    parts.append((atired_coo.row, atired_coo.col + wpn, atired_coo.data))

    reduced = _dense_to_sparse(parts, result.dofreal)
    reduced.eliminate_zeros()

    result.A = reduced
    result.Meig = meig
    result.MAcoeff = macoeff
    result.B_exc = b_exc
    result.A_wp = a_wp
    result.A_wp_nn = a_wp_nn
    result.Attred = attred
    result.Atired = atired
    result.top_left = top_left
    result.B = np.zeros((result.dofreal, wpn), dtype=complex)
    result.B[np.diag_indices(wpn)] = np.diag(b_exc)

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


def _extract_solution_full(system: FDSystem, reduced: np.ndarray) -> np.ndarray:
    """``solver::extract_solution_full``: expand the reduced solution.

    ``reduced`` is the ``(wave_ports_num + Nonwave, wave_ports_num)`` solution
    of the TFE system.  The reference stores it back into a ``(dofnum, n_rhs)``
    array: the interior rows go to their global DOF numbers, the modal rows to
    ``Sp``, and every free port DOF is rebuilt from the modal amplitudes with
    the ``sqrt(j*k0*Z/beta)`` weight (``Z`` is ``z0`` or the lumped-port
    impedance).  Dirichlet DOFs stay zero.
    """
    wpn = int(system.wave_ports_num)
    solution = np.zeros((system.dofnum, reduced.shape[1]), dtype=complex)
    solution[system.nonwave_ids, :] = reduced[wpn:, :]

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
