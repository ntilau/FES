"""2D transmission-line cross-section eigenmode analysis.

Line-by-line port of ``cpp/src/assemble_em_e_tl_eig.cpp``: the scalar and
H(curl) bases are combined into a mixed block pencil

.. math::

    \\begin{bmatrix} S_t - k_0^2 T_t & 0 \\\\
                    0 & S_z - k_0^2 T_z \\end{bmatrix} x
    = \\lambda
    \\begin{bmatrix} T_{t2} & G \\\\ G^T & 0 \\end{bmatrix} x

and the first ``min(5, n/2 + 1)`` modes are reported as

.. math::

    \\beta = \\operatorname{Im}\\sqrt{\\lambda}, \\qquad
    \\alpha = \\operatorname{Re}\\sqrt{\\lambda}, \\qquad
    k_c = \\sqrt{|k_0^2 - \\beta^2 + \\alpha^2|}.

Reproducing the reference's eigen pipeline
---------------------------------------

The reference C++ hands ARPACK a *hand-written* shift-inverted operator.  With
``SS = A``, ``TT = B`` and ``kk = shift`` (the parameter is named ``kk`` in
``eigen.cpp`` even though the caller passes the shift), it factorises

.. math::

    A_s = SS/kk - TT

and defines ``op(x) = OO * (A_s^{-1} (TT x / kk))``, then calls ``zneupd`` with
``sigma = kk`` and ``which = "LM"``.  ``zneupd`` converts each Ritz value
``theta`` of ``op`` into ``d = sigma + 1/theta`` and the constructor takes
``sqrt(d)`` (zeroing components below ``1e-9``).

``OO`` is the Cangellaris orthogonalisation matrix: the identity on the
H(curl) block plus ``solve(-TT[zz,zz], TT[zv])`` on the lower-left block.  In
this formulation ``TT[zz,zz]`` is *never assigned* (it is the zero matrix), so
``arma::solve`` cannot factor it and falls back to
``auxlib::solve_approx_svd`` -- the SVD least-squares solution, which is zero
for a zero matrix.  ``OO`` is therefore ``diag(I_numt, 0)``, not the identity:
it projects the operator onto the H(curl) block.  That projection is what makes
the pencil produce the physical modes, and it is reproduced verbatim here.

Selection
---------

Three solver paths are available:

* ``"dense"`` — builds the full operator and calls ``scipy.linalg.eigvals``.
  BLAS-independent, used for systems small enough to be dense.
* ``"arpack"`` — calls ``scipy.sparse.linalg.eigs`` on a ``LinearOperator``
  wrapping an LU factorised shift-inverted operator; ARPACK's own Arnoldi.
* ``"arnoldi"`` — hand-written Arnoldi shift-invert, factorising the
  shift-inverted matrix once with SuperLU and iterating the Krylov
  subspace manually.

The default is ``"auto"``: dense for ``numtot <= DENSE_LIMIT``, arpack above.
Set ``solver`` explicitly to ``"dense"``, ``"arpack"`` or ``"arnoldi"`` to force a path.
The dense path cannot represent the ``Coaxial`` cross-section (156k DOFs, and
the reason the frozen binary is killed by the OOM killer); the sparse paths
exist for those systems.

User-facing controls
--------------------

``+n_modes N``
    Number of eigenmodes to compute.  Defaults to the reference's
    ``min(5, n/2 + 1)``; when set to a positive value it overrides that
    (clamped to ``[1, n/2 + 1]``).

``+shift F``
    User-specified shift (``gamma^2``) for the shift-inverted operator.
    Defaults to ``-k0^2 * maxepsr * maxmur`` when not given.

``+ortho`` / ``-ortho``
    Enable or disable the Cangellaris orthogonalisation projection
    (``OO = diag(I_numt, 0)``) that suppresses spurious modes
    (``k=0`` and ``div D = 0``).  Defaults to on.

``+tol T``
    ARPACK tolerance for the sparse solver path.  Defaults to ``0.0``
    (ARPACK's default).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np
import scipy.linalg
import scipy.sparse as sp
import scipy.sparse.linalg

from ...constants import C0, PI
from ..dof import face_dof_map, scalar_dof_count, vector_dof_count
from ..element_matrix import element_matrices_2d_mixed

__all__ = ["TLEigSystem", "assemble_em_e_tl_eig", "DENSE_LIMIT"]

#: Largest ``numtot`` for which the dense oracle replay is used.  Above this
#: the matrices are kept sparse and ARPACK resolves the pencil; the dense
#: operator at 156k DOFs (``Coaxial``) needs ~390 GB, which is exactly why the
#: frozen oracle is killed.
DENSE_LIMIT = 2000

#: ``gamtol`` in ``eigen.cpp``: real/imaginary parts below this are zeroed.
_GAMMA_TOL = 1e-9


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
    #: Complex ``sqrt(lambda)`` per mode, already ordered like the oracle.
    mode_beta: np.ndarray = field(default_factory=lambda: np.zeros(0, complex))
    #: ``"dense"`` (oracle replay), ``"arpack"`` (ARPACK shift-invert) or
    #: ``"arnoldi"`` (hand-written Arnoldi shift-invert).
    solver: str = "dense"
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
        return np.imag(self.mode_beta)

    @property
    def alpha(self) -> np.ndarray:
        return np.real(self.mode_beta)

    @property
    def kc(self) -> np.ndarray:
        return np.sqrt(
            np.abs(self.k0 * self.k0 - self.beta**2 + self.alpha**2)
        )

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
    maxepsr = 1.0
    maxmur = 1.0

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
    """PEC edge DOFs, including the ``lev * nMeshEdges + s`` higher levels.

    The level numbering is the oracle's, even for order 3 where it no longer
    matches ``face_dof_map``'s H(curl) numbering; the set is reproduced
    verbatim so the free-DOF subspace is identical.
    """
    n_mesh_edges = mesh.n_edges
    dofs = set()
    for edge_id in range(n_mesh_edges):
        marker = int(mesh.edge_labels[edge_id])
        if bc_types.get(marker) != "PerfectE":
            continue
        for level in range(order):
            dofs.add(level * n_mesh_edges + edge_id)
    return np.array(sorted(dofs), dtype=np.int64)


def _pencil(A_blocks, numt: int, numz: int):
    """Return ``(A, B)`` as dense arrays or sparse matrices, matching storage."""
    if isinstance(A_blocks[0], sp.spmatrix):
        A_vv, A_zz, B_vv, B_vz, B_zv = A_blocks
        n = numt + numz
        A = sp.bmat(
            [[A_vv, None], [None, A_zz]], format="csc", dtype=complex
        )
        B = sp.bmat(
            [[B_vv, B_vz], [B_zv, None]], format="csc", dtype=complex
        )
        return A, B

    A_vv, A_zz, B_vv, B_vz, B_zv = A_blocks
    n = numt + numz
    A = np.zeros((n, n), dtype=complex)
    B = np.zeros((n, n), dtype=complex)
    A[:numt, :numt] = A_vv
    A[numt:, numt:] = A_zz
    B[:numt, :numt] = B_vv
    B[:numt, numt:] = B_vz
    B[numt:, :numt] = B_zv
    return A, B


def _finalize_modes(theta: np.ndarray, shift: float, num_modes: int) -> np.ndarray:
    """Apply ``zneupd``'s transform, sqrt and the oracle's two sorts."""
    theta = np.asarray(theta, dtype=complex)
    if theta.size > num_modes:
        order = np.argsort(-np.abs(theta), kind="stable")[:num_modes]
        theta = theta[order]

    with np.errstate(divide="ignore", invalid="ignore"):
        d = shift + 1.0 / theta
    be = np.sqrt(d)

    real = np.where(np.abs(be.real) < _GAMMA_TOL, 0.0, np.abs(be.real))
    imag = np.where(np.abs(be.imag) < _GAMMA_TOL, 0.0, np.abs(be.imag))
    be = real + 1j * imag

    # ``zneupd``: ascending real part, then ascending imaginary part.
    be = be[np.lexsort((be.real, be.imag))]
    # ``eigen`` constructor: ``mode_beta(i) = Evals(nmodes - 1 - i)``.
    be = be[::-1]
    # ``assemble_em_e_tl_eig``: stable |beta| descending.
    return be[np.argsort(-np.abs(be), kind="stable")]


def _solve_dense(A, B, numt: int, numz: int, shift: float, num_modes: int, ortho: bool = True):
    """Replay ``eigen.cpp`` on a dense pencil.

    ``OO`` is ``diag(I_numt, 0)`` because the lower-left block
    ``solve(-TT[zz,zz], TT[zv])`` is the SVD least-squares solution of a zero
    system; see the module docstring.  When ``ortho`` is False, ``OO`` is the
    identity (no spurious-mode suppression projection).
    """
    n = numt + numz
    A_s = A / shift - B
    if ortho:
        OO = np.zeros((n, n), dtype=complex)
        OO[:numt, :numt] = np.eye(numt)
        operator = OO @ scipy.linalg.solve(A_s, B / shift)
    else:
        operator = scipy.linalg.solve(A_s, B / shift)
    theta = scipy.linalg.eigvals(operator)
    return _finalize_modes(theta, shift, num_modes)


def _solve_sparse(
    A, B, numt: int, numz: int, shift: float, num_modes: int, ortho: bool = True, eig_tol: float = 0.0
):
    """Resolve the same operator with ARPACK for systems too large to be dense."""
    n = numt + numz
    A_s = (A / shift - B).tocsc()
    lu = scipy.sparse.linalg.splu(A_s)

    def matvec(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=complex).reshape(n)
        y = np.asarray(B @ x, dtype=complex).reshape(n) / shift
        y = lu.solve(y)
        if ortho:
            y[numt:] = 0.0  # OO = diag(I_numt, 0)
        return y

    operator = scipy.sparse.linalg.LinearOperator(
        (n, n), matvec=matvec, dtype=complex
    )
    ncv = min(4 * num_modes, n - 1)
    theta = scipy.sparse.linalg.eigs(
        operator,
        k=num_modes,
        which="LM",
        ncv=ncv,
        tol=eig_tol,
        return_eigenvectors=False,
    )
    return _finalize_modes(theta, shift, num_modes)


def _arnoldi_shift_invert(
    A, B, numt: int, numz: int, shift: float, num_modes: int, ortho: bool = True, tol: float = 1e-10, max_iter: Optional[int] = None
):
    """Hand-written Arnoldi shift-invert solver.

    Factorises ``A_s = A/kk - TT`` once with SuperLU, then iterates the Krylov
    subspace

    .. math::

        \\mathcal{K}_m(op, v_1), \\qquad op(x) = OO \\, (A_s^{-1} (TT x / kk))

    until the Ritz values of the Hessenberg matrix converge.  Each Ritz value
    ``theta`` of ``op`` is converted by ``zneupd`` into ``d = sigma + 1/theta``,
    matching the dense/ARPACK paths exactly.

    Parameters
    ----------
    A, B : sparse or dense
        The pencil matrices (already restricted to free DOFs).
    numt, numz : int
        Numbers of free edge (H(curl)) and node (H(grad)) DOFs.
    shift : float
        The shift ``sigma`` for the shift-inverted operator.
    num_modes : int
        Number of eigenmodes to compute.
    ortho : bool
        Whether to apply the ``OO = diag(I_numt, 0)`` projection.
    tol : float
        Convergence tolerance for the Arnoldi iteration (lucky breakdown).
    max_iter : int | None
        Maximum Arnoldi iterations; defaults to ``min(4 * num_modes, n - 1)``.
    """
    n = numt + numz
    A_s = (A / shift - B).tocsc()
    lu = scipy.sparse.linalg.splu(A_s)

    def matvec(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=complex).reshape(n)
        y = np.asarray(B @ x, dtype=complex).reshape(n) / shift
        y = lu.solve(y)
        if ortho:
            y[numt:] = 0.0  # OO = diag(I_numt, 0)
        return y

    # Random starting vector, deterministic seed for reproducibility.
    rng = np.random.default_rng(0)
    v = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    v = v / np.linalg.norm(v)

    max_iter = max_iter or min(4 * num_modes, n - 1)
    max_iter = min(max_iter, n - 1)
    if max_iter < 1:
        max_iter = 1

    V = np.zeros((n, max_iter + 1), dtype=complex)
    H = np.zeros((max_iter + 1, max_iter), dtype=complex)
    V[:, 0] = v

    j = 0
    while j < max_iter:
        w = matvec(V[:, j])
        for i in range(j + 1):
            H[i, j] = np.vdot(V[:, i], w)
            w = w - H[i, j] * V[:, i]
        H[j + 1, j] = np.linalg.norm(w)
        if H[j + 1, j] < tol:
            break  # lucky breakdown — Krylov subspace is invariant
        V[:, j + 1] = w / H[j + 1, j]
        j += 1

    # Ritz values from the Hessenberg matrix H_m (upper m x m block).
    Hm = H[:j, :j]
    theta = np.linalg.eigvals(Hm)

    # Select the num_modes Ritz values with largest |theta| (closest to shift),
    # matching ARPACK's which="LM".
    if theta.size > num_modes:
        order = np.argsort(-np.abs(theta), kind="stable")[:num_modes]
        theta = theta[order]

    return _finalize_modes(theta, shift, num_modes)


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
    dirichlet_set = set(int(dof) for dof in dirichlet)
    free_v = np.array(
        [i for i in range(n_edges) if i not in dirichlet_set], dtype=np.int64
    )
    numt = int(free_v.size)
    numz = n_nodes
    numtot = numt + numz

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
        shift=shift if shift is not None else -kk * maxepsr * maxmur,
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

    if n_modes is not None and n_modes > 0:
        num_modes = min(max(int(n_modes), 1), numtot // 2 + 1)
    else:
        num_modes = min(5, numtot // 2 + 1)
    if num_modes < 1:
        num_modes = 1

    Stf = St[free_v][:, free_v]
    Ttf = Tt[free_v][:, free_v]
    Tt2f = Tt2[free_v][:, free_v]
    Gf = G[free_v]
    Azz = Sz - kk * Tz
    A_vv = Stf - kk * Ttf
    B_vv = Tt2f

    if solver == "auto":
        use_dense = numtot <= DENSE_LIMIT
    elif solver == "dense":
        use_dense = True
    elif solver == "arpack":
        use_dense = False
    elif solver == "arnoldi":
        use_dense = False
    else:
        raise ValueError(
            f"unknown solver {solver!r}; use 'auto', 'dense', 'arpack' or 'arnoldi'"
        )

    if solver == "arnoldi":
        A_blocks = (A_vv.tocsc(), Azz.tocsc(), B_vv.tocsc(), Gf.tocsc(), Gf.T.tocsc())
        A, B = _pencil(A_blocks, numt, numz)
        system.mode_beta = _arnoldi_shift_invert(
            A, B, numt, numz, system.shift, num_modes, ortho=ortho, tol=eig_tol
        )
        system.solver = "arnoldi"
    elif use_dense:
        A_blocks = (
            A_vv.toarray(),
            Azz.toarray(),
            B_vv.toarray(),
            Gf.toarray(),
            Gf.toarray().T,  # ``Gf.st()``: plain transpose
        )
        A, B = _pencil(A_blocks, numt, numz)
        system.mode_beta = _solve_dense(A, B, numt, numz, system.shift, num_modes, ortho=ortho)
        system.solver = "dense"
    else:
        A_blocks = (A_vv.tocsc(), Azz.tocsc(), B_vv.tocsc(), Gf.tocsc(), Gf.T.tocsc())
        A, B = _pencil(A_blocks, numt, numz)
        system.mode_beta = _solve_sparse(A, B, numt, numz, system.shift, num_modes, ortho=ortho, eig_tol=eig_tol)
        system.solver = "arpack"
    return system
