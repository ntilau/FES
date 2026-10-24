"""2D TMz scalar Helmholtz assembly with transfinite-element ports.

Line-by-line port of ``cpp/src/assemble_em_ez_fd.cpp``.  The pipeline is:

1. ``k0 = 2*pi*f/c0`` with the reference constants.
2. Build the full 2D edge connectivity (see :mod:`fes.mesh.mesh2d`) and the
   scalar DOF numbering (:mod:`fes.core.dof`).
3. Assemble ``A = S - k0^2 T`` per triangle, with ``epsr``/``mur`` taken from
   the material whose ``#Solids`` line index equals the triangle's region
   attribute.
4. Collect the PEC Dirichlet DOFs: both endpoints of every ``PerfectE`` edge,
   plus the ``p_ord-1`` edge-interior levels.
5. For every ``WavePort`` boundary, build the 1D edge mass/stiffness matrices
   with Lagrange interpolation at ``p_ord+1`` Gauss-Legendre points, solve the
   generalized eigenproblem ``St x = lambda Tt x`` on the free port DOFs, sort
   by ``lambda``, and keep the first ``num_modes`` modes.
6. Couple the modes to the interior with the TFE blocks (see the C++ source for
   the derivation).
7. Solve the complex non-symmetric system with SuperLU and form
   ``S = (X[:n_modes] - I) * sqrt(power)``.

Two intentional differences from the frozen reference:

* ``legacy_edge_connectivity`` (default ``False``) selects the v0.1
  ``build_2d_edge_connectivity`` bug that collapses edge DOFs and makes every
  port S-matrix ``-I``.  The default is the corrected connectivity described
  in :meth:`fes.mesh.mesh2d.Mesh2D.build_2d_edge_connectivity`.
* The linear solve uses ``scipy.sparse.linalg.splu`` instead of MUMPS.  Both
  are direct LU factorizations of the same matrix, so the solutions agree to
  round-off.

Notes on reproducing the reference's floating-point details:

* ``z0`` is the reference's ``376.730313461``, not ``120*pi``.
* The port eigenproblem is solved with the complex LAPACK routine (``zggev``)
  exactly as ``arma::eig_pair`` does, so the "largest component real"
  normalisation matches.  The explicit sign fix and the ``sqrt(|x^T T x|)``
  normalisation are then applied verbatim.
* ``Mg*Aw*Mg.t()`` uses a plain transpose (not a Hermitian one) in the TFE
  coupling; that is reproduced here as well.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import scipy.linalg
import scipy.sparse as sp
import scipy.sparse.linalg

from ...constants import C0, PI, Z0
from ..dof import scalar_dof_count, scalar_dof_map
from ..element_matrix import triangle_jacobian
from ..quadrature import quadrature_2d
from ..shape import shape_2d

__all__ = [
    "WavePort",
    "TMzSystem",
    "assemble_em_ez_fd",
    "extract_solution_full",
    "GL_X",
    "GL_W",
]

#: Gauss-Legendre abscissae on [-1, 1] for 1..6 points (C++ static table).
#: Ragged on purpose: ``GL_X[ng - 1]`` has exactly ``ng`` entries.
GL_X = (
    (0.0,),
    (0.5773502691896257, -0.5773502691896257),
    (0.7745966692414834, 0.0, -0.7745966692414834),
    (0.8611363115940526, 0.3399810435848563, -0.3399810435848563, -0.8611363115940526),
    (0.9061798459386640, 0.5384693101056831, 0.0, -0.5384693101056831, -0.9061798459386640),
    (
        0.9324695142031520,
        0.6612093864662645,
        0.2386191860831969,
        -0.2386191860831969,
        -0.6612093864662645,
        -0.9324695142031520,
    ),
)

#: Matching Gauss-Legendre weights.
GL_W = (
    (2.0,),
    (1.0, 1.0),
    (0.5555555555555556, 0.8888888888888888, 0.5555555555555556),
    (0.3478548451374538, 0.6521451548625461, 0.6521451548625461, 0.3478548451374538),
    (
        0.2369268850561891,
        0.4786286704993665,
        0.5688888888888889,
        0.4786286704993665,
        0.2369268850561891,
    ),
    (
        0.1713244923791704,
        0.3607615730481386,
        0.4679139345726910,
        0.4679139345726910,
        0.3607615730481386,
        0.1713244923791704,
    ),
)


@dataclass
class WavePort:
    """One ``WavePort`` boundary after the modal eigenproblem."""

    name: str
    label: int
    #: ``pn``: the port DOF ids in the reference's insertion order.
    dofs: np.ndarray
    #: Indices into :attr:`dofs` of the DOFs that are not Dirichlet.
    free: np.ndarray
    #: Eigenvalue-derived propagation constants, ``sqrt(k0^2 - lambda)``.
    mode_beta: np.ndarray
    #: ``(len(dofs), n_modes)`` mode vectors, zero on Dirichlet DOFs.
    mode_vec: np.ndarray
    #: Number of modes the ``.poly`` requested (before truncation).
    requested_modes: int = 0

    @property
    def n_modes(self) -> int:
        return int(self.mode_beta.size)


@dataclass
class TMzSystem:
    """Everything the 2D TMz assembler produced."""

    mesh: object
    freq: float
    order: int
    k0: float
    dof: int
    A: sp.csr_matrix
    B: np.ndarray
    dirichlet: np.ndarray
    ports: List[WavePort] = field(default_factory=list)
    n_modes: int = 0
    S: Optional[np.ndarray] = None
    X: Optional[np.ndarray] = None
    legacy_edge_connectivity: bool = False
    #: ``nnV`` from the TFE reduction: global DOF ids of the reduced rows
    #: after the port rows, in order.
    nonwave_ids: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.int64)
    )
    #: The ``+pow`` value, needed by :func:`extract_solution_full`.
    power: float = 1.0


def extract_solution_full(system: TMzSystem) -> np.ndarray:
    """Expand the reduced TFE solution like ``solver::extract_solution_full``.

    The reduced vector holds ``n_modes`` port rows followed by the interior
    rows listed in :attr:`TMzSystem.nonwave_ids`.  The reference's ``Sol_mat`` is
    ``(dof, n_cols)``; each wave port then adds its modal contribution with
    the ``sqrt(j k0 z0 power / beta)`` weight from ``solver.cpp``.

    When the model has no wave ports the reference solves ``A x = B`` with one
    right-hand side and leaves ``Sol`` at zero unless ``B`` is non-zero (the
    ``+einc`` path); the Python port does not implement ``+einc`` yet, so a
    zero column is returned.
    """
    dof = int(system.dof)
    if system.X is None:
        columns = 1 if system.B.ndim > 1 else 0
        return np.zeros((dof, columns), dtype=complex)

    solution = np.asarray(system.X)
    n_modes = int(system.n_modes)
    sol = np.zeros((dof, n_modes), dtype=complex)
    sp = solution[:n_modes, :]
    for row, node in enumerate(system.nonwave_ids):
        node = int(node)
        if 0 <= node < dof:
            sol[node, :] = solution[n_modes + row, :]

    jk0z0 = 1j * system.k0 * Z0 * float(system.power)
    mode_offset = 0
    for port in system.ports:
        for mode in range(port.n_modes):
            weight = np.sqrt(jk0z0 / port.mode_beta[mode])
            for j in range(port.dofs.size):
                sol[int(port.dofs[j]), :] += (
                    weight * port.mode_vec[j, mode] * sp[mode_offset + mode, :]
                )
        mode_offset += port.n_modes
    return sol


# ── element matrices ────────────────────────────────────────────────────────


#: Kept as a private alias for tests that exercise the element geometry.
_triangle_jacobian = triangle_jacobian


def _assemble_volume(mesh, order: int, k0: float, total_dof: int) -> sp.csr_matrix:
    """Assemble ``S - k0^2 T`` for the scalar Helmholtz operator."""
    points, weights = quadrature_2d(order + 1)
    nq = points.shape[0]
    k0sq = k0 * k0

    rows: List[np.ndarray] = []
    cols: List[np.ndarray] = []
    vals: List[np.ndarray] = []

    for face_id in range(mesh.n_faces):
        nodes = mesh.faces[face_id]
        geo = mesh.nodes[nodes, :2]
        det, inv_j = triangle_jacobian(geo)
        if 0.5 * det < 1e-30:
            continue

        solid = mesh.material_for_face(face_id)
        if solid is None:
            epsr, mur = 1.0, 1.0
        else:
            epsr, mur = float(solid.epsr), float(solid.mur)
        inv_mur = 1.0 / mur

        shp = shape_2d(order, points, inv_j)
        ns, dns = shp.Ns, shp.dNs
        nld = ns.shape[1]

        se = np.zeros((nld, nld))
        te = np.zeros((nld, nld))
        for iq in range(nq):
            w = weights[iq] * det
            grad = dns[iq]
            se += (grad @ grad.T) * (w * inv_mur)
            te += np.outer(ns[iq], ns[iq]) * (w * epsr)

        l2g = scalar_dof_map(mesh, order, face_id)
        block = se - k0sq * te
        rows.append(np.repeat(l2g, nld))
        cols.append(np.tile(l2g, nld))
        vals.append(block.ravel())

    if not vals:
        return sp.csr_matrix((total_dof, total_dof), dtype=complex)

    matrix = sp.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(total_dof, total_dof),
        dtype=complex,
    ).tocsr()
    matrix.sum_duplicates()
    return matrix


# ── Dirichlet DOFs ──────────────────────────────────────────────────────────


def _bc_type_map(boundaries: Sequence) -> Dict[int, str]:
    """``std::map<label, type>``: a later boundary with the same label wins."""
    types: Dict[int, str] = {}
    for bc in boundaries:
        types[int(bc.label)] = bc.type
    return types


def _dirichlet_dofs(mesh, order: int, bc_types: Dict[int, str]) -> np.ndarray:
    n_nodes = mesh.n_nodes
    n_edges = mesh.n_edges
    levels = order - 1 if order > 1 else 0

    dofs = set()
    for edge_id in range(n_edges):
        marker = int(mesh.edge_labels[edge_id])
        if bc_types.get(marker) != "PerfectE":
            continue
        dofs.add(int(mesh.edges[edge_id, 0]))
        dofs.add(int(mesh.edges[edge_id, 1]))
        for level in range(levels):
            dofs.add(n_nodes + level * n_edges + edge_id)
    return np.array(sorted(dofs), dtype=np.int64)


# ── waveports ───────────────────────────────────────────────────────────────


def _port_dofs(mesh, bc, order: int) -> List[int]:
    """The ``pn`` list: port DOFs in first-touch order, no duplicates."""
    n_nodes, n_edges = mesh.n_nodes, mesh.n_edges
    levels = order - 1 if order > 1 else 0
    seen = set()
    ordered: List[int] = []
    for edge_id in range(n_edges):
        if int(mesh.edge_labels[edge_id]) != int(bc.label):
            continue
        n0 = int(mesh.edges[edge_id, 0])
        n1 = int(mesh.edges[edge_id, 1])
        for node in (n0, n1):
            if node not in seen:
                seen.add(node)
                ordered.append(node)
        for level in range(levels):
            dof = n_nodes + level * n_edges + edge_id
            if dof not in seen:
                seen.add(dof)
                ordered.append(dof)
    return ordered


def _lagrange_1d(ne: int, t: float):
    """Values and derivatives of the ``ne`` Lagrange polynomials at ``t``."""
    t_node = -1.0 + 2.0 * np.arange(ne) / (ne - 1)
    values = np.ones(ne)
    derivs = np.zeros(ne)
    for i in range(ne):
        for j in range(ne):
            if j != i:
                values[i] *= (t - t_node[j]) / (t_node[i] - t_node[j])
        for k in range(ne):
            if k == i:
                continue
            term = 1.0 / (t_node[i] - t_node[k])
            for j in range(ne):
                if j != i and j != k:
                    term *= (t - t_node[j]) / (t_node[i] - t_node[j])
            derivs[i] += term
    return values, derivs


def _edge_mass_stiffness(mesh, bc, order: int, dofs: List[int]):
    """Build ``St`` and ``Tt`` on the port DOF list exactly as the reference does."""
    n_nodes, n_edges = mesh.n_nodes, mesh.n_edges
    levels = order - 1 if order > 1 else 0
    positions = {dof: i for i, dof in enumerate(dofs)}
    size = len(dofs)
    st = np.zeros((size, size))
    tt = np.zeros((size, size))

    for edge_id in range(n_edges):
        if int(mesh.edge_labels[edge_id]) != int(bc.label):
            continue
        n0 = int(mesh.edges[edge_id, 0])
        n1 = int(mesh.edges[edge_id, 1])
        dx = mesh.nodes[n1, 0] - mesh.nodes[n0, 0]
        dy = mesh.nodes[n1, 1] - mesh.nodes[n0, 1]
        length = float(np.hypot(dx, dy))

        edge_dofs = [n0, n1]
        for level in range(levels):
            edge_dofs.append(n_nodes + level * n_edges + edge_id)
        ne = len(edge_dofs)

        local = []
        for dof in edge_dofs:
            index = positions.get(dof)
            if index is None:
                local = None
                break
            local.append(index)
        if local is None:
            # The C++ silently skips an edge whose DOFs are not all on the port.
            continue

        gx = GL_X[ne - 1]
        gw = GL_W[ne - 1]
        for q in range(ne):
            t = float(gx[q])
            w = float(gw[q]) * length / 2.0
            values, derivs = _lagrange_1d(ne, t)
            block = np.ix_(local, local)
            st[block] += np.outer(derivs, derivs) * (4.0 / (length * length) * w)
            tt[block] += np.outer(values, values) * w

    return st, tt


def _port_modes(
    mesh,
    bc,
    order: int,
    k0: float,
    dirichlet: set,
) -> WavePort:
    """Solve the port eigenproblem, reproducing the reference's conventions."""
    dofs = _port_dofs(mesh, bc, order)
    port = WavePort(
        name=bc.name,
        label=int(bc.label),
        dofs=np.array(dofs, dtype=np.int64),
        free=np.zeros(0, dtype=np.int64),
        mode_beta=np.zeros(0, dtype=complex),
        mode_vec=np.zeros((len(dofs), 0), dtype=complex),
        requested_modes=int(bc.num_modes),
    )
    if not dofs:
        return port

    st, tt = _edge_mass_stiffness(mesh, bc, order, dofs)

    free = np.array([i for i, dof in enumerate(dofs) if dof not in dirichlet], dtype=np.int64)
    port.free = free
    nf = free.size
    if nf < 2:
        return port

    # Keep the pencil real so LAPACK uses dggev, exactly as ``arma::eig_pair``
    # does for ``arma::mat`` arguments.  The complex driver (zggev) leaves an
    # arbitrary phase on the eigenvectors, which changes the TFE coupling.
    sf = st[np.ix_(free, free)]
    tf = tt[np.ix_(free, free)]
    ev, evec = scipy.linalg.eig(sf, tf)
    evec = np.asarray(evec, dtype=complex)

    order_idx = np.argsort(np.real(ev), kind="stable")
    ev = ev[order_idx]
    evec = evec[:, order_idx]

    nm = min(int(bc.num_modes), ev.size)
    k0sq = k0 * k0
    beta = np.zeros(nm, dtype=complex)
    modes = np.zeros((nf, nm), dtype=complex)
    for m in range(nm):
        beta[m] = np.sqrt(complex(k0sq - float(np.real(ev[m])), 0.0))
        vec = evec[:, m].copy()
        pivot = int(np.argmax(np.abs(np.real(vec))))
        if np.real(vec[pivot]) < 0:
            vec = -vec
        modes[:, m] = vec

    for m in range(nm):
        vec = modes[:, m]
        norm = np.sqrt(abs(vec.T @ tf @ vec))
        if norm > 1e-15:
            modes[:, m] = vec / norm

    full = np.zeros((len(dofs), nm), dtype=complex)
    full[free, :] = modes
    port.mode_beta = beta
    port.mode_vec = full
    return port


# ── driver ──────────────────────────────────────────────────────────────────


def assemble_em_ez_fd(
    mesh,
    freq: float,
    order: int = 1,
    *,
    power: float = 1.0,
    legacy_edge_connectivity: bool = False,
    solve: bool = True,
) -> TMzSystem:
    """Assemble and (by default) solve the 2D TMz system.

    Parameters
    ----------
    mesh : fes.mesh.mesh2d.Mesh2D
        Mesh from :func:`fes.mesh.mesh2d.mesh_from_poly`.  Its edge
        connectivity is (re)built with the requested convention.
    freq : float
        Frequency in Hz.  ``k0`` uses the reference's ``c0``.
    order : int
        Polynomial order, 1-4.
    power : float
        Port power scaling; the reference multiplies ``S`` by ``sqrt(power)``.
    legacy_edge_connectivity : bool
        Reproduce the v0.1 unsorted-pair connectivity bug (default: corrected).
    solve : bool
        When ``False`` only the matrices and ports are built.

    Returns
    -------
    TMzSystem
    """
    order = int(order)
    n_nodes, n_edges = mesh.n_nodes, mesh.n_edges

    mesh.build_2d_edge_connectivity(legacy_unsorted_pairs=legacy_edge_connectivity)
    n_edges = mesh.n_edges  # the connectivity call may have grown the edge list

    k0 = 2.0 * PI * float(freq) / C0
    total_dof = scalar_dof_count(mesh, order)
    matrix = _assemble_volume(mesh, order, k0, total_dof)

    bc_types = _bc_type_map(mesh.boundaries)
    dirichlet = _dirichlet_dofs(mesh, order, bc_types)
    dirichlet_set = set(int(d) for d in dirichlet)

    ports: List[WavePort] = []
    wave_port_boundaries = [bc for bc in mesh.boundaries if bc.type == "WavePort"]

    if not wave_port_boundaries:
        # No ports: the reference still eliminates the Dirichlet rows/columns.
        result = TMzSystem(
            mesh=mesh,
            freq=float(freq),
            order=order,
            k0=k0,
            dof=total_dof,
            A=matrix,
            B=np.zeros((total_dof, 1), dtype=complex),
            dirichlet=dirichlet,
            legacy_edge_connectivity=legacy_edge_connectivity,
            power=float(power),
        )
        if dirichlet.size:
            result.A = _apply_dirichlet(matrix, dirichlet)
            result.B = np.zeros((total_dof, 1), dtype=complex)
        return result

    for bc in wave_port_boundaries:
        ports.append(_port_modes(mesh, bc, order, k0, dirichlet_set))

    n_modes = sum(port.n_modes for port in ports)
    if n_modes == 0:
        result = TMzSystem(
            mesh=mesh,
            freq=float(freq),
            order=order,
            k0=k0,
            dof=total_dof,
            A=matrix,
            B=np.zeros((total_dof, 1), dtype=complex),
            dirichlet=dirichlet,
            ports=ports,
            n_modes=0,
            legacy_edge_connectivity=legacy_edge_connectivity,
            power=float(power),
        )
        if dirichlet.size:
            result.A = _apply_dirichlet(matrix, dirichlet)
        return result

    # ── power normalisation over the port edges ─────────────────────────────
    ymin, ymax = 1e30, -1e30
    for edge_id in range(mesh.n_edges):
        marker = int(mesh.edge_labels[edge_id])
        if bc_types.get(marker) != "WavePort":
            continue
        for node in (int(mesh.edges[edge_id, 0]), int(mesh.edges[edge_id, 1])):
            y = float(mesh.nodes[node, 1])
            ymin = min(ymin, y)
            ymax = max(ymax, y)
    height = ymax - ymin
    wp_pow = 4.0 / height if height > 1e-12 else 1.0
    pfac = wp_pow * float(power)

    # ── TFE block system ────────────────────────────────────────────────────
    port_ids: List[int] = []
    for port in ports:
        port_ids.extend(int(d) for d in port.dofs)

    port_set = set()
    pwp: List[int] = []
    for node_id in port_ids:
        if node_id >= total_dof:
            continue
        if node_id not in dirichlet_set and node_id not in port_set:
            port_set.add(node_id)
            pwp.append(node_id)

    n_none = [i for i in range(total_dof) if i not in dirichlet_set and i not in port_set]
    n_w = len(pwp)
    w_local = {dof: i for i, dof in enumerate(pwp)}

    mg = np.zeros((n_modes, n_w), dtype=complex)
    mc = np.zeros((n_modes, n_modes), dtype=complex)
    bx = np.zeros((n_modes, n_modes), dtype=complex)
    mi = 0
    for port in ports:
        if port.n_modes == 0:
            continue
        for m in range(port.n_modes):
            beta_mag = float(np.abs(port.mode_beta[m]))
            with np.errstate(divide="ignore", invalid="ignore"):
                scale = np.sqrt(k0 * Z0 * pfac / beta_mag)
            mc[mi, mi] = 1j * k0 * Z0 * pfac
            bx[mi, mi] = 2j * k0 * Z0 * pfac
            for j in range(port.dofs.size):
                wi = w_local.get(int(port.dofs[j]))
                if wi is not None and wi < n_w:
                    mg[mi, wi] = scale * port.mode_vec[j, m]
            mi += 1

    aw = _dense_block(matrix, pwp, pwp)
    awn = _dense_block(matrix, pwp, n_none)
    ann = _sparse_block(matrix, n_none, n_none)

    n_modes_used = n_modes
    tl = mg @ aw @ mg.T + mc if n_w > 0 else mc
    pi_mat = mg @ awn if n_w > 0 else np.zeros((n_modes_used, len(n_none)), dtype=complex)

    size = n_modes_used + len(n_none)
    blocks = [
        (_nz_coords(tl), 0, 0),
        (_nz_coords(pi_mat), 0, n_modes_used),
        (_nz_coords(np.conj(pi_mat).T), n_modes_used, 0),
        (_nz_coords(ann), n_modes_used, n_modes_used),
    ]
    rows = np.concatenate([coords[0] + row0 for coords, row0, _ in blocks])
    cols = np.concatenate([coords[1] + col0 for coords, _, col0 in blocks])
    vals = np.concatenate([coords[2] for coords, _, _ in blocks])
    keep = vals != 0
    system = sp.coo_matrix(
        (vals[keep], (rows[keep], cols[keep])), shape=(size, size), dtype=complex
    ).tocsr()
    system.sum_duplicates()

    rhs = np.zeros((size, n_modes_used), dtype=complex)
    for i in range(n_modes_used):
        rhs[i, i] = bx[i, i]

    result = TMzSystem(
        mesh=mesh,
        freq=float(freq),
        order=order,
        k0=k0,
        dof=total_dof,
        A=system,
        B=rhs,
        dirichlet=dirichlet,
        ports=ports,
        n_modes=n_modes_used,
        legacy_edge_connectivity=legacy_edge_connectivity,
        nonwave_ids=np.asarray(n_none, dtype=np.int64),
        power=float(power),
    )

    if solve:
        lu = scipy.sparse.linalg.splu(system.tocsc())
        solution = lu.solve(rhs)
        result.X = solution
        sp_mat = solution[:n_modes_used, :]
        result.S = (sp_mat - np.eye(n_modes_used)) * np.sqrt(float(power))

    return result


def _submatrix(matrix, rows: Sequence[int], cols: Sequence[int]):
    """Sparse ``A[rows, :][:, cols]`` preserving the reference's entry values."""
    if len(rows) == 0 or len(cols) == 0:
        return sp.csr_matrix((len(rows), len(cols)), dtype=complex)
    return matrix[np.asarray(rows, dtype=np.int64)][:, np.asarray(cols, dtype=np.int64)]


def _dense_block(matrix, rows: Sequence[int], cols: Sequence[int]) -> np.ndarray:
    """Dense ``A[rows, cols]``; entries absent from the sparse matrix are zero."""
    return np.asarray(_submatrix(matrix, rows, cols).todense(), dtype=complex)


def _sparse_block(matrix, rows: Sequence[int], cols: Sequence[int]) -> sp.csr_matrix:
    return _submatrix(matrix, rows, cols).tocsr()


def _nz_coords(block):
    """``(row, col, value)`` for the non-zero entries of a dense or sparse block.

    Mirrors the C++ ``if(std::abs(x) > 0)`` guard when the block is copied into
    the TFE system.
    """
    if sp.issparse(block):
        coo = block.tocoo()
        return coo.row.astype(np.int64), coo.col.astype(np.int64), coo.data.astype(complex)
    block = np.asarray(block)
    if block.size == 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, np.zeros(0, dtype=complex)
    row, col = np.nonzero(np.abs(block) > 0)
    return row.astype(np.int64), col.astype(np.int64), block[row, col].astype(complex)


def _apply_dirichlet(matrix: sp.csr_matrix, dirichlet: np.ndarray) -> sp.csr_matrix:
    """Zero the Dirichlet rows/columns and put ones on the diagonal."""
    size = matrix.shape[0]
    keep = np.ones(size, dtype=float)
    keep[np.asarray(dirichlet, dtype=np.int64)] = 0.0
    mask = sp.diags(keep)
    out = (mask @ matrix @ mask).tolil()
    for dof in np.asarray(dirichlet, dtype=np.int64):
        out[dof, dof] = 1.0
    return out.tocsr()
