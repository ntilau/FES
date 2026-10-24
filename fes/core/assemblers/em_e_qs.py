"""2D electrostatics (``em_e_qs``) with nodal P1 triangles.

Line-by-line port of the ``msh->nTetras == 0`` branch of
``cpp/src/assemble_em_e_qs.cpp``:

1. Number one DOF per mesh node (``sys.set_dofnum(nNodes)``).  The polynomial
   order is *not* used: the reference C++ always assembles linear nodal elements in
   2D, so ``+p`` changes nothing here.
2. Accumulate ``area * epsr * grad(N_i) . grad(N_j)`` per triangle into a
   triplet map.  ``epsr`` comes from the material whose ``#Solids`` line index
   equals the triangle's region attribute (default 1.0).
3. Collect the endpoints of every ``PerfectE`` PLC segment as Dirichlet DOFs
   and set their prescribed voltage from ``+volt <name> <V>`` (default 0).
4. Eliminate the Dirichlet rows and columns sequentially, in ascending DOF
   order, exactly as the reference C++ does -- the order is observable because each
   eliminated column updates the whole right-hand side.
5. Solve the non-symmetric complex system; ``splu`` stands in for MUMPS.

The reference C++ throws ``Null Right Hand Side`` when no boundary carries a voltage;
the port raises :class:`~.errors.NullRightHandSideError` (a ``RuntimeError``)
with the same message.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg

from .errors import NullRightHandSideError

__all__ = ["QSSystem", "assemble_em_e_qs"]


@dataclass
class QSSystem:
    """Everything the 2D electrostatic assembler produced."""

    mesh: object
    freq: float
    order: int
    dof: int
    A: sp.csr_matrix
    B: np.ndarray
    dirichlet: np.ndarray
    #: The solved nodal potential, ``(dof,)``; ``None`` when ``solve=False``.
    phi: Optional[np.ndarray] = None
    #: ``+volt`` values keyed by boundary name.
    voltages: Dict[str, float] = field(default_factory=dict)
    #: Electrostatics have no ports; kept so :class:`SolverResult` can report.
    n_modes: int = 0
    S: Optional[np.ndarray] = None


def _bc_maps(boundaries) -> tuple:
    """``std::map`` semantics: a later boundary with the same label wins."""
    types: Dict[int, str] = {}
    names: Dict[int, str] = {}
    for bc in boundaries:
        types[int(bc.label)] = bc.type
        names[int(bc.label)] = bc.name
    return types, names


def _add_entry(rows: Dict[int, Dict[int, complex]], i: int, j: int, value) -> None:
    row = rows.setdefault(i, {})
    row[j] = row.get(j, 0.0) + value


def _stiffness_rows(mesh) -> Dict[int, Dict[int, complex]]:
    """Accumulate ``area * epsr * grad(N_i) . grad(N_j)`` per triangle.

    The map is keyed ``[row][col]``, matching the oracle's
    ``std::map<pair<uword, uword>, complex>``.  Stored zeros are kept: the
    Dirichlet sweep below relies on the same explicit-zero structure as
    Armadillo's ``SpMat``.
    """
    rows: Dict[int, Dict[int, complex]] = {}
    for face_id in range(mesh.n_faces):
        nodes = mesh.faces[face_id]
        x0, y0 = float(mesh.nodes[nodes[0], 0]), float(mesh.nodes[nodes[0], 1])
        x1, y1 = float(mesh.nodes[nodes[1], 0]), float(mesh.nodes[nodes[1], 1])
        x2, y2 = float(mesh.nodes[nodes[2], 0]), float(mesh.nodes[nodes[2], 1])

        j11, j12 = x1 - x0, y1 - y0
        j21, j22 = x2 - x0, y2 - y0
        det_j = j11 * j22 - j12 * j21
        area = 0.5 * abs(det_j)
        if area < 1e-30:
            continue

        solid = mesh.material_for_face(face_id)
        epsr = 1.0 if solid is None else float(solid.epsr)

        inv_j00 = j22 / det_j
        inv_j01 = -j12 / det_j
        inv_j10 = -j21 / det_j
        inv_j11 = j11 / det_j
        gx = (
            inv_j00 * (-1) + inv_j01 * (-1),
            inv_j00 * 1 + inv_j01 * 0,
            inv_j00 * 0 + inv_j01 * 1,
        )
        gy = (
            inv_j10 * (-1) + inv_j11 * (-1),
            inv_j10 * 1 + inv_j11 * 0,
            inv_j10 * 0 + inv_j11 * 1,
        )

        local = [int(n) for n in nodes]
        scale = area * epsr
        for i in range(3):
            for j in range(3):
                _add_entry(
                    rows,
                    local[i],
                    local[j],
                    scale * (gx[i] * gx[j] + gy[i] * gy[j]),
                )
    return rows


def assemble_em_e_qs(
    mesh,
    order: int = 1,
    voltages: Optional[Mapping[str, float]] = None,
    *,
    freq: float = 0.0,
    solve: bool = True,
) -> QSSystem:
    """Assemble (and by default solve) the 2D electrostatic system.

    Parameters
    ----------
    mesh : fes.mesh.mesh2d.Mesh2D
        Mesh from :func:`fes.mesh.mesh2d.mesh_from_poly`.
    order : int
        Accepted for signature symmetry with the other assemblers; the 2D
        branch of the oracle ignores it.
    voltages : mapping, optional
        ``+volt <boundary name> <V>`` values, keyed by boundary name.
    freq : float
        Frequency in Hz, stored on the result.  Electrostatics ignore it.
    solve : bool
        When ``False`` only the eliminated matrix and right-hand side are
        built (the oracle's ``Null Right Hand Side`` check still applies).
    """
    voltages = dict(voltages or {})
    dof = int(mesh.n_nodes)

    # ── nodal P1 stiffness ──────────────────────────────────────────────────
    # ``std::map<std::pair<row, col>, value>``: rows[d][c] is the
    # accumulation, and ``cols`` mirrors it for the column sweep below.
    rows = _stiffness_rows(mesh)

    cols: Dict[int, Dict[int, complex]] = {}
    for i, line in rows.items():
        for j, value in line.items():
            cols.setdefault(j, {})[i] = value

    # ── PerfectE voltages ───────────────────────────────────────────────────
    bc_types, bc_names = _bc_maps(mesh.boundaries)
    dirichlet: List[int] = []
    rhs: Dict[int, complex] = {}
    # ``Savefield`` runs before assembly, so the oracle still sees only the
    # PLC segments here; ``mesh_from_poly`` has already appended interior
    # edges, and ``n_boundary_edges`` marks where the original ones end.
    for edge_id in range(int(mesh.n_boundary_edges)):
        marker = int(mesh.edge_labels[edge_id])
        if bc_types.get(marker) != "PerfectE":
            continue
        value = float(voltages.get(bc_names[marker], 0.0))
        n0 = int(mesh.edges[edge_id, 0])
        n1 = int(mesh.edges[edge_id, 1])
        dirichlet.append(n0)
        dirichlet.append(n1)
        if value != 0.0:
            rhs[n0] = complex(value, 0.0)
            rhs[n1] = complex(value, 0.0)

    dirichlet = np.unique(np.asarray(dirichlet, dtype=np.int64))

    # ── Dirichlet elimination (sequential, ascending DOF order) ─────────────
    brhs = np.zeros(dof, dtype=complex)
    for d in dirichlet:
        brhs[int(d)] = rhs.get(int(d), 0.0)

    for d in dirichlet:
        d = int(d)
        # Zero the row (stored zeros stay in the structure, as in Armadillo).
        for col in rows.get(d, {}):
            rows[d][col] = 0.0
            if col in cols and d in cols[col]:
                cols[col][d] = 0.0
        # Eliminate the column and fold it into the right-hand side.
        for row, value in list(cols.get(d, {}).items()):
            if row != d:
                brhs[row] -= value * brhs[d]
            rows.setdefault(row, {})[d] = 0.0
            cols[d][row] = 0.0
        rows.setdefault(d, {})[d] = 1.0
        cols.setdefault(d, {})[d] = 1.0

    # ── sparse assembly ─────────────────────────────────────────────────────
    a_rows: List[int] = []
    a_cols: List[int] = []
    a_vals: List[complex] = []
    for i, line in rows.items():
        for j, value in line.items():
            a_rows.append(i)
            a_cols.append(j)
            a_vals.append(value)

    matrix = sp.csr_matrix(
        (np.asarray(a_vals, dtype=complex), (a_rows, a_cols)),
        shape=(dof, dof),
    )
    matrix.sum_duplicates()

    b = np.zeros((dof, 1), dtype=complex)
    for row, value in enumerate(brhs):
        if value != 0.0:
            b[row, 0] = value
    if not np.any(b != 0.0):
        raise NullRightHandSideError("Null Right Hand Side")

    phi = None
    if solve:
        lu = scipy.sparse.linalg.splu(matrix.tocsc())
        phi = lu.solve(b[:, 0])

    return QSSystem(
        mesh=mesh,
        freq=float(freq),
        order=int(order),
        dof=dof,
        A=matrix,
        B=b,
        dirichlet=dirichlet,
        phi=phi,
        voltages=voltages,
    )
