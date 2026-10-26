"""Field export ported from ``cpp/src/field.cpp``.

The original ``field`` class is constructed by ``post_processor::save_data``
when ``+field`` is given.  It writes ``<name>_<freq>.vtk``:

1. ``Dumpmesh()`` truncates the file and writes a ``Solution data`` mesh with
   ``float`` points (six significant digits, unlike ``mesh::Savefield``'s
   ``double``/16 digits),
2. one E-field block for the summed solution (``cnt = 0``),
3. one E block per solution column (``cnt = jj + 1``) while the solution has
   fewer than ten columns.  The 3D H(curl) branch adds an H block per column.

Both the 2D TMz and the 3D H(curl) branches reproduce the reference's
``sqrt(2)`` RMS-to-amplitude factor and its node averaging
(``NodeVal /= SumTimes``) exactly.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

from ..constants import C0, MU0, PI
from ..core.dof import scalar_dof_map, vector_dof_map
from ..core.element_matrix import tetrahedron_jacobian
from ..core.shape3d import shape_3d

__all__ = ["freq_tag", "write_field_vtk"]

_SQRT2 = np.sqrt(2.0)

#: Reference-tetrahedron vertices, in the oracle's ``locNode`` order.
_LOC_NODE = np.array(
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
)


def freq_tag(freq: float) -> str:
    """``std::stringstream << double`` with its default precision of 6."""
    return f"{float(freq):.6g}"


def _f32(value: float) -> str:
    """``std::ostream << (float) value``: six significant digits of a float."""
    return f"{float(np.float32(value)):.6g}"


def _norm(row: np.ndarray) -> float:
    """``arma::norm(arma::abs(row), 2)`` for a complex 3-vector."""
    return float(np.sqrt(np.sum(np.abs(row) ** 2)))


def _append_field(
    out: List[str],
    freq_text: str,
    prefix: str,
    node_val: np.ndarray,
    count: int,
    first: bool,
    unit: str,
) -> None:
    add = out.append
    if first:
        add(f"POINT_DATA {node_val.shape[0]}\n")
    add(f"SCALARS {freq_text}_{prefix}_norm_{count}_[{unit}] float 1\n")
    add("LOOKUP_TABLE jet\n")
    for row in node_val:
        add(f"{_f32(_norm(row))}\n")
    for label, values in (
        ("abs", np.abs(node_val)),
        ("real", np.real(node_val)),
        ("imag", np.imag(node_val)),
    ):
        add(f"VECTORS {freq_text}_{prefix}_{label}_{count}_[{unit}] float\n")
        for x, y, z in values:
            add(f"{_f32(x)} {_f32(y)} {_f32(z)}\n")


def _append_efield(
    out: List[str],
    freq_text: str,
    node_val: np.ndarray,
    count: int,
    first: bool,
    unit: str = "V/m",
) -> None:
    _append_field(out, freq_text, "E", node_val, count, first, unit)


def _append_hfield(
    out: List[str],
    freq_text: str,
    node_val: np.ndarray,
    count: int,
    first: bool,
) -> None:
    _append_field(out, freq_text, "H", node_val, count, first, "A/m")


def _dumpmesh(out: List[str], mesh, assembly: str) -> None:
    add = out.append
    add("# vtk DataFile Version 2.0\n")
    add("Solution data\n")
    add("ASCII\n")
    add("DATASET UNSTRUCTURED_GRID\n")
    add(f"POINTS {mesh.n_nodes} float \n")
    for x, y, z in mesh.nodes:
        add(f"{_f32(x)} {_f32(y)} {_f32(z)}\n")

    if assembly == "em_ez_fd" and mesh.n_faces > 0:
        faces = np.asarray(mesh.faces)
        n_faces = int(faces.shape[0])
        add(f"CELLS {n_faces} {4 * n_faces}\n")
        for a, b, c in faces:
            add(f"3 {int(a)} {int(b)} {int(c)}\n")
        add(f"CELL_TYPES {n_faces}\n")
        out.extend("5\n" for _ in range(n_faces))
    else:
        # ``Cells = msh->tetNodes`` is empty for 2D meshes, which is what the
        # oracle's ``em_e_qs`` branch writes: ``CELLS 0 0`` and no cell types.
        tetras = getattr(mesh, "tetras", None)
        tetras = np.zeros((0, 4), dtype=np.int64) if tetras is None else np.asarray(tetras)
        n_tetras = int(tetras.shape[0])
        add(f"CELLS {n_tetras} {5 * n_tetras}\n")
        for a, b, c, d in tetras:
            add(f"4 {int(a)} {int(b)} {int(c)} {int(d)}\n")
        add(f"CELL_TYPES {n_tetras}\n")
        out.extend("10\n" for _ in range(n_tetras))


def _tmz_node_values(mesh, sol: np.ndarray) -> List[tuple]:
    """The 2D ``em_ez_fd`` blocks: ``(count, NodeVal)`` in oracle order."""
    n_nodes = mesh.n_nodes
    columns = int(sol.shape[1])
    blocks = []

    full = np.sum(sol, axis=1)
    node_val = np.zeros((n_nodes, 3), dtype=complex)
    stop = min(n_nodes, full.size)
    node_val[:stop, 2] = full[:stop] * _SQRT2
    blocks.append((0, node_val))

    if columns < 10:
        for jj in range(columns):
            node_val = np.zeros((n_nodes, 3), dtype=complex)
            column = sol[:, jj]
            stop = min(n_nodes, column.size)
            node_val[:stop, 2] = column[:stop] * _SQRT2
            blocks.append((jj + 1, node_val))
    return blocks


def _evaluate_hcurl(mesh, full: np.ndarray, order: int) -> np.ndarray:
    """Node-averaged H(curl) field for one solution column.

    Reproduces the ``NodeVal`` loop shared by the 3D ``em_e_fd`` E and H
    branches, including ``*= sqrt(2)`` and the ``SumTimes`` division.
    """
    n_nodes = mesh.n_nodes
    node_val = np.zeros((n_nodes, 3), dtype=complex)
    sum_times = np.zeros(n_nodes)

    for tit in range(mesh.n_tetras):
        dofs = vector_dof_map(mesh, order, tit)
        _, inv_j = tetrahedron_jacobian(mesh.tet_geo(tit))
        t_sol = full[dofs]
        shp = shape_3d(order, _LOC_NODE, inv_j)
        for i in range(4):
            node = int(mesh.tetras[tit, i])
            node_val[node] += shp.Nv[i].T @ t_sol
            sum_times[node] += 1.0

    node_val *= _SQRT2
    with np.errstate(divide="ignore", invalid="ignore"):
        node_val /= sum_times[:, None]
    return node_val


def _evaluate_curl_h(mesh, full: np.ndarray, order: int, freq: float) -> np.ndarray:
    """The 3D H block: ``Hconst * curl`` with ``Hconst = j/(2 pi mu0 f)``."""
    n_nodes = mesh.n_nodes
    node_val = np.zeros((n_nodes, 3), dtype=complex)
    sum_times = np.zeros(n_nodes)
    h_const = 1j / (2.0 * PI * MU0 * freq)

    for tit in range(mesh.n_tetras):
        dofs = vector_dof_map(mesh, order, tit)
        _, inv_j = tetrahedron_jacobian(mesh.tet_geo(tit))
        t_sol = full[dofs]
        shp = shape_3d(order, _LOC_NODE, inv_j)
        for i in range(4):
            node = int(mesh.tetras[tit, i])
            node_val[node] += h_const * (shp.dNv[i].T @ t_sol)
            sum_times[node] += 1.0

    node_val *= _SQRT2
    with np.errstate(divide="ignore", invalid="ignore"):
        node_val /= sum_times[:, None]
    return node_val


# ── AV field reconstruction ───────────────────────────────────────────────


def _evaluate_av_e(
    mesh, sol_v: np.ndarray, sol_a: np.ndarray, order: int, freq: float,
    dofnum_s_offset: int = 0,
) -> np.ndarray:
    """E = -grad(V) - i*omega*A at mesh nodes, node-averaged.

    *sol_v* is the scalar (V) part, indexed by scalar DOF numbers.
    *sol_a* is the vector (A) part, indexed by vector DOF numbers
    (i.e. already offset by dofnum_s).
    """
    n_nodes = mesh.n_nodes
    omega = 2.0 * PI * freq
    node_val = np.zeros((n_nodes, 3), dtype=complex)
    sum_times = np.zeros(n_nodes)

    for tit in range(mesh.n_tetras):
        s_dofs = scalar_dof_map(mesh, order, tit)
        v_dofs = vector_dof_map(mesh, order, tit)
        _, inv_j = tetrahedron_jacobian(mesh.tet_geo(tit))
        shp = shape_3d(order, _LOC_NODE, inv_j)
        v_local = sol_v[s_dofs]
        a_local = sol_a[v_dofs]
        for i in range(4):
            node = int(mesh.tetras[tit, i])
            # grad(V) at reference node i: dNs[i].T @ v_local
            grad_v = shp.dNs[i].T @ v_local
            # A at reference node i: Nv[i].T @ a_local
            a_val = shp.Nv[i].T @ a_local
            node_val[node] += -grad_v - 1j * omega * a_val
            sum_times[node] += 1.0

    node_val *= _SQRT2
    with np.errstate(divide="ignore", invalid="ignore"):
        node_val /= sum_times[:, None]
    return node_val


def _evaluate_av_h(
    mesh, sol_a: np.ndarray, order: int, freq: float,
) -> np.ndarray:
    """H = (1/mur) * curl(A) at mesh nodes, node-averaged.

    *sol_a* is the vector (A) part, indexed by vector DOF numbers.
    """
    n_nodes = mesh.n_nodes
    node_val = np.zeros((n_nodes, 3), dtype=complex)
    sum_times = np.zeros(n_nodes)

    for tit in range(mesh.n_tetras):
        material = mesh.material_for_tet(tit)
        mur = float(material.mur) if material is not None else 1.0
        v_dofs = vector_dof_map(mesh, order, tit)
        _, inv_j = tetrahedron_jacobian(mesh.tet_geo(tit))
        t_sol = sol_a[v_dofs]
        shp = shape_3d(order, _LOC_NODE, inv_j)
        for i in range(4):
            node = int(mesh.tetras[tit, i])
            node_val[node] += (1.0 / mur) * (shp.dNv[i].T @ t_sol)
            sum_times[node] += 1.0

    node_val *= _SQRT2
    with np.errstate(divide="ignore", invalid="ignore"):
        node_val /= sum_times[:, None]
    return node_val


def write_field_vtk(
    mesh,
    sol: np.ndarray,
    freq: float,
    field_name: str | Path,
    *,
    assembly: str,
    order: int,
    dofnum_s: int = 0,
) -> Path:
    """Write ``<field_name>_<freq>.vtk`` exactly like ``field::field``.

    *sol* is the full DOF solution, shaped ``(n_dof, n_columns)``: for 2D
    ``em_ez_fd`` that is the expanded TFE solution
    (:func:`fes.core.assemblers.em_ez_fd.extract_solution_full`); for 3D
    ``em_e_fd`` it is :attr:`~fes.core.assemblers.em_e_fd.FDSystem.Sol`;
    for 3D ``em_av_fd`` it is :attr:`~fes.core.assemblers.em_av_fd.AVSystem.Sol`
    with *dofnum_s* giving the split between the scalar (V) and vector (A)
    parts.
    """
    if assembly not in ("em_ez_fd", "em_e_fd", "em_e_qs", "em_av_fd"):
        raise NotImplementedError(
            f"field export for {assembly!r} is not ported yet"
        )

    sol = np.asarray(sol)
    freq_text = freq_tag(freq)
    target = Path(f"{field_name}_{freq_text}.vtk")

    out: List[str] = []
    _dumpmesh(out, mesh, assembly)
    first = True

    if assembly == "em_e_qs":
        node_val = np.full((mesh.n_nodes, 3), complex(np.nan, np.nan))
        out.append(f"POINT_DATA {mesh.n_nodes}\n")
        out.append(f"SCALARS {freq_text}_Phi_0_[V] float 1\n")
        out.append("LOOKUP_TABLE jet\n")
        out.extend(f"{_f32(_norm(row))}\n" for row in node_val)
        _append_efield(out, freq_text, node_val, 1, False)
        first = False
    elif assembly == "em_ez_fd":
        for count, node_val in _tmz_node_values(mesh, sol):
            _append_efield(out, freq_text, node_val, count, first)
            first = False
    elif assembly == "em_av_fd":
        # Split [V, A] solution: sol[:dofnum_s, :] is V, sol[dofnum_s:, :] is A.
        sol_v = sol[:dofnum_s, :]
        sol_a = sol[dofnum_s:, :]
        full_v = np.sum(sol_v, axis=1)
        full_a = np.sum(sol_a, axis=1)
        _append_efield(
            out, freq_text, _evaluate_av_e(mesh, full_v, full_a, order, freq), 0, first
        )
        first = False
        columns = int(sol.shape[1])
        if columns < 10:
            for jj in range(columns):
                _append_efield(
                    out,
                    freq_text,
                    _evaluate_av_e(mesh, sol_v[:, jj], sol_a[:, jj], order, freq),
                    jj + 1,
                    first,
                )
                _append_hfield(
                    out,
                    freq_text,
                    _evaluate_av_h(mesh, sol_a[:, jj], order, freq),
                    jj + 1,
                    first,
                )
                first = False
    else:
        full = np.sum(sol, axis=1)
        _append_efield(
            out, freq_text, _evaluate_hcurl(mesh, full, order), 0, first
        )
        first = False
        columns = int(sol.shape[1])
        if columns < 10:
            for jj in range(columns):
                column = sol[:, jj]
                _append_efield(
                    out,
                    freq_text,
                    _evaluate_hcurl(mesh, column, order),
                    jj + 1,
                    first,
                )
                _append_hfield(
                    out,
                    freq_text,
                    _evaluate_curl_h(mesh, column, order, freq),
                    jj + 1,
                    first,
                )
                first = False

    target.write_text("".join(out))
    return target
