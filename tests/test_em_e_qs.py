"""Tests for the 2D electrostatic assembler and its driver.

These pin the assembly bookkeeping, the P1 stiffness, the boundary conditions
and the driver's output naming.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fes import api
from fes.core.assemblers import assemble_em_e_qs
from fes.core.assemblers.em_e_qs import _stiffness_rows
from fes.core.options import parse_argv
from fes.mesh.mesh2d import Mesh2D, mesh_from_poly
from fes.mesh.poly_header import Boundary, Solid

PY_DIR = Path(__file__).resolve().parents[1]
REPO = PY_DIR
CAPSENSE = REPO / "data" / "CapSense.poly"


def _triangle_mesh(*, epsr: float = 1.0, boundaries=None) -> Mesh2D:
    """One right triangle, (0,0)-(1,0)-(0,1), with an optional BC table.

    ``edges`` carries the three PLC segments and ``n_boundary_edges`` marks
    them all as such; ``mesh_from_poly`` would append derived interior edges
    after this prefix, which the assembler must ignore.
    """
    nodes = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    edges = np.array([[0, 1], [0, 2], [1, 2]], dtype=np.int64)
    labels = np.array([1, 2, 3], dtype=np.uint64)
    return Mesh2D(
        nodes=nodes,
        faces=faces,
        face_labels=np.zeros(1, dtype=np.int64),
        edges=edges,
        edge_labels=labels,
        materials={0: Solid(name="mat0", epsr=epsr, label=0)},
        boundaries=list(boundaries or []),
        n_boundary_edges=edges.shape[0],
    )


def _rows_to_dense(rows, n: int) -> np.ndarray:
    matrix = np.zeros((n, n), dtype=complex)
    for i, line in rows.items():
        for j, value in line.items():
            matrix[i, j] += value
    return matrix


# ── assembly ────────────────────────────────────────────────────────────────


def test_p1_stiffness_of_one_triangle() -> None:
    """``0.5 * [[2,-1,-1],[-1,1,0],[-1,0,1]]`` for the unit right triangle."""
    rows = _stiffness_rows(_triangle_mesh())
    expected = 0.5 * np.array(
        [[2.0, -1.0, -1.0], [-1.0, 1.0, 0.0], [-1.0, 0.0, 1.0]]
    )
    np.testing.assert_allclose(
        _rows_to_dense(rows, 3), expected, rtol=0, atol=1e-15
    )


def test_p1_stiffness_scales_with_epsr() -> None:
    scaled = _rows_to_dense(_stiffness_rows(_triangle_mesh(epsr=10.0)), 3)
    plain = _rows_to_dense(_stiffness_rows(_triangle_mesh()), 3)
    np.testing.assert_allclose(scaled, 10.0 * plain, rtol=1e-15, atol=1e-15)


def test_dofs_are_nodes_and_order_is_ignored() -> None:
    mesh = _triangle_mesh(
        boundaries=[Boundary(name="A", label=1, type="PerfectE")]
    )
    systems = [assemble_em_e_qs(mesh, order, {"A": 1.0}) for order in (1, 2, 3)]
    for system in systems:
        assert system.dof == mesh.n_nodes
        assert system.dirichlet.tolist() == [0, 1]


def test_no_voltage_raises_null_right_hand_side() -> None:
    mesh = _triangle_mesh(
        boundaries=[Boundary(name="A", label=1, type="PerfectE")]
    )
    with pytest.raises(RuntimeError, match="Null Right Hand Side"):
        assemble_em_e_qs(mesh, 1, {})


def test_voltage_on_an_unknown_boundary_is_zero() -> None:
    """``std::map::operator[]`` defaults to 0, so the RHS stays empty."""
    mesh = _triangle_mesh(
        boundaries=[Boundary(name="A", label=1, type="PerfectE")]
    )
    with pytest.raises(RuntimeError, match="Null Right Hand Side"):
        assemble_em_e_qs(mesh, 1, {"Other": 1.0})


def test_free_rows_satisfy_the_eliminated_system() -> None:
    mesh = _triangle_mesh(
        boundaries=[Boundary(name="A", label=1, type="PerfectE")]
    )
    system = assemble_em_e_qs(mesh, 1, {"A": 1.0})
    free = np.setdiff1d(np.arange(system.dof), system.dirichlet)
    assert free.tolist() == [2]
    residual = system.A @ system.phi - system.B[:, 0]
    np.testing.assert_allclose(residual[free], 0.0, atol=1e-13)
    np.testing.assert_allclose(system.phi.imag, 0.0, atol=1e-14)


def test_dirichlet_uses_plc_edges_only() -> None:
    """Derived interior edges carry ``MAX_LABEL`` and must not constrain."""
    mesh = _triangle_mesh(
        boundaries=[Boundary(name="A", label=1, type="PerfectE")]
    )
    mesh.edges = np.vstack([mesh.edges, [[1, 3]]])
    mesh.edge_labels = np.concatenate(
        [mesh.edge_labels, [np.iinfo(np.uint64).max]]
    )
    system = assemble_em_e_qs(mesh, 1, {"A": 1.0})
    assert system.dirichlet.tolist() == [0, 1]


def _with_materials(mesh: Mesh2D, materials: dict) -> Mesh2D:
    """The same mesh with a different ``#Solids`` table."""
    return Mesh2D(
        nodes=mesh.nodes,
        faces=mesh.faces,
        face_labels=mesh.face_labels,
        edges=mesh.edges,
        edge_labels=mesh.edge_labels,
        materials=materials,
        boundaries=mesh.boundaries,
        n_boundary_edges=mesh.n_boundary_edges,
    )


@pytest.mark.triangle
def test_material_comes_from_the_solids_line_index() -> None:
    """Region attribute 1 selects the second ``#Solids`` line."""
    mesh = mesh_from_poly(REPO / "data" / "CapSense2.poly")
    # mat0 (epsr 1) sits on attribute 0, mat1 (epsr 10) on attribute 1.
    epsr_by_face = [
        float(mesh.material_for_face(face).epsr) for face in range(mesh.n_faces)
    ]
    assert set(epsr_by_face) == {1.0, 10.0}

    rows = _rows_to_dense(_stiffness_rows(mesh), mesh.n_nodes)
    assert np.allclose(rows, rows.T)
    assert np.all(np.diag(rows).real > 0.0)

    # Vacuum is the default lookup: dropping the table changes the matrix.
    vacuum_rows = _rows_to_dense(
        _stiffness_rows(_with_materials(mesh, {})), mesh.n_nodes
    )
    assert not np.allclose(rows, vacuum_rows)

    # Scaling every solid's epsr scales the assembled stiffness exactly.
    scaled = _rows_to_dense(
        _stiffness_rows(
            _with_materials(
                mesh,
                {
                    key: Solid(
                        name=solid.name,
                        epsr=10.0 * solid.epsr,
                        label=solid.label,
                    )
                    for key, solid in mesh.materials.items()
                },
            )
        ),
        mesh.n_nodes,
    )
    np.testing.assert_allclose(scaled, 10.0 * rows, rtol=1e-14, atol=1e-13)


# ── driver ──────────────────────────────────────────────────────────────────


@pytest.mark.triangle
def test_field_tag_is_not_forced_by_an_explicit_formula(tmp_path: Path) -> None:
    options = parse_argv(
        [str(CAPSENSE), "+f", "0", "+p", "2", "+volt", "Elec", "1",
         "+formula", "em_e_qs"]
    )
    plan = api.plan_run(options=options, command="run")
    assert plan.assembly == "em_e_qs"
    assert options.field is False
    plan.output_dir = str(tmp_path)
    api.load_drivers()
    result = api.DRIVERS[plan.assembly](plan)
    assert result.systems[0].phi is not None
    assert sorted(path.name for path in tmp_path.iterdir()) == ["CapSense.vtk"]
