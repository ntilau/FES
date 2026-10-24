"""Tests for 2D homogeneous refinement (``mesh::refine_homogeneous``).

The reference implementation refines after preprocessing and *before* ``normalize_order``,
so the raw Triangle vertex order decides which mid-edge node gets which
index.  These tests pin that numbering on a hand-built mesh, then check that
the 2D drivers apply ``+h``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fes import api
from fes.core.options import parse_argv
from fes.mesh.mesh2d import Mesh2D

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"


def _raw_mesh() -> Mesh2D:
    """Two triangles sharing ``(1, 2)``, stored in unnormalized order."""
    xy = np.asarray([(0, 0), (1, 0), (0, 1), (1, 1)], dtype=float)
    return Mesh2D(
        nodes=np.column_stack([xy, np.zeros(xy.shape[0])]),
        faces=np.asarray([[2, 0, 1], [3, 2, 1]], dtype=np.int64),
        face_labels=np.asarray([7, 9], dtype=np.int64),
        edges=np.asarray([[1, 0], [3, 1], [3, 2], [2, 0]], dtype=np.int64),
        edge_labels=np.arange(1, 5, dtype=np.uint64),
        n_boundary_edges=4,
    )


def test_refine_splits_faces_edges_and_labels() -> None:
    mesh = _raw_mesh()
    mesh.refine_homogeneous()

    assert mesh.n_nodes == 9
    assert mesh.n_faces == 8
    assert mesh.n_boundary_edges == 8
    assert mesh.n_edges == 8  # only the split PLC segments exist yet

    # Mid-node numbers follow the face-encounter order of the sorted keys:
    # face 0 visits (0,2), (0,1), (1,2); face 1 adds (2,3), (1,3).
    np.testing.assert_allclose(
        mesh.nodes[4:],
        [(0.0, 0.5, 0.0), (0.5, 0.0, 0.0), (0.5, 0.5, 0.0),
         (0.5, 1.0, 0.0), (1.0, 0.5, 0.0)],
        rtol=0, atol=0,
    )
    assert mesh.faces.tolist() == [
        [2, 4, 6], [0, 5, 4], [1, 6, 5], [4, 5, 6],
        [3, 7, 8], [2, 6, 7], [1, 8, 6], [7, 6, 8],
    ]
    assert mesh.face_labels.tolist() == [7, 7, 7, 7, 9, 9, 9, 9]
    assert mesh.edges.tolist() == [
        [1, 5], [5, 0], [3, 8], [8, 1], [3, 7], [7, 2], [2, 4], [4, 0],
    ]
    assert mesh.edge_labels.tolist() == [1, 1, 2, 2, 3, 3, 4, 4]
    assert mesh.face_edges is None
    assert mesh.edge_connectivity_legacy is None


def test_refine_z_coordinate_stays_flat() -> None:
    mesh = _raw_mesh()
    mesh.refine_homogeneous()
    assert np.all(mesh.nodes[:, 2] == 0.0)


def test_refine_drops_stale_connectivity_before_splitting() -> None:
    mesh = _raw_mesh()
    mesh.normalize_order()
    mesh.build_2d_edge_connectivity()
    assert mesh.n_edges == 5  # four PLC segments plus the diagonal
    mesh.refine_homogeneous()
    # The reference splits only the PLC segments it read from the .poly file;
    # midpoint lookup for the diagonal comes from the face loop.
    assert mesh.n_boundary_edges == 8
    assert mesh.edges.shape == (8, 2)
    mesh.normalize_order()
    mesh.build_2d_edge_connectivity()
    assert mesh.n_nodes - mesh.n_edges + mesh.n_faces == 1


def test_two_levels_quadruple_again() -> None:
    mesh = _raw_mesh()
    # The reference refines the raw (unnormalized) mesh at every level.
    mesh.refine_homogeneous()
    mesh.refine_homogeneous()
    # Level 1: 9 nodes, 8 faces, 8 PLC edges.  The 8 triangles have 16 unique
    # edges (8 boundary + 8 interior), so level 2 adds 16 mid nodes.
    assert mesh.n_nodes == 25
    assert mesh.n_faces == 32
    assert mesh.n_boundary_edges == 16


def _mesh_counts(path: Path) -> tuple:
    """``(points, triangles, boundary_segments)`` from the driver's mesh VTK.

    ``mesh::Savefield`` writes triangles first, then the PLC segments, and
    ``CELL_TYPES`` tags them 5 and 3 (VTK_TRIANGLE / VTK_LINE).
    """
    lines = path.read_text().splitlines()
    points = int(next(line.split()[1] for line in lines if line.startswith("POINTS ")))
    n_cells = int(next(line.split()[1] for line in lines if line.startswith("CELLS ")))
    start = lines.index(next(line for line in lines if line.startswith("CELL_TYPES ")))
    types = lines[start + 1 : start + 1 + n_cells]
    return points, types.count("5"), types.count("3")


@pytest.mark.triangle
@pytest.mark.parametrize(
    "model, assembly, argv",
    [
        ("ModelScatteringVoid.poly", "em_ez_fd", ["+f", "299792458", "+p", "2"]),
        ("WR90_TL.poly", "em_e_tl_eig", ["+f", "1e10", "+p", "2"]),
    ],
)
def test_2d_drivers_apply_h_ord(
    model: str, assembly: str, argv: list, tmp_path: Path
) -> None:
    def run(h_ord: int, out: Path) -> tuple:
        options = parse_argv([str(DATA / model), *argv, "+h", str(h_ord)])
        plan = api.plan_run(options=options, command="run")
        assert plan.assembly == assembly
        assert plan.options.h_ord == h_ord
        plan.output_dir = str(out)
        api.load_drivers()
        api.DRIVERS[plan.assembly](plan)
        return _mesh_counts(out / f"{Path(model).stem}.vtk")

    base_points, base_triangles, base_segments = run(0, tmp_path / "h0")
    refined_points, refined_triangles, refined_segments = run(1, tmp_path / "h1")
    assert refined_triangles == 4 * base_triangles
    assert refined_segments == 2 * base_segments
    # Every unique edge gains a midpoint and no node is discarded.
    assert refined_points > base_points
