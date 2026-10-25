"""Tests for the 2D mesh layer (:mod:`fes.mesh.mesh2d`).

``build_2d_edge_connectivity`` decides the global edge numbering, which in
turn fixes the finite-element DOF numbering.  These tests pin the algorithm
with a hand-built mesh (no mesher needed) and check the real ``BilatFilter``
mesh where Triangle is available.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fes.mesh.mesh2d import MAX_LABEL, Mesh2D, mesh_from_poly

DATA = Path(__file__).resolve().parents[1] / "data"


def _mesh(nodes, faces, edges) -> Mesh2D:
    xy = np.asarray(nodes, dtype=float).reshape(-1, 2)
    mesh = Mesh2D(
        nodes=np.column_stack([xy, np.zeros(xy.shape[0])]),
        faces=np.asarray(faces, dtype=np.int64),
        face_labels=np.ones(len(faces), dtype=np.int64),
        edges=np.asarray(edges, dtype=np.int64),
        edge_labels=np.arange(1, len(edges) + 1, dtype=np.uint64),
        n_boundary_edges=len(edges),
    )
    mesh.normalize_order()
    mesh.build_2d_edge_connectivity()
    return mesh


def test_sorting_normalizes_faces_and_edges() -> None:
    mesh = _mesh(
        [(0, 0), (1, 0), (0, 1), (1, 1)],
        [[2, 0, 1], [3, 2, 1]],
        [[1, 0], [3, 1], [3, 2], [2, 0]],
    )
    assert mesh.faces.tolist() == [[0, 1, 2], [1, 2, 3]]
    assert mesh.edges[:4].tolist() == [[0, 1], [1, 3], [2, 3], [0, 2]]
    assert mesh.n_edges == 5  # plus the (1, 2) diagonal


def test_connectivity_matches_euler() -> None:
    """Two triangles sharing a diagonal: V - E + F = 1, every edge used."""
    mesh = _mesh(
        [(0, 0), (1, 0), (0, 1), (1, 1)],
        [[0, 1, 2], [1, 2, 3]],
        [[0, 1], [1, 3], [2, 3], [0, 2]],
    )
    # PLC has four boundary edges plus the (1, 2) diagonal.
    assert mesh.n_edges == 5
    assert mesh.n_nodes - mesh.n_edges + mesh.n_faces == 1
    assert sorted(np.unique(mesh.face_edges).tolist()) == list(range(5))
    diagonal = [i for i, edge in enumerate(mesh.edges) if edge.tolist() == [1, 2]]
    assert len(diagonal) == 1
    assert mesh.face_edges[:, :].tolist() == [
        [0, 3, diagonal[0]],
        [diagonal[0], 1, 2],
    ]


def test_derived_edges_get_the_sentinel_label() -> None:
    mesh = _mesh(
        [(0, 0), (1, 0), (0, 1), (1, 1)],
        [[0, 1, 2], [1, 2, 3]],
        [[0, 1], [1, 3], [2, 3], [0, 2]],
    )
    assert mesh.edge_labels[: mesh.n_boundary_edges].tolist() == [1, 2, 3, 4]
    assert mesh.edge_labels[mesh.n_boundary_edges :].tolist() == [MAX_LABEL]
    assert MAX_LABEL == 2**32 - 1


def test_legacy_connectivity_duplicates_shared_edges() -> None:
    """The v0.1 unsorted-pair quirk, in miniature.

    Both triangles contain the diagonal ``(0, 3)`` as their *smallest and
    largest* vertices, so the sorted key is never inserted: the ``(n0, n2)``
    lookup misses and ``std::map::operator[]`` hands back edge 0.
    """
    nodes = [(0, 0), (1, 0), (0, 1), (1, 1)]
    faces = [[0, 1, 3], [0, 2, 3]]
    edges = [[0, 1], [1, 3], [3, 2], [2, 0]]

    corrected = _mesh(nodes, faces, edges)
    assert corrected.n_edges == 5
    assert corrected.n_nodes - corrected.n_edges + corrected.n_faces == 1
    shared = [i for i, e in enumerate(corrected.edges) if e.tolist() == [0, 3]]
    assert len(shared) == 1
    assert corrected.face_edges[:, 1].tolist() == [shared[0], shared[0]]

    mesh = Mesh2D(
        nodes=np.column_stack([np.asarray(nodes, dtype=float), np.zeros(len(nodes))]),
        faces=np.asarray(faces, dtype=np.int64),
        face_labels=np.ones(len(faces), dtype=np.int64),
        edges=np.asarray(edges, dtype=np.int64),
        edge_labels=np.zeros(len(edges), dtype=np.uint64),
        n_boundary_edges=len(edges),
    )
    mesh.normalize_order()
    mesh.build_2d_edge_connectivity(legacy_unsorted_pairs=True)
    # ``(3, 0)`` is stored as an extra edge while the lookup asks for ``(0, 3)``.
    assert [3, 0] in mesh.edges.tolist()
    assert [0, 3] not in mesh.edges.tolist()
    assert mesh.face_edges[0, 1] == 0
    assert mesh.edge_connectivity_legacy is True


def test_rebuild_switches_modes() -> None:
    mesh = _mesh(
        [(0, 0), (1, 0), (0, 1), (1, 1)],
        [[0, 1, 2], [1, 2, 3]],
        [[0, 1], [1, 3], [2, 3], [0, 2]],
    )
    corrected_edges = mesh.edges.copy()
    mesh.build_2d_edge_connectivity(legacy_unsorted_pairs=True)
    legacy_count = mesh.n_edges
    assert legacy_count > corrected_edges.shape[0]
    mesh.build_2d_edge_connectivity(legacy_unsorted_pairs=False)
    assert np.array_equal(mesh.edges, corrected_edges)


def test_missing_face_edges_is_an_error() -> None:
    mesh = Mesh2D(
        nodes=np.zeros((3, 3)),
        faces=np.array([[0, 1, 2]]),
        face_labels=np.ones(1, dtype=np.int64),
        edges=np.zeros((0, 2), dtype=np.int64),
        edge_labels=np.zeros(0, dtype=np.uint64),
    )
    from fes.core.dof import scalar_dof_map

    with pytest.raises(ValueError):
        scalar_dof_map(mesh, 2, 0)


# ── the real mesh ───────────────────────────────────────────────────────────


@pytest.mark.triangle
def test_bilatfilter_connectivity() -> None:
    mesh = mesh_from_poly(DATA / "BilatFilter.poly")
    # Triangle may produce slightly different mesh counts on different
    # platforms (floating-point differences in its predicates), so pin the
    # structural properties instead of exact counts.
    assert mesh.n_nodes - mesh.n_edges + mesh.n_faces == 1
    used = np.unique(mesh.face_edges)
    assert used.size == mesh.n_edges  # no orphan edges


@pytest.mark.triangle
def test_bilatfilter_legacy_connectivity_leaves_orphans() -> None:
    mesh = mesh_from_poly(DATA / "BilatFilter.poly")
    mesh.build_2d_edge_connectivity(legacy_unsorted_pairs=True)
    # Triangle may produce slightly different mesh counts on different
    # platforms; pin the structural property instead of exact counts.
    used = np.unique(mesh.face_edges)
    assert used.size < mesh.n_edges  # some edges belong to no triangle
