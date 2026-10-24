"""Tests for the 3D half of :mod:`fes.core.dof`.

``dof.cpp`` turns TetGen's global edge and face numbering into per-tetrahedron
DOF maps.  The maps are transcribed by hand in ``EXPECTED_*`` below rather than
recomputed from the implementation, and the mesh uses a deliberately shuffled
edge/face numbering so that a lookup that ignores ``tet_edges``/``tet_faces``
cannot pass.

The mesh is two tetrahedra, ``(0,1,2,3)`` and ``(0,1,2,4)``, sharing the face
``(0,1,2)``; that face is where the sharing checks live.
"""

from __future__ import annotations

import numpy as np
import pytest

from fes.core.dof import (
    face_dof_map,
    scalar_dof_count,
    scalar_dof_map,
    vector_dof_count,
    vector_dof_map,
)
from fes.mesh.mesh3d import Mesh3D

N_NODES = 5
N_EDGES = 9
N_FACES = 7
N_TETRAS = 2

#: Shuffled global edge numbering.  Index: (1,2)=0 (0,3)=1 (0,1)=2 (2,4)=3
#: (1,4)=4 (0,2)=5 (1,3)=6 (0,4)=7 (2,3)=8.
EDGES = np.array(
    [[1, 2], [0, 3], [0, 1], [2, 4], [1, 4], [0, 2], [1, 3], [0, 4], [2, 3]],
    dtype=np.int64,
)

#: Shuffled global face numbering (rows sorted ascending, as TetGen emits).
FACES = np.array(
    [
        [1, 2, 3],
        [0, 1, 4],
        [0, 2, 4],
        [1, 2, 4],
        [0, 1, 3],
        [0, 2, 3],
        [0, 1, 2],
    ],
    dtype=np.int64,
)

TET_EDGES = np.array(
    [
        [2, 5, 1, 0, 6, 8],  # (0,1) (0,2) (0,3) (1,2) (1,3) (2,3)
        [2, 5, 7, 0, 4, 3],  # (0,1) (0,2) (0,4) (1,2) (1,4) (2,4)
    ],
    dtype=np.int64,
)

TET_FACES = np.array(
    [
        [0, 5, 4, 6],  # opposite 0:(1,2,3) 1:(0,2,3) 2:(0,1,3) 3:(0,1,2)
        [3, 2, 1, 6],  # opposite 0:(1,2,4) 1:(0,2,4) 2:(0,1,4) 3:(0,1,2)
    ],
    dtype=np.int64,
)


def make_mesh(*, drop_edge=None, drop_face=None) -> Mesh3D:
    nodes = np.zeros((N_NODES, 3))
    nodes[1] = (1.0, 0.0, 0.0)
    nodes[2] = (0.0, 1.0, 0.0)
    nodes[3] = (0.0, 0.0, 1.0)
    nodes[4] = (0.0, 0.0, -1.0)

    edges = EDGES.copy()
    edge_labels = np.arange(N_EDGES, dtype=np.uint64)
    if drop_edge is not None:
        keep = [i for i, edge in enumerate(edges) if tuple(edge) != drop_edge]
        edges = edges[keep]
        edge_labels = edge_labels[keep]

    faces = FACES.copy()
    face_labels = np.zeros(N_FACES, dtype=np.uint64)
    if drop_face is not None:
        keep = [i for i, face in enumerate(faces) if tuple(face) != drop_face]
        faces = faces[keep]
        face_labels = face_labels[keep]

    mesh = Mesh3D(
        nodes=nodes,
        tetras=np.array([[0, 1, 2, 3], [0, 1, 2, 4]], dtype=np.int64),
        tet_labels=np.zeros(N_TETRAS, dtype=np.uint64),
        edges=edges,
        edge_labels=edge_labels,
        faces=faces,
        face_labels=face_labels,
    )
    mesh.normalize_order()
    mesh.build_tet_connectivity()
    return mesh


# ── connectivity ────────────────────────────────────────────────────────────


def test_connectivity_matches_the_oracle_local_order() -> None:
    mesh = make_mesh()
    np.testing.assert_array_equal(mesh.tet_edges, TET_EDGES)
    np.testing.assert_array_equal(mesh.tet_faces, TET_FACES)


def test_repeated_edges_and_faces_resolve_to_the_last_entry() -> None:
    """``Createmesh`` assigns with ``map[key] = i``; later rows win."""
    mesh = make_mesh()
    mesh.edges = np.vstack([mesh.edges, [[0, 3]]])  # duplicate of edge 1
    mesh.faces = np.vstack([mesh.faces, [[0, 1, 2]]])
    mesh.build_tet_connectivity()
    assert mesh.tet_edges[0, 2] == N_EDGES  # (0,3) now points at the copy
    assert mesh.tet_faces[0, 3] == N_FACES  # (0,1,2) now points at the copy


def test_missing_edge_or_face_maps_to_zero() -> None:
    """``std::map::operator[]`` inserts zero for a missing key."""
    mesh = make_mesh(drop_edge=(0, 3), drop_face=(0, 2, 3))
    assert mesh.tet_edges[0, 2] == 0  # (0,3) was removed
    assert mesh.tet_faces[0, 1] == 0  # (0,2,3) was removed
    # Everything else still resolves; removing entry 1 of the edge list
    # renumbers (0,1) from 2 to 1.
    assert mesh.tet_edges[0, 0] == 1
    assert mesh.tet_faces[0, 0] == 0


# ── counts ──────────────────────────────────────────────────────────────────


def test_dof_counts_match_the_oracle_formulas() -> None:
    mesh = make_mesh()
    assert [scalar_dof_count(mesh, p) for p in range(1, 5)] == [
        N_NODES,
        N_NODES + N_EDGES,
        N_NODES + 2 * N_EDGES + N_FACES,
        N_NODES + 3 * N_EDGES + 3 * N_FACES + N_TETRAS,
    ] == [5, 14, 30, 55]
    assert [vector_dof_count(mesh, p) for p in range(1, 5)] == [
        N_EDGES,
        2 * (N_EDGES + N_FACES),
        3 * (N_EDGES + N_TETRAS) + 6 * N_FACES,
        4 * N_EDGES + 12 * N_FACES + 12 * N_TETRAS,
    ] == [9, 32, 75, 144]


# ── per-element maps ────────────────────────────────────────────────────────

EXPECTED_SCALAR = {
    (0, 1): [0, 1, 2, 3],
    (0, 2): [0, 1, 2, 3, 7, 10, 6, 5, 11, 13],
    (0, 3): [
        0, 1, 2, 3,
        7, 10, 6, 5, 11, 13,
        16, 19, 15, 14, 20, 22,
        23, 28, 27, 29,
    ],
    (0, 4): [
        0, 1, 2, 3,
        7, 10, 6, 5, 11, 13,
        16, 19, 15, 14, 20, 22,
        25, 28, 24, 23, 29, 31,
        32, 37, 36, 38,
        39, 44, 43, 45,
        46, 51, 50, 52,
        53,
    ],
    (1, 1): [0, 1, 2, 4],
    (1, 2): [0, 1, 2, 4, 7, 10, 12, 5, 9, 8],
    (1, 3): [
        0, 1, 2, 4,
        7, 10, 12, 5, 9, 8,
        16, 19, 21, 14, 18, 17,
        26, 25, 24, 29,
    ],
    (1, 4): [
        0, 1, 2, 4,
        7, 10, 12, 5, 9, 8,
        16, 19, 21, 14, 18, 17,
        25, 28, 30, 23, 27, 26,
        35, 34, 33, 38,
        42, 41, 40, 45,
        49, 48, 47, 52,
        54,
    ],
}

EXPECTED_VECTOR = {
    (0, 1): [2, 5, 1, 0, 6, 8],
    (1, 1): [2, 5, 7, 0, 4, 3],
    (0, 2): [
        2, 5, 1, 0, 6, 8,
        11, 14, 10, 9, 15, 17,
        18, 25, 23, 30, 22, 29, 24, 31,
    ],
    (1, 2): [
        2, 5, 7, 0, 4, 3,
        11, 14, 16, 9, 13, 12,
        21, 28, 20, 27, 19, 26, 24, 31,
    ],
    (0, 3): [
        2, 5, 1, 0, 6, 8,
        11, 14, 10, 9, 15, 17,
        18, 25, 23, 30, 22, 29, 24, 31,
        34, 37, 33, 32, 38, 40,
        41, 48, 55, 62,
        46, 53, 60, 67,
        45, 52, 59, 66,
        47, 54, 61, 68,
        69, 71, 73,
    ],
    (1, 3): [
        2, 5, 7, 0, 4, 3,
        11, 14, 16, 9, 13, 12,
        21, 28, 20, 27, 19, 26, 24, 31,
        34, 37, 39, 32, 36, 35,
        44, 51, 58, 65,
        43, 50, 57, 64,
        42, 49, 56, 63,
        47, 54, 61, 68,
        70, 72, 74,
    ],
}

#: The order-4 tail (indices 45-83) of tetrahedron 0.  Indices 0-44 are the
#: order-3 map, which ``test_order_four_extends_order_three`` pins separately.
EXPECTED_VECTOR_4_TAIL = [
    77, 80, 76, 75, 81, 83,
    84, 91, 98, 105, 112, 119,
    89, 96, 103, 110, 117, 124,
    88, 95, 102, 109, 116, 123,
    90, 97, 104, 111, 118, 125,
    126, 128, 130, 132, 134, 136, 138, 140, 142,
]


@pytest.mark.parametrize("key", sorted(EXPECTED_SCALAR))
def test_scalar_map_matches_the_oracle(key) -> None:
    tet_id, order = key
    mesh = make_mesh()
    np.testing.assert_array_equal(
        scalar_dof_map(mesh, order, tet_id), EXPECTED_SCALAR[key]
    )


@pytest.mark.parametrize("key", sorted(EXPECTED_VECTOR))
def test_vector_map_matches_the_oracle(key) -> None:
    tet_id, order = key
    mesh = make_mesh()
    np.testing.assert_array_equal(
        vector_dof_map(mesh, order, tet_id), EXPECTED_VECTOR[key]
    )


def test_order_four_extends_order_three() -> None:
    mesh = make_mesh()
    for tet_id in range(N_TETRAS):
        order_four = vector_dof_map(mesh, 4, tet_id)
        np.testing.assert_array_equal(
            order_four[:45], vector_dof_map(mesh, 3, tet_id)
        )


def test_order_four_vector_tail_matches_the_oracle() -> None:
    mesh = make_mesh()
    np.testing.assert_array_equal(
        vector_dof_map(mesh, 4, 0)[45:], EXPECTED_VECTOR_4_TAIL
    )


@pytest.mark.parametrize("order", range(1, 5))
def test_scalar_maps_cover_every_dof(order: int) -> None:
    mesh = make_mesh()
    used = set()
    for tet_id in range(N_TETRAS):
        dofs = scalar_dof_map(mesh, order, tet_id)
        assert len(set(dofs.tolist())) == len(dofs)
        assert dofs.min() >= 0
        assert dofs.max() < scalar_dof_count(mesh, order)
        used.update(dofs.tolist())
    assert used == set(range(scalar_dof_count(mesh, order)))


@pytest.mark.parametrize("order", range(1, 5))
def test_vector_maps_cover_every_dof(order: int) -> None:
    mesh = make_mesh()
    used = set()
    for tet_id in range(N_TETRAS):
        dofs = vector_dof_map(mesh, order, tet_id)
        assert len(set(dofs.tolist())) == len(dofs)
        assert dofs.min() >= 0
        assert dofs.max() < vector_dof_count(mesh, order)
        used.update(dofs.tolist())
    assert used == set(range(vector_dof_count(mesh, order)))


def test_shared_face_dofs_are_identical_in_both_tetrahedra() -> None:
    """Two tets sharing ``(0,1,2)`` must agree on every shared DOF."""
    mesh = make_mesh()
    shared = (
        {0, 1, 2}  # nodes of the shared face
        | {N_NODES + edge for edge in (2, 5, 0)}  # (0,1) (0,2) (1,2)
        | {N_NODES + N_EDGES + edge for edge in (2, 5, 0)}
        | {N_NODES + 2 * N_EDGES + edge for edge in (2, 5, 0)}
        | {N_NODES + 3 * N_EDGES + face for face in (6,)}  # face (0,1,2)
        | {N_NODES + 3 * N_EDGES + N_FACES + face for face in (6,)}
        | {N_NODES + 3 * N_EDGES + 2 * N_FACES + face for face in (6,)}
    )
    first = set(scalar_dof_map(mesh, 4, 0).tolist())
    second = set(scalar_dof_map(mesh, 4, 1).tolist())
    assert first & second == shared


# ── boundary faces ──────────────────────────────────────────────────────────

#: ``dof(prj, 2, 6)`` on the shared face (0,1,2), whose 3D ``face_edges`` row
#: is ``[e12, e02, e01] = [0, 5, 2]``.  The vector block is therefore the
#: reverse of what the 2D path would produce for the same triangle.
EXPECTED_FACE_SCALAR = {
    1: [0, 1, 2],
    2: [0, 1, 2, 5, 10, 7],
    3: [0, 1, 2, 5, 10, 7, 14, 19, 16, 29],
    4: [0, 1, 2, 5, 10, 7, 14, 19, 16, 23, 28, 25, 38, 45, 52],
}

EXPECTED_FACE_VECTOR = {
    1: [0, 5, 2],
    2: [0, 5, 2, 9, 14, 11, 24, 31],
    3: [0, 5, 2, 9, 14, 11, 24, 31, 32, 37, 34, 47, 54, 61, 68],
    4: [
        0, 5, 2, 9, 14, 11, 24, 31, 32, 37, 34, 47, 54, 61, 68,
        69, 74, 71, 84, 91, 98, 105, 112, 119,
    ],
}

SHARED_FACE = 6


def make_boundary_mesh() -> Mesh3D:
    mesh = make_mesh()
    mesh.build_face_edges()
    mesh.build_face_adjacency()
    return mesh


@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_boundary_face_scalar_map_matches_the_oracle(order: int) -> None:
    mesh = make_boundary_mesh()
    result = face_dof_map(mesh, order, SHARED_FACE)
    np.testing.assert_array_equal(result.s, EXPECTED_FACE_SCALAR[order])


@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_boundary_face_vector_map_matches_the_oracle(order: int) -> None:
    mesh = make_boundary_mesh()
    result = face_dof_map(mesh, order, SHARED_FACE)
    np.testing.assert_array_equal(result.v, EXPECTED_FACE_VECTOR[order])


def test_boundary_face_vector_block_is_the_reversed_face_edges_row() -> None:
    """``Createmesh`` stores ``[e12, e02, e01]``; the shape basis wants the reverse."""
    mesh = make_boundary_mesh()
    np.testing.assert_array_equal(mesh.face_edges[SHARED_FACE], [0, 5, 2])
    for order in (1, 2, 3, 4):
        v = face_dof_map(mesh, order, SHARED_FACE).v
        np.testing.assert_array_equal(v[:3], [0, 5, 2])
        if order > 1:
            np.testing.assert_array_equal(v[3:6], N_EDGES + np.array([0, 5, 2]))


def test_order_four_face_block_aliases_the_tet_volume_block() -> None:
    """v0.1 puts the order-4 face edge DOFs at ``3 nE + 6 nF + e``.

    The tetrahedron path puts its own order-4 edge block at
    ``3 nE + 6 nF + 3 nT + e``, three volume blocks higher, so the two
    disagree and the face block overlaps the first volume block whenever
    ``e < nT``.  Nothing reaches order 4 (``ele_mat`` stops at 3), but the map
    itself is reproduced literally.
    """
    mesh = make_boundary_mesh()
    v = face_dof_map(mesh, 4, SHARED_FACE).v
    base = 3 * N_EDGES + 6 * N_FACES
    np.testing.assert_array_equal(v[15:18], base + np.array([0, 5, 2]))
    assert v[15] < base + N_TETRAS  # overlaps the tet volume block


def test_boundary_face_dofs_agree_with_the_shared_tet_dofs() -> None:
    """The shared face's DOF numbers must already exist in both tetrahedra.

    Only orders 1-3: see ``test_order_four_face_block_aliases_the_tet_volume_block``.
    """
    mesh = make_boundary_mesh()
    for order in (1, 2, 3):
        face = face_dof_map(mesh, order, SHARED_FACE)
        for kind, dofs in (("s", face.s), ("v", face.v)):
            mapper = scalar_dof_map if kind == "s" else vector_dof_map
            first = set(mapper(mesh, order, 0).tolist())
            second = set(mapper(mesh, order, 1).tolist())
            assert set(dofs.tolist()) <= first & second


def test_face_map_requires_face_edges() -> None:
    mesh = make_mesh()
    with pytest.raises(ValueError, match="build_face_edges"):
        face_dof_map(mesh, 1, SHARED_FACE)


def test_connectivity_is_required() -> None:
    mesh = make_mesh()
    mesh.tet_edges = None
    with pytest.raises(ValueError, match="build_tet_connectivity"):
        scalar_dof_map(mesh, 2, 0)
    mesh = make_mesh()
    mesh.tet_faces = None
    with pytest.raises(ValueError, match="build_tet_connectivity"):
        vector_dof_map(mesh, 2, 0)


@pytest.mark.parametrize("order", [0, 5])
def test_rejects_unsupported_order(order: int) -> None:
    mesh = make_mesh()
    with pytest.raises(ValueError):
        scalar_dof_count(mesh, order)
    with pytest.raises(ValueError):
        scalar_dof_map(mesh, order, 0)
    with pytest.raises(ValueError):
        vector_dof_count(mesh, order)
    with pytest.raises(ValueError):
        vector_dof_map(mesh, order, 0)
