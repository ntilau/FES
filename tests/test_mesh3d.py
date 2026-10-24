"""Reference tests for the boundary-face half of :class:`fes.mesh.mesh3d.Mesh3D`.

The fixture is the same deliberately shuffled mesh as ``test_dof3d``: two
tetrahedra sharing face ``(0,1,2)``, with global edge and face numbering that
does not follow the natural order.  Every expected table below is written out
by hand from ``preprocessing::Createmesh`` rather than recomputed from the
implementation.

The point of the ``face_edges`` tests is the v0.1 quirk: the 3D path stores
``[e12, e02, e01]`` while both the 2D path and the H(curl) basis use
``[e01, e02, e12]``.
"""

from __future__ import annotations

import numpy as np
import pytest

from fes.mesh.mesh3d import Mesh3D
from fes.mesh.poly_header import Boundary, Solid

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

#: ``Createmesh``'s columns: ``(n1,n2) (n0,n2) (n0,n1)``.
EXPECTED_FACE_EDGES = np.array(
    [
        [8, 6, 0],  # (1,2,3): e23, e13, e12
        [4, 7, 2],  # (0,1,4): e14, e04, e01
        [3, 7, 5],  # (0,2,4): e24, e04, e02
        [3, 4, 0],  # (1,2,4): e24, e14, e12
        [6, 1, 2],  # (0,1,3): e13, e03, e01
        [8, 1, 5],  # (0,2,3): e23, e03, e02
        [0, 5, 2],  # (0,1,2): e12, e02, e01
    ],
    dtype=np.int64,
)

#: Face ``i`` touches tetrahedron 0 first for 0,4,5,6 and tetrahedron 1 otherwise.
EXPECTED_FACE_ADJ = [
    [0],
    [1],
    [1],
    [1],
    [0],
    [0],
    [0, 1],
]

#: The vertex of the first adjacent tet that is not on the face.
EXPECTED_INT_NODE = [
    (0.0, 0.0, 0.0),  # face 0 (1,2,3), tet 0 -> node 0
    (0.0, 1.0, 0.0),  # face 1 (0,1,4), tet 1 -> node 2
    (1.0, 0.0, 0.0),  # face 2 (0,2,4), tet 1 -> node 1
    (0.0, 0.0, 0.0),  # face 3 (1,2,4), tet 1 -> node 0
    (0.0, 1.0, 0.0),  # face 4 (0,1,3), tet 0 -> node 2
    (1.0, 0.0, 0.0),  # face 5 (0,2,3), tet 0 -> node 1
    (0.0, 0.0, 1.0),  # face 6 (0,1,2), tet 0 -> node 3
]


def make_mesh(**kwargs) -> Mesh3D:
    nodes = np.zeros((N_NODES, 3))
    nodes[1] = (1.0, 0.0, 0.0)
    nodes[2] = (0.0, 1.0, 0.0)
    nodes[3] = (0.0, 0.0, 1.0)
    nodes[4] = (0.0, 0.0, -1.0)
    mesh = Mesh3D(
        nodes=nodes,
        tetras=np.array([[0, 1, 2, 3], [0, 1, 2, 4]], dtype=np.int64),
        tet_labels=np.zeros(N_TETRAS, dtype=np.uint64),
        edges=EDGES.copy(),
        edge_labels=np.arange(N_EDGES, dtype=np.uint64),
        faces=FACES.copy(),
        face_labels=np.zeros(N_FACES, dtype=np.uint64),
        **kwargs,
    )
    mesh.normalize_order()
    mesh.build_tet_connectivity()
    return mesh


# ── face-edge table ─────────────────────────────────────────────────────────


def test_face_edges_matches_the_oracle_column_order() -> None:
    mesh = make_mesh()
    mesh.build_face_edges()
    np.testing.assert_array_equal(mesh.face_edges, EXPECTED_FACE_EDGES)


def test_face_edges_is_the_reverse_of_the_basis_order() -> None:
    """The 2D and shape-function order is ``[e01, e02, e12]``; 3D reverses it."""
    mesh = make_mesh()
    mesh.build_face_edges()
    for face_id, face in enumerate(FACES):
        n0, n1, n2 = (int(v) for v in face)
        for column, pair in enumerate(((n0, n1), (n0, n2), (n1, n2))):
            expected = int(np.flatnonzero((EDGES == np.array(pair)).all(axis=1))[0])
            assert mesh.face_edges[face_id, 2 - column] == expected


def test_missing_face_edge_maps_to_zero() -> None:
    """``std::map::operator[]`` inserts zero for a key that never got inserted."""
    mesh = make_mesh()
    keep = [
        i for i, edge in enumerate(mesh.edges) if tuple(edge) != (1, 2)
    ]
    mesh.edges = mesh.edges[keep]
    mesh.edge_labels = mesh.edge_labels[keep]
    mesh.build_face_edges()
    # e12 is column 0 for face (0,1,2) and column 2 for faces (1,2,3)/(1,2,4).
    assert mesh.face_edges[6, 0] == 0
    assert mesh.face_edges[0, 2] == 0
    assert mesh.face_edges[3, 2] == 0
    # The remaining columns still resolve; (0,1) moved from 2 to 1.
    assert mesh.face_edges[6, 2] == 1


def test_later_duplicate_edge_wins() -> None:
    mesh = make_mesh()
    mesh.edges = np.vstack([mesh.edges, [[1, 2]]])
    mesh.build_face_edges()
    assert mesh.face_edges[0, 2] == N_EDGES


# ── adjacency and geometry ─────────────────────────────────────────────────


def test_face_adjacency_matches_the_oracle() -> None:
    mesh = make_mesh()
    mesh.build_face_adjacency()
    assert [ids.tolist() for ids in mesh.face_adj_tet] == EXPECTED_FACE_ADJ


def test_face_adjacency_requires_tet_faces() -> None:
    mesh = make_mesh()
    mesh.tet_faces = None
    with pytest.raises(ValueError, match="build_tet_connectivity"):
        mesh.build_face_adjacency()


def test_tet_and_face_geometry_follow_the_stored_tables() -> None:
    mesh = make_mesh()
    np.testing.assert_array_equal(mesh.tet_geo(1), mesh.nodes[[0, 1, 2, 4]])
    np.testing.assert_array_equal(mesh.fac_geo(3), mesh.nodes[[1, 2, 4]])


def test_int_node_is_the_opposite_vertex_of_the_first_adjacent_tet() -> None:
    mesh = make_mesh()
    mesh.build_face_adjacency()
    for face_id, expected in enumerate(EXPECTED_INT_NODE):
        np.testing.assert_array_equal(mesh.int_node(face_id), expected)


def test_int_node_uses_the_first_adjacent_tet_only() -> None:
    """Face 6 is shared; the oracle takes the vertex across it in tet 0."""
    mesh = make_mesh()
    mesh.build_face_adjacency()
    assert mesh.face_adj_tet[6].tolist() == [0, 1]
    np.testing.assert_array_equal(mesh.int_node(6), mesh.nodes[3])


def test_int_node_requires_adjacency() -> None:
    mesh = make_mesh()
    with pytest.raises(ValueError, match="build_face_adjacency"):
        mesh.int_node(6)


# ── boundary labels and materials ──────────────────────────────────────────


def test_faces_with_label_are_reported_in_ascending_order() -> None:
    mesh = make_mesh()
    mesh.face_labels = np.array([7, 0, 7, 0, 0, 0, 3], dtype=np.uint64)
    np.testing.assert_array_equal(mesh.faces_with_label(7), [0, 2])
    np.testing.assert_array_equal(mesh.faces_with_label(3), [6])
    np.testing.assert_array_equal(mesh.faces_with_label(1), [])

    boundary = Boundary(name="port", label=7, type="WavePort", num_modes=1)
    np.testing.assert_array_equal(mesh.faces_for_boundary(boundary), [0, 2])


def test_boundary_face_material_comes_from_the_first_adjacent_tet() -> None:
    solid = Solid(name="diel", epsr=2.5, mur=1.3, sigma=0.02, tand=0.001, label=0)
    mesh = make_mesh(materials={0: solid})
    mesh.build_face_adjacency()
    assert mesh.boundary_face_material(6) is solid
    assert mesh.boundary_face_properties(6) == (2.5, 1.3)

    mesh.materials = {}
    assert mesh.boundary_face_material(6) is None
    assert mesh.boundary_face_properties(6) == (1.0, 1.0)


# ── one call for the whole pipeline ────────────────────────────────────────


def test_build_topology_fills_every_table() -> None:
    mesh = make_mesh()
    mesh.build_topology()
    np.testing.assert_array_equal(mesh.face_edges, EXPECTED_FACE_EDGES)
    assert [ids.tolist() for ids in mesh.face_adj_tet] == EXPECTED_FACE_ADJ
    assert mesh.tet_edges is not None and mesh.tet_faces is not None
