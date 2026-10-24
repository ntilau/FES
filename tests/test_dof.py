"""Tests for :mod:`fes.core.dof` (the DOF numbering).

The numbering is hierarchical and global: nodes first, then edge-interior
unknowns, then face-interior unknowns.  These tests pin the layout and the
per-element maps on a hand-built mesh, where every number can be written out.
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
from fes.mesh.mesh2d import Mesh2D


def _two_triangles() -> Mesh2D:
    """A unit square split into two triangles (``nN=4``, ``nE=5``, ``nF=2``)."""
    nodes = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]
    )
    mesh = Mesh2D(
        nodes=nodes,
        faces=np.array([[0, 1, 2], [1, 2, 3]]),
        face_labels=np.ones(2, dtype=np.int64),
        edges=np.array([[0, 1], [1, 3], [2, 3], [0, 2]]),
        edge_labels=np.zeros(4, dtype=np.uint64),
        n_boundary_edges=4,
    )
    mesh.normalize_order()
    mesh.build_2d_edge_connectivity()
    return mesh


@pytest.mark.parametrize(
    "order, expected",
    [(1, 4), (2, 4 + 5), (3, 4 + 2 * 5 + 2), (4, 4 + 3 * 5 + 3 * 2)],
)
def test_scalar_dof_count(order: int, expected: int) -> None:
    assert scalar_dof_count(_two_triangles(), order) == expected


@pytest.mark.parametrize(
    "order, expected",
    [(1, 5), (2, 2 * (5 + 2)), (3, 3 * 5 + 6 * 2), (4, 4 * 5 + 12 * 2)],
)
def test_vector_dof_count(order: int, expected: int) -> None:
    assert vector_dof_count(_two_triangles(), order) == expected


def test_scalar_map_starts_with_the_vertices() -> None:
    mesh = _two_triangles()
    for order in (1, 2, 3, 4):
        for face_id, face in enumerate(mesh.faces):
            dofs = scalar_dof_map(mesh, order, face_id)
            assert dofs[:3].tolist() == face.tolist()
            assert len(np.unique(dofs)) == len(dofs)


def test_scalar_map_layout() -> None:
    mesh = _two_triangles()
    nN, nE, nF = mesh.n_nodes, mesh.n_edges, mesh.n_faces
    for face_id in range(nF):
        edges = mesh.face_edges[face_id]
        assert scalar_dof_map(mesh, 2, face_id).tolist() == (
            [*mesh.faces[face_id], *(nN + edges)]
        )
        assert scalar_dof_map(mesh, 3, face_id).tolist() == (
            [*mesh.faces[face_id], *(nN + edges), *(nN + nE + edges),
             nN + 2 * nE + face_id]
        )
        assert scalar_dof_map(mesh, 4, face_id).tolist() == (
            [*mesh.faces[face_id], *(nN + edges), *(nN + nE + edges),
             *(nN + 2 * nE + edges),
             nN + 3 * nE + face_id,
             nN + 3 * nE + nF + face_id,
             nN + 3 * nE + 2 * nF + face_id]
        )


def test_scalar_maps_cover_every_dof() -> None:
    """Every allocated DOF is referenced; shared vertices appear once per face."""
    mesh = _two_triangles()
    for order in (1, 2, 3, 4):
        used = np.concatenate(
            [scalar_dof_map(mesh, order, f) for f in range(mesh.n_faces)]
        )
        assert set(used.tolist()) == set(range(scalar_dof_count(mesh, order)))


def test_vector_map_layout() -> None:
    mesh = _two_triangles()
    nE, nF = mesh.n_edges, mesh.n_faces
    for face_id in range(nF):
        edges = mesh.face_edges[face_id]
        assert vector_dof_map(mesh, 1, face_id).tolist() == edges.tolist()
        assert vector_dof_map(mesh, 2, face_id).tolist() == [
            *edges, *(nE + edges),
            2 * nE + face_id, 2 * nE + nF + face_id,
        ]
        assert len(vector_dof_map(mesh, 3, face_id)) == 15
        assert len(vector_dof_map(mesh, 4, face_id)) == 24


def test_face_map_is_the_2d_element_map() -> None:
    """In 2D the element *is* the face, so both spellings must agree."""
    mesh = _two_triangles()
    for order in (1, 2, 3, 4):
        for face_id in range(mesh.n_faces):
            face = face_dof_map(mesh, order, face_id)
            np.testing.assert_array_equal(face.s, scalar_dof_map(mesh, order, face_id))
            np.testing.assert_array_equal(face.v, vector_dof_map(mesh, order, face_id))


@pytest.mark.parametrize("order", [0, 5, -1])
def test_unsupported_orders_raise(order: int) -> None:
    mesh = _two_triangles()
    with pytest.raises(ValueError):
        scalar_dof_count(mesh, order)
    with pytest.raises(ValueError):
        vector_dof_count(mesh, order)
    with pytest.raises(ValueError):
        scalar_dof_map(mesh, order, 0)
    with pytest.raises(ValueError):
        vector_dof_map(mesh, order, 0)
