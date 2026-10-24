"""Tests for the TetGen mesh backend.

``Mesh3D.from_tetgen`` indexing is checked on synthetic ``TetGenMesh``
objects, and one test checks the generated ``WR90`` mesh's edge labels.

The library is optional tooling; tests that call it carry the ``tetgen``
marker and are skipped when ``build/libfetgen.*`` is absent (as in CI).
"""

from __future__ import annotations

import numpy as np
import pytest

from fes.mesh.mesh3d import MAX_LABEL, Mesh3D, mesh3d_from_poly
from fes.mesh.poly_header import Solid
from fes.mesh.tetgen_backend import (
    SWITCH_DEFAULT_3D,
    TetGenMesh,
)
from fes.paths import model_path


#: Marks the tests whose ``mesh`` fixture (or explicit call) needs the library.
needs_tetgen = pytest.mark.tetgen


@pytest.fixture(scope="module")
def mesh() -> Mesh3D:
    return mesh3d_from_poly(model_path("WR90"))


# ── the generated mesh ──────────────────────────────────────────────────────


@needs_tetgen
def test_edge_labels_stay_empty_as_in_the_reference(mesh: Mesh3D) -> None:
    # ``Createmesh`` only assigns ``edgLab`` on the 2D path; the dump carries an
    # empty vector for 3D and the 3D assembler never reads it.
    assert mesh.edge_labels.size == 0


# ── the switch string and index handling ───────────────────────────────────


def test_from_tetgen_offsets_by_first_number() -> None:
    """A 0-based ``tetgenio`` is stored unchanged; file input is shifted."""
    mesh = Mesh3D.from_tetgen(
        TetGenMesh(
            points=np.array(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
            ),
            edges=np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]),
            faces=np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]]),
            face_markers=np.ones(4, dtype=np.int64),
            tetrahedra=np.array([[0, 1, 2, 3]]),
            tetra_attributes=np.zeros(1),
            n_attributes=1,
            first_number=0,
            switches=SWITCH_DEFAULT_3D,
        )
    )
    np.testing.assert_array_equal(mesh.tetras, [[0, 1, 2, 3]])
    np.testing.assert_array_equal(mesh.edges[0], [0, 1])
    assert mesh.tet_labels.tolist() == [0]
    # [e12, e02, e01] on the sorted face (1, 2, 3).
    mesh.build_topology()
    np.testing.assert_array_equal(mesh.face_edges[0], [5, 4, 3])


def test_from_tetgen_defaults_labels_to_the_reference_max() -> None:
    mesh = TetGenMesh(
        points=np.zeros((4, 3)),
        edges=np.zeros((0, 2), dtype=np.int64),
        faces=np.zeros((0, 3), dtype=np.int64),
        face_markers=np.zeros(0, dtype=np.int64),
        tetrahedra=np.array([[1, 2, 3, 4]]),
        tetra_attributes=None,
        n_attributes=0,
        first_number=1,
        switches=SWITCH_DEFAULT_3D,
    )
    built = Mesh3D.from_tetgen(mesh)
    assert built.tetras.tolist() == [[0, 1, 2, 3]]
    assert built.tet_labels.tolist() == [MAX_LABEL]


def test_from_tetgen_reads_the_flat_attribute_array() -> None:
    """The reference implementation indexes ``tetrahedronattributelist[i]``, not ``i*n_attrs``."""
    mesh = TetGenMesh(
        points=np.zeros((6, 3)),
        edges=np.zeros((0, 2), dtype=np.int64),
        faces=np.zeros((0, 3), dtype=np.int64),
        face_markers=np.zeros(0, dtype=np.int64),
        tetrahedra=np.array([[1, 2, 3, 4], [1, 2, 3, 5]]),
        tetra_attributes=np.array([7.0, 8.0, 1.0, 2.0]),
        n_attributes=2,
        first_number=1,
        switches=SWITCH_DEFAULT_3D,
    )
    built = Mesh3D.from_tetgen(mesh)
    assert built.tet_labels.tolist() == [7, 8]


def test_unknown_material_falls_back_to_vacuum() -> None:
    mesh = TetGenMesh(
        points=np.zeros((4, 3)),
        edges=np.zeros((0, 2), dtype=np.int64),
        faces=np.zeros((0, 3), dtype=np.int64),
        face_markers=np.zeros(0, dtype=np.int64),
        tetrahedra=np.array([[1, 2, 3, 4]]),
        tetra_attributes=np.array([3.0]),
        n_attributes=1,
        first_number=1,
        switches=SWITCH_DEFAULT_3D,
    )
    built = Mesh3D.from_tetgen(
        mesh, materials={0: Solid(name="air", epsr=1.0, label=0)}, boundaries=[]
    )
    assert built.material_for_tet(0) is None
    assert built.material_properties(0) == (1.0, 1.0)
    assert built.boundaries == []
