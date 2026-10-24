"""Global degree-of-freedom numbering, transcribed from ``cpp/src/dof.cpp``.

Two things in the reference C++ decide a DOF number: the *layout* (``dof(project*)``)
and the *per-element map* (``dof(project*, dim, id)``).  Both are reproduced
here literally, index by index, because the assembled matrices are only
entry-for-entry identical if the numbering is.

Layout for a 2D mesh with ``nN`` nodes, ``nE`` edges and ``nF`` faces:

===================  ============================================  =========
order                scalar DOFs                                   vector
===================  ============================================  =========
1                    ``nN``                                        ``nE``
2                    ``nN + nE``                                   ``2(nE+nF)``
3                    ``nN + 2 nE + nF``                            ``3 nE + 6 nF``
4                    ``nN + 3 nE + 3 nF``                          ``4 nE + 12 nF``
===================  ============================================  =========

For a 3D mesh the same expressions apply, with ``nE`` the number of *all*
edges and ``nF`` the number of *all* faces (TetGen is run with ``-e -f``), plus
the interior unknowns of each tetrahedron at the highest order:

===================  ============================================  =========
order                scalar DOFs                                   vector
===================  ============================================  =========
1                    ``nN``                                        ``nE``
2                    ``nN + nE``                                   ``2(nE+nF)``
3                    ``nN + 2 nE + nF``                            ``3(nE+nT) + 6 nF``
4                    ``nN + 3 nE + 3 nF + nT``                     ``4 nE + 12 nF + 12 nT``
===================  ============================================  =========

Both families are *hierarchical*: the scalar block starts with the vertex
values and adds edge-interior then face-interior unknowns, while the vector
block starts with the lowest-order edge functions and adds the next layer on top.
``edge`` here means the entry of ``mesh.face_edges`` for that face (2D) or
``mesh.tet_edges`` for that tetrahedron (3D), so the numbers depend on the
global edge numbering the mesher produced.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..mesh.mesh2d import Mesh2D
    from ..mesh.mesh3d import Mesh3D

__all__ = [
    "MAX_ORDER",
    "FaceDOF",
    "face_dof_map",
    "scalar_dof_count",
    "scalar_dof_map",
    "vector_dof_count",
    "vector_dof_map",
]

#: Highest order ``dof.cpp`` implements (both the 2D and the 3D paths).
MAX_ORDER = 4


def _checked_order(order: int) -> int:
    order = int(order)
    if order < 1 or order > MAX_ORDER:
        raise ValueError(f"dof mapping order {order} not yet implemented")
    return order


def _is_3d(mesh) -> bool:
    return getattr(mesh, "dim", 2) == 3


class FaceDOF(NamedTuple):
    """The scalar and vector DOF numbers of one triangular face.

    Mirrors the two vectors a C++ ``dof(prj, 2, id)`` object carries.  On a 2D
    mesh *id* is a triangle (the face *is* the element); on a 3D mesh it is a
    boundary face and the numbers refer to the triangle's edge and interior
    unknowns inside the 3D global numbering.
    """

    s: np.ndarray
    v: np.ndarray


def _face_edges(mesh: "Mesh2D | Mesh3D", face_id: int) -> np.ndarray:
    if getattr(mesh, "face_edges", None) is None:
        builder = "build_face_edges()" if _is_3d(mesh) else "build_2d_edge_connectivity()"
        raise ValueError(f"mesh.face_edges is empty; run {builder} first")
    return np.asarray(mesh.face_edges[face_id], dtype=np.int64)


def _tet_edges(mesh: "Mesh3D", tet_id: int) -> np.ndarray:
    if mesh.tet_edges is None:
        raise ValueError(
            "mesh.tet_edges is empty; run build_tet_connectivity() first"
        )
    return np.asarray(mesh.tet_edges[tet_id], dtype=np.int64)


def _tet_faces(mesh: "Mesh3D", tet_id: int) -> np.ndarray:
    if mesh.tet_faces is None:
        raise ValueError(
            "mesh.tet_faces is empty; run build_tet_connectivity() first"
        )
    return np.asarray(mesh.tet_faces[tet_id], dtype=np.int64)


# ── scalar (nodal / H1) ─────────────────────────────────────────────────────


def scalar_dof_count(mesh: "Mesh2D | Mesh3D", order: int) -> int:
    """Return ``dof(project*)``'s ``dofnums`` for a 2D or 3D mesh."""
    order = _checked_order(order)
    n_nodes, n_edges, n_faces = mesh.n_nodes, mesh.n_edges, mesh.n_faces
    n_tetras = mesh.n_tetras  # 0 in 2D
    if order == 1:
        return n_nodes
    if order == 2:
        return n_nodes + n_edges
    if order == 3:
        return n_nodes + 2 * n_edges + n_faces
    return n_nodes + 3 * n_edges + 3 * n_faces + n_tetras


def scalar_dof_map(mesh: "Mesh2D | Mesh3D", order: int, element_id: int) -> np.ndarray:
    """Return the scalar DOF numbers of one element.

    *element_id* is a triangle id for a 2D mesh and a tetrahedron id for a 3D
    one.  The length is 3/6/10/15 (2D) or 4/10/20/35 (3D), matching the
    columns of :func:`fes.core.shape.shape_2d` and
    :func:`fes.core.shape3d.shape_3d`.

    For scalar DOFs on a *boundary face* of a 3D mesh use
    :func:`face_dof_map`: ``dof.cpp``'s ``case 2`` block is shared with the 2D
    element path.
    """
    order = _checked_order(order)
    if _is_3d(mesh):
        return _scalar_dof_map_tet(mesh, order, element_id)
    return face_dof_map(mesh, order, element_id).s


def face_dof_map(mesh: "Mesh2D | Mesh3D", order: int, face_id: int) -> FaceDOF:
    """Return ``dof(prj, 2, face_id)``: the scalar and vector blocks of a face.

    The oracle uses this branch both for every element of a 2D mesh and for
    every boundary face of a 3D one.  The face node and edge tables differ
    between those paths only in how the mesh was built — in particular the 3D
    ``face_edges`` columns are reversed, so the vector block permutes the
    three edge functions here rather than at the call site.
    """
    order = _checked_order(order)
    edges = _face_edges(mesh, face_id)
    nodes = np.asarray(mesh.faces[face_id], dtype=np.int64)
    n_nodes, n_edges, n_faces = mesh.n_nodes, mesh.n_edges, mesh.n_faces

    if order == 1:
        s = nodes.copy()
        v = edges.copy()
        return FaceDOF(s=s, v=v)

    if order == 2:
        s = np.concatenate([nodes, n_nodes + edges])
        v = np.empty(8, dtype=np.int64)
        v[0:3] = edges
        v[3:6] = n_edges + edges
        v[6] = 2 * n_edges + face_id
        v[7] = 2 * n_edges + n_faces + face_id
        return FaceDOF(s=s, v=v)

    if order == 3:
        s = np.concatenate(
            [
                nodes,
                n_nodes + edges,
                n_nodes + n_edges + edges,
                [n_nodes + 2 * n_edges + face_id],
            ]
        )
        v = np.empty(15, dtype=np.int64)
        v[0:3] = edges
        v[3:6] = n_edges + edges
        v[6] = 2 * n_edges + face_id
        v[7] = 2 * n_edges + n_faces + face_id
        v[8:11] = 2 * n_edges + 2 * n_faces + edges
        v[11] = 3 * n_edges + 2 * n_faces + face_id
        v[12] = 3 * n_edges + 3 * n_faces + face_id
        v[13] = 3 * n_edges + 4 * n_faces + face_id
        v[14] = 3 * n_edges + 5 * n_faces + face_id
        return FaceDOF(s=s, v=v)

    s = np.concatenate(
        [
            nodes,
            n_nodes + edges,
            n_nodes + n_edges + edges,
            n_nodes + 2 * n_edges + edges,
            [
                n_nodes + 3 * n_edges + face_id,
                n_nodes + 3 * n_edges + n_faces + face_id,
                n_nodes + 3 * n_edges + 2 * n_faces + face_id,
            ],
        ]
    )
    v = np.empty(24, dtype=np.int64)
    v[0:3] = edges
    v[3:6] = n_edges + edges
    v[6] = 2 * n_edges + face_id
    v[7] = 2 * n_edges + n_faces + face_id
    v[8:11] = 2 * n_edges + 2 * n_faces + edges
    v[11] = 3 * n_edges + 2 * n_faces + face_id
    v[12] = 3 * n_edges + 3 * n_faces + face_id
    v[13] = 3 * n_edges + 4 * n_faces + face_id
    v[14] = 3 * n_edges + 5 * n_faces + face_id
    v[15:18] = 3 * n_edges + 6 * n_faces + edges
    v[18] = 4 * n_edges + 6 * n_faces + face_id
    v[19] = 4 * n_edges + 7 * n_faces + face_id
    v[20] = 4 * n_edges + 8 * n_faces + face_id
    v[21] = 4 * n_edges + 9 * n_faces + face_id
    v[22] = 4 * n_edges + 10 * n_faces + face_id
    v[23] = 4 * n_edges + 11 * n_faces + face_id
    return FaceDOF(s=s, v=v)


def _scalar_dof_map_tet(mesh: "Mesh3D", order: int, tet_id: int) -> np.ndarray:
    edges = _tet_edges(mesh, tet_id)
    faces = _tet_faces(mesh, tet_id)
    nodes = np.asarray(mesh.tetras[tet_id], dtype=np.int64)
    n_nodes, n_edges, n_faces = mesh.n_nodes, mesh.n_edges, mesh.n_faces

    if order == 1:
        return nodes.copy()
    if order == 2:
        return np.concatenate([nodes, n_nodes + edges])
    if order == 3:
        return np.concatenate(
            [
                nodes,
                n_nodes + edges,
                n_nodes + n_edges + edges,
                n_nodes + 2 * n_edges + faces,
            ]
        )
    return np.concatenate(
        [
            nodes,
            n_nodes + edges,
            n_nodes + n_edges + edges,
            n_nodes + 2 * n_edges + edges,
            n_nodes + 3 * n_edges + faces,
            n_nodes + 3 * n_edges + n_faces + faces,
            n_nodes + 3 * n_edges + 2 * n_faces + faces,
            [n_nodes + 3 * n_edges + 3 * n_faces + tet_id],
        ]
    )


# ── vector (H(curl)) ────────────────────────────────────────────────────────


def vector_dof_count(mesh: "Mesh2D | Mesh3D", order: int) -> int:
    """Return ``dof(project*)``'s ``dofnumv`` for a 2D or 3D mesh."""
    order = _checked_order(order)
    n_edges, n_faces = mesh.n_edges, mesh.n_faces
    n_tetras = mesh.n_tetras  # 0 in 2D; kept for fidelity with the oracle
    if order == 1:
        return n_edges
    if order == 2:
        return 2 * (n_edges + n_faces)
    if order == 3:
        return 3 * (n_edges + n_tetras) + 6 * n_faces
    return 4 * n_edges + 12 * n_faces + 12 * n_tetras


def vector_dof_map(mesh: "Mesh2D | Mesh3D", order: int, element_id: int) -> np.ndarray:
    """Return the H(curl) DOF numbers of one element.

    *element_id* is a triangle id for a 2D mesh and a tetrahedron id for a 3D
    one.  The length is 3/8/15/24 (2D) or 6/20/45/84 (3D), matching the
    columns of :func:`fes.core.shape.shape_2d`'s and
    :func:`fes.core.shape3d.shape_3d`'s vector bases.

    For a *boundary face* of a 3D mesh use :func:`face_dof_map`.
    """
    order = _checked_order(order)
    if _is_3d(mesh):
        return _vector_dof_map_tet(mesh, order, element_id)
    return face_dof_map(mesh, order, element_id).v


def _vector_dof_map_tet(mesh: "Mesh3D", order: int, tet_id: int) -> np.ndarray:
    edges = _tet_edges(mesh, tet_id)
    faces = _tet_faces(mesh, tet_id)
    n_edges, n_faces, n_tetras = mesh.n_edges, mesh.n_faces, mesh.n_tetras

    size = {1: 6, 2: 20, 3: 45, 4: 84}[order]
    v = np.empty(size, dtype=np.int64)

    if order == 1:
        return edges.copy()

    v[0:6] = edges
    v[6:12] = n_edges + edges
    for i in range(4):
        v[12 + 2 * i] = 2 * n_edges + faces[i]
        v[13 + 2 * i] = 2 * n_edges + n_faces + faces[i]

    if order == 2:
        return v

    v[20:26] = 2 * n_edges + 2 * n_faces + edges
    for i in range(4):
        v[26 + 4 * i] = 3 * n_edges + 2 * n_faces + faces[i]
        v[27 + 4 * i] = 3 * n_edges + 3 * n_faces + faces[i]
        v[28 + 4 * i] = 3 * n_edges + 4 * n_faces + faces[i]
        v[29 + 4 * i] = 3 * n_edges + 5 * n_faces + faces[i]
    v[42] = 3 * n_edges + 6 * n_faces + tet_id
    v[43] = 3 * n_edges + 6 * n_faces + n_tetras + tet_id
    v[44] = 3 * n_edges + 6 * n_faces + 2 * n_tetras + tet_id

    if order == 3:
        return v

    base = 3 * n_edges + 6 * n_faces + 3 * n_tetras
    v[45:51] = base + edges
    for i in range(4):
        for k in range(6):
            v[51 + 6 * i + k] = base + n_edges + k * n_faces + faces[i]
    vol_base = base + n_edges + 6 * n_faces
    for k in range(9):
        v[75 + k] = vol_base + k * n_tetras + tet_id
    return v
