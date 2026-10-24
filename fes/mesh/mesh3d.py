"""Three-dimensional tetrahedral meshes in the v0.1 C++ storage order.

The reference C++ 3D path hands the PLC to TetGen with ``pqAfeeQ``; the ``e`` and
``f`` switches make TetGen return *all* edges and *all* triangular faces, not
just the boundary ones.  ``preprocessing::Createmesh`` then stores them in the
order TetGen produced and derives two per-tetrahedron lookup tables:

``tetEdges``  local edges ``(0,1) (0,2) (0,3) (1,2) (1,3) (2,3)``
``tetFaces``  the face opposite local node 0, 1, 2, 3

Both tables are global ids into the TetGen-ordered ``edgNodes``/``facNodes``
arrays, so the global *numbering* — and therefore every finite-element DOF
number — is decided by the mesher, not by this module.  What this module
fixes is the local ordering and the lookup semantics exactly as
``Createmesh`` implements them, including ``std::map::operator[]`` inserting
an entry with value 0 when a key is missing.

Layout of :class:`Mesh3D`:

``nodes``        ``(n, 3)`` coordinates
``tetras``       ``(nt, 4)`` node indices, sorted ascending (``normalize_order``)
``tet_labels``   ``(nt,)`` regional attribute; selects the material
``edges``        ``(ne, 2)`` endpoints, sorted ascending
``edge_labels``  always empty: ``Createmesh`` never fills ``edgLab`` in 3D
``faces``        ``(nf, 3)`` node indices of every tet face, sorted ascending
``face_labels``  ``(nf,)`` boundary marker; the reference's ``maxLab`` for interior
``face_edges``   ``(nf, 3)`` local edge ids, built by :meth:`build_face_edges`
``tet_edges``    ``(nt, 6)`` local edge ids, built by :meth:`build_tet_connectivity`
``tet_faces``    ``(nt, 4)`` local face ids, built by :meth:`build_tet_connectivity`
``face_adj_tet`` ``(nf,)`` tet ids touching each face, ascending

``face_edges`` is where the 3D and 2D paths disagree.  ``CopyNewmesh`` (2D)
stores ``[e01, e02, e12]``, matching the order of the three edge functions in
``shape::hcurl``.  ``Createmesh`` (3D) stores ``[e12, e02, e01]`` instead: it
fills column 0 from ``facNodes(i,1),facNodes(i,2)``.  Since
``dof(prj, 2, id)`` binds the boundary-face vector basis to ``facEdges``
column by column, on a 3D boundary face basis function 0 (which ``shape.cpp``
builds along ``e01``) receives the global id of ``e12`` and vice versa.  The
quirk is reproduced rather than corrected.

As in :class:`~fes.mesh.mesh2d.Mesh2D`, materials are keyed by *line index*
inside ``#Solids`` (the C++ ``mtrl.label``), not by the region attribute.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .constants import MAX_LABEL
from .poly_header import Boundary, Solid, read_poly_metadata
from .tetgen_backend import SWITCH_DEFAULT_3D, TetGenMesh, tetrahedralize_poly

__all__ = ["LOCAL_EDGES", "LOCAL_FACES", "Mesh3D", "mesh3d_from_poly"]

#: Local edge order of ``tetEdges`` in ``pre_processor.cpp``.  It is also the
#: order of the six edge basis functions in ``shape.cpp`` and of the first six
#: scalar edge-interior DOFs in ``dof.cpp``.
LOCAL_EDGES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))

#: ``tetFaces`` column ``i`` is the face opposite local node ``i``.
LOCAL_FACES = ((1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2))

#: ``bc::bcTYPE`` values used by the reference solver.
_BC_TYPE_NAMES = {
    -1: "PerfectE",
    0: "PerfectH",
    1: "Radiation",
    2: "WavePort",
    3: "LumpedPort",
}


@dataclass
class Mesh3D:
    nodes: np.ndarray
    tetras: np.ndarray
    tet_labels: np.ndarray
    edges: np.ndarray
    edge_labels: np.ndarray
    faces: np.ndarray
    face_labels: np.ndarray
    face_edges: Optional[np.ndarray] = None
    tet_edges: Optional[np.ndarray] = None
    tet_faces: Optional[np.ndarray] = None
    face_adj_tet: Optional[List[np.ndarray]] = None
    materials: Dict[int, Solid] = field(default_factory=dict)
    boundaries: List[Boundary] = field(default_factory=list)
    name: str = ""

    # ── counts ──────────────────────────────────────────────────────────────
    @classmethod
    def from_oracle_dump(cls, payload: dict) -> "Mesh3D":
        """Build a mesh from a JSON payload of raw arrays.

        The arrays arrive already in reference order (including ``face_edges``
        and the per-tet tables), so they are stored verbatim, not rebuilt.
        """
        materials = {}
        for item in payload.get("tetmtrl", []):
            label = int(item["label"])
            materials[label] = Solid(
                name=str(item.get("sld_name", "")),
                epsr=float(item.get("epsr", 1.0)),
                mur=float(item.get("mur", 1.0)),
                sigma=float(item.get("sigma", 0.0)),
                tand=float(item.get("tand", 0.0)),
                material=str(item.get("name", "")),
                label=label,
            )
        boundaries = [
            Boundary(
                name=str(item["name"]),
                label=int(item["label"]),
                type=_BC_TYPE_NAMES.get(int(item["type"]), str(item["type"])),
                num_modes=int(item.get("num_modes", 0)),
            )
            for item in payload.get("facbc", [])
        ]
        edges = np.asarray(payload["edgNodes"], dtype=np.int64)
        # ``Createmesh`` never assigns ``edgLab`` on the 3D path, so the dump
        # carries an empty vector; it is stored as-is rather than zero-filled.
        edge_labels = np.asarray(payload.get("edgLab", []), dtype=np.int64)
        face_edges = payload.get("facEdges")
        tet_edges = payload.get("tetEdges")
        tet_faces = payload.get("tetFaces")
        adjacency = payload.get("facAdjTet")
        return cls(
            nodes=np.asarray(payload["nodPos"], dtype=float),
            tetras=np.asarray(payload["tetNodes"], dtype=np.int64),
            tet_labels=np.asarray(payload["tetLab"], dtype=np.int64),
            edges=edges,
            edge_labels=edge_labels,
            faces=np.asarray(payload["facNodes"], dtype=np.int64),
            face_labels=np.asarray(payload["facLab"], dtype=np.int64),
            face_edges=None if face_edges is None else np.asarray(face_edges, dtype=np.int64),
            tet_edges=None if tet_edges is None else np.asarray(tet_edges, dtype=np.int64),
            tet_faces=None if tet_faces is None else np.asarray(tet_faces, dtype=np.int64),
            face_adj_tet=(
                None
                if adjacency is None
                else [np.asarray(ids, dtype=np.int64) for ids in adjacency]
            ),
            materials=materials,
            boundaries=boundaries,
            name=str(payload.get("name", "")),
        )

    @classmethod
    def from_tetgen(
        cls,
        mesh: TetGenMesh,
        *,
        materials: Optional[Dict[int, Solid]] = None,
        boundaries: Optional[List[Boundary]] = None,
        name: str = "",
    ) -> "Mesh3D":
        """Store a TetGen result the way ``preprocessing::Createmesh`` does.

        TetGen numbers nodes from ``first_number`` (1 for ``.poly`` input) and
        ``Createmesh`` subtracts one; every connectivity array is offset here
        by the same amount.  Face markers are kept raw — unlike the 2D path,
        the 3D code does not shift them — and a tetrahedron's label is the
        *first* attribute, read as a flat array index exactly as the oracle
        reads ``tetrahedronattributelist[i]``.  When TetGen produced no
        attributes the oracle leaves ``tetLab`` at ``maxLab``.

        Derived tables are built afterwards by :meth:`build_topology`, which
        is where the oracle's sorting and lookup semantics live.
        """
        offset = int(mesh.first_number)
        edges = np.asarray(mesh.edges, dtype=np.int64)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        tetrahedra = np.asarray(mesh.tetrahedra, dtype=np.int64)
        if offset and edges.size:
            edges = edges - offset
        if offset and faces.size:
            faces = faces - offset
        if offset and tetrahedra.size:
            tetrahedra = tetrahedra - offset

        if mesh.tetra_attributes is None or mesh.n_attributes == 0:
            tet_labels = np.full(tetrahedra.shape[0], MAX_LABEL, dtype=np.int64)
        else:
            # The C++ assignment is ``tetLab(i) = tetrahedronattributelist[i]``:
            # index ``i``, not ``i * n_attributes``, and a double -> size_t cast.
            flat = np.asarray(mesh.tetra_attributes, dtype=float)
            tet_labels = flat[: tetrahedra.shape[0]].astype(np.int64)

        return cls(
            nodes=np.asarray(mesh.points, dtype=float),
            tetras=tetrahedra,
            tet_labels=tet_labels,
            edges=edges,
            edge_labels=np.zeros(0, dtype=np.int64),
            faces=faces,
            face_labels=np.asarray(mesh.face_markers, dtype=np.int64),
            materials=dict(materials or {}),
            boundaries=list(boundaries or []),
            name=name,
        )

    @property
    def dim(self) -> int:
        return 3

    @property
    def n_nodes(self) -> int:
        return int(self.nodes.shape[0])

    @property
    def n_tetras(self) -> int:
        return int(self.tetras.shape[0])

    @property
    def n_edges(self) -> int:
        return int(self.edges.shape[0])

    @property
    def n_faces(self) -> int:
        return int(self.faces.shape[0])

    # ── the oracle's canonical ordering ─────────────────────────────────────
    def normalize_order(self) -> None:
        """Sort the node indices inside every tetrahedron, edge and face.

        ``Createmesh`` sorts each tet row *before* it looks up the local edges
        and faces, so the local order is always the ascending node order.  It
        sorts edges and faces as it stores them; TetGen already emits sorted
        rows, but the sort is repeated here so callers cannot depend on it.
        """
        if self.tetras.size:
            self.tetras = np.sort(self.tetras, axis=1)
        if self.edges.size:
            self.edges = np.sort(self.edges, axis=1)
        if self.faces.size:
            self.faces = np.sort(self.faces, axis=1)

    # ── the oracle's per-tet lookup tables ──────────────────────────────────
    def build_tet_connectivity(self) -> None:
        """Fill ``tet_edges`` and ``tet_faces`` from the global edge/face lists.

        Reproduces ``preprocessing::Createmesh``: a dictionary keyed by the
        sorted node pair (or triple) is built from the global lists, later
        duplicates overwrite earlier ones, and a missing key yields 0 because
        that is what ``std::map::operator[]`` does in the oracle.

        Call :meth:`normalize_order` first; the local columns are defined on
        the sorted tetrahedron, exactly as the C++ code indexes it.
        """
        edge_map: Dict[tuple, int] = {}
        for edge_id, edge in enumerate(self.edges):
            a, b = int(edge[0]), int(edge[1])
            if a > b:
                a, b = b, a
            edge_map[(a, b)] = edge_id

        face_map: Dict[tuple, int] = {}
        for face_id, face in enumerate(self.faces):
            key = tuple(sorted(int(v) for v in face))
            face_map[key] = face_id

        tet_edges = np.zeros((self.n_tetras, 6), dtype=np.int64)
        tet_faces = np.zeros((self.n_tetras, 4), dtype=np.int64)
        for tet_id, tet in enumerate(self.tetras):
            local = [int(v) for v in tet]
            for column, (i, j) in enumerate(LOCAL_EDGES):
                a, b = local[i], local[j]
                if a > b:
                    a, b = b, a
                tet_edges[tet_id, column] = edge_map.get((a, b), 0)
            for column, (i, j, k) in enumerate(LOCAL_FACES):
                key = tuple(sorted((local[i], local[j], local[k])))
                tet_faces[tet_id, column] = face_map.get(key, 0)

        self.tet_edges = tet_edges
        self.tet_faces = tet_faces

    def build_face_edges(self) -> None:
        """Fill ``face_edges`` the way ``Createmesh`` does.

        Each row is ``[e12, e02, e01]``, i.e. the *reverse* of the 2D order and
        of the H(curl) basis order.  A missing key yields 0 because the oracle
        looks the pair up with ``std::map::operator[]``.  Call
        :meth:`normalize_order` first, as the oracle does.
        """
        edge_map: Dict[tuple, int] = {}
        for edge_id, edge in enumerate(self.edges):
            a, b = int(edge[0]), int(edge[1])
            if a > b:
                a, b = b, a
            edge_map[(a, b)] = edge_id

        face_edges = np.zeros((self.n_faces, 3), dtype=np.int64)
        for face_id, face in enumerate(self.faces):
            n0, n1, n2 = (int(face[0]), int(face[1]), int(face[2]))
            for column, (a, b) in enumerate(((n1, n2), (n0, n2), (n0, n1))):
                face_edges[face_id, column] = edge_map.get((a, b), 0)
        self.face_edges = face_edges

    def build_face_adjacency(self) -> None:
        """Fill ``face_adj_tet`` exactly as ``Createmesh`` appends it.

        Every tetrahedron contributes its id to the four faces in
        ``tet_faces``; ids therefore arrive in ascending tet order.  A tet
        whose lookup table repeats a face (the zero fallback of a missing key)
        is appended to that face twice, as in the oracle.
        """
        if self.tet_faces is None:
            raise ValueError(
                "mesh.tet_faces is empty; run build_tet_connectivity() first"
            )
        adjacency: List[List[int]] = [[] for _ in range(self.n_faces)]
        for tet_id in range(self.n_tetras):
            for column in range(4):
                face_id = int(self.tet_faces[tet_id, column])
                adjacency[face_id].append(tet_id)
        self.face_adj_tet = [np.asarray(ids, dtype=np.int64) for ids in adjacency]

    def build_topology(self) -> None:
        """Build every derived table the assembler needs, in oracle order."""
        self.normalize_order()
        self.build_tet_connectivity()
        self.build_face_edges()
        self.build_face_adjacency()

    # ── boundary-face geometry ──────────────────────────────────────────────
    def tet_geo(self, tet_id: int) -> np.ndarray:
        """Return the ``(4, 3)`` coordinates of one tetrahedron."""
        return np.asarray(self.nodes[np.asarray(self.tetras[tet_id])], dtype=float)

    def fac_geo(self, face_id: int) -> np.ndarray:
        """Return the ``(3, 3)`` coordinates of one face, in ``facNodes`` order."""
        return np.asarray(self.nodes[np.asarray(self.faces[face_id])], dtype=float)

    def int_node(self, face_id: int) -> np.ndarray:
        """Return the coordinates of the tet vertex across ``face_id``.

        ``mesh::int_node`` looks at the *first* adjacent tetrahedron and picks
        the vertex not on the face; if every vertex were on the face (which
        cannot happen for a real tetrahedron) it would keep the last one.  The
        fallback is reproduced so this function never raises mid-assembly.
        """
        return self.int_node_ref(face_id)[0]

    def int_node_ref(self, face_id: int, ref_face: int = 0) -> tuple:
        """Return ``(coordinates, RefFace)`` like the two-argument ``int_node``.

        The radiation post-processor uses this overload: ``RefFace`` selects
        the adjacent tetrahedron *before* the oracle overwrites it with the
        local index of the vertex across the face.  Callers pass 0, so the
        first adjacent tet is used, exactly as the coordinate-only variant.
        """
        if self.face_adj_tet is None:
            raise ValueError(
                "mesh.face_adj_tet is empty; run build_face_adjacency() first"
            )
        adjacent = self.face_adj_tet[face_id]
        if adjacent.size == 0:
            raise ValueError(f"face {face_id} has no adjacent tetrahedron")
        face_nodes = set(int(v) for v in self.faces[face_id])
        tet = self.tetras[int(adjacent[ref_face])]
        interior = int(tet[-1])
        found_index = int(ref_face)
        for index, node in enumerate(tet):
            if int(node) not in face_nodes:
                interior = int(node)
                found_index = index
                break
        return np.asarray(self.nodes[interior], dtype=float), found_index

    # ── boundary lookup ─────────────────────────────────────────────────────
    def faces_with_label(self, label: int) -> np.ndarray:
        """Return face ids whose boundary marker equals *label*, ascending.

        ``Createmesh`` stores the raw TetGen marker (``trifacemarkerlist[i]``,
        without the ``-1`` that the 2D path applies), so *label* is compared
        as-is against the ``#Boundaries`` entry.
        """
        return np.flatnonzero(np.asarray(self.face_labels) == label).astype(np.int64)

    def faces_for_boundary(self, boundary: Boundary) -> np.ndarray:
        """Return the face ids belonging to a ``#Boundaries`` entry."""
        return self.faces_with_label(int(boundary.label))

    def boundary_face_material(self, face_id: int) -> Optional[Solid]:
        """Return the material of the first tet behind a boundary face."""
        if self.face_adj_tet is None:
            raise ValueError(
                "mesh.face_adj_tet is empty; run build_face_adjacency() first"
            )
        adjacent = self.face_adj_tet[face_id]
        if adjacent.size == 0:
            raise ValueError(f"face {face_id} has no adjacent tetrahedron")
        return self.material_for_tet(int(adjacent[0]))

    def boundary_face_properties(self, face_id: int) -> tuple:
        """Return ``(epsr, mur)`` of the solid behind a boundary face."""
        material = self.boundary_face_material(face_id)
        if material is None:
            return 1.0, 1.0
        return float(material.epsr), float(material.mur)

    # ── material lookup ─────────────────────────────────────────────────────
    def material_for_tet(self, tet_id: int) -> Optional[Solid]:
        """Return the material the oracle would use, or ``None`` for vacuum."""
        return self.materials.get(int(self.tet_labels[tet_id]))

    def material_properties(self, tet_id: int) -> tuple:
        """Return ``(epsr, mur)`` with the oracle's defaults of 1.0."""
        material = self.material_for_tet(tet_id)
        if material is None:
            return 1.0, 1.0
        return float(material.epsr), float(material.mur)


def mesh3d_from_poly(filename: str | Path, switches: Optional[str] = None) -> Mesh3D:
    """Read *filename*, mesh it with TetGen, and return a :class:`Mesh3D`.

    *switches* follows the oracle's ``+poly`` grammar: TetGen always needs
    ``p`` and receives ``AfeeQ`` so it returns every face and edge; a user
    string is prefixed with ``p`` and suffixed with ``AfeeQ``, exactly as
    ``pre_processor.cpp`` assembles it.
    """
    path = Path(filename)
    metadata = read_poly_metadata(path)
    command = SWITCH_DEFAULT_3D if switches is None else f"p{switches}AfeeQ"
    result = tetrahedralize_poly(path, command)
    materials = {solid.label: solid for solid in metadata.solids}
    mesh = Mesh3D.from_tetgen(
        result,
        materials=materials,
        boundaries=metadata.boundaries,
        name=metadata.name,
    )
    mesh.build_topology()
    return mesh
