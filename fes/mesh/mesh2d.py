"""Two-dimensional meshes in the v0.1 C++ storage order.

The v0.1 solver stores only boundary segments when it reads Triangle's output;
all interior edges (and, because of an orientation quirk, several duplicates)
are created later by ``mesh::build_2d_edge_connectivity``.  That function is
reproduced here verbatim, because the global edge numbering determines the
finite-element DOF numbering and therefore every assembled matrix entry.

Layout of :class:`Mesh2D`:

``nodes``        ``(n, 3)`` coordinates (the z column is zero)
``faces``        ``(nt, 3)`` node indices, sorted ascending (``normalize_order``)
``face_labels``  ``(nt,)`` regional attribute; selects the material
``edges``        ``(ne, 2)`` endpoints, ordered as the C++ map stores them
``edge_labels``  ``(ne,)`` boundary marker; ``MAX_LABEL`` for derived edges
``face_edges``   ``(nt, 3)`` local edge ids for (0,1), (0,2), (1,2)

Materials are keyed by *line index* inside ``#Solids`` (the C++ ``mtrl.label``),
not by the region attribute found in ``#Regions``.  Use
:meth:`Mesh2D.material_for_face` to reproduce the reference's lookup.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .constants import MAX_LABEL
from .geometry import read_poly_plc
from .poly_header import Boundary, Solid, read_poly_metadata
from .triangle_backend import TriangleMesh, triangulate_plc

__all__ = ["MAX_LABEL", "Mesh2D", "mesh_from_poly"]


@dataclass
class Mesh2D:
    nodes: np.ndarray
    faces: np.ndarray
    face_labels: np.ndarray
    edges: np.ndarray
    edge_labels: np.ndarray
    face_edges: Optional[np.ndarray] = None
    materials: Dict[int, Solid] = field(default_factory=dict)
    boundaries: List[Boundary] = field(default_factory=list)
    name: str = ""
    #: Number of edges that came from the input PLC (the oracle's ``Edges``
    #: count before ``build_2d_edge_connectivity`` adds interior edges).
    n_boundary_edges: int = 0
    #: ``True`` when :meth:`build_2d_edge_connectivity` used the v0.1
    #: unsorted-pair quirk; ``None`` before the first call.
    edge_connectivity_legacy: Optional[bool] = None

    # ── counts ──────────────────────────────────────────────────────────────
    @property
    def dim(self) -> int:
        return 2

    @property
    def n_nodes(self) -> int:
        return int(self.nodes.shape[0])

    @property
    def n_faces(self) -> int:
        return int(self.faces.shape[0])

    @property
    def n_edges(self) -> int:
        return int(self.edges.shape[0])

    @property
    def n_tetras(self) -> int:
        return 0

    # ── construction ────────────────────────────────────────────────────────
    @classmethod
    def from_triangle(
        cls,
        triangulation: TriangleMesh,
        *,
        materials: Optional[Dict[int, Solid]] = None,
        boundaries: Optional[List[Boundary]] = None,
        name: str = "",
    ) -> "Mesh2D":
        """Store a Triangle result the way ``triangulatemesh2D`` does."""
        points = np.asarray(triangulation.points, dtype=float)
        nodes = np.zeros((points.shape[0], 3))
        nodes[:, :2] = points

        faces = np.asarray(triangulation.triangles, dtype=np.int64)
        face_labels = np.asarray(triangulation.triangle_attributes).astype(np.int64)

        raw_edges = np.asarray(triangulation.segments, dtype=np.int64)
        if raw_edges.size:
            edges = raw_edges - 1
        else:
            edges = np.zeros((0, 2), dtype=np.int64)
        if triangulation.segment_markers is not None:
            edge_labels = np.asarray(
                triangulation.segment_markers, dtype=np.uint64
            ).astype(np.uint64, copy=True)
        else:
            edge_labels = np.zeros(edges.shape[0], dtype=np.uint64)

        return cls(
            nodes=nodes,
            faces=faces,
            face_labels=face_labels,
            edges=edges,
            edge_labels=edge_labels,
            materials=dict(materials or {}),
            boundaries=list(boundaries or []),
            name=name,
            n_boundary_edges=int(edges.shape[0]),
        )

    # ── the oracle's canonical ordering ─────────────────────────────────────
    def normalize_order(self) -> None:
        """Sort node indices inside every edge and face.

        ``project::project`` calls this after preprocessing and before
        assembly, so the edge connectivity is built from *sorted* triangles.
        That ordering is observable: it decides which interior edges are
        duplicates and therefore the global DOF numbering.
        """
        if self.edges.size:
            self.edges = np.sort(self.edges, axis=1)
        if self.faces.size:
            self.faces = np.sort(self.faces, axis=1)

    # ── homogeneous refinement (``mesh::refine_homogeneous``) ───────────────
    def refine_homogeneous(self) -> None:
        """Subdivide every triangle into four, preserving markers.

        This is ``mesh::refine_homogeneous``'s 2D branch.  It runs before
        :meth:`normalize_order` in the oracle, so the raw Triangle vertex
        order decides which mid-edge node gets which index: faces are visited
        in storage order, each visiting edges ``(0,1), (1,2), (2,0)`` with the
        key sorted, and a new node number is handed out on first encounter.
        Callers must therefore refine *before* sorting (``mesh_from_poly``
        does), and must not assume the new triangles are ordered.

        Only the PLC segments in ``edges[:n_boundary_edges]`` are split; the
        interior-edge connectivity is rebuilt by
        :meth:`build_2d_edge_connectivity` afterwards, exactly as the oracle
        runs it inside the assembler.
        """
        if self.n_faces == 0:
            return

        mid: Dict[tuple, int] = {}
        for face in self.faces:
            for ei in range(3):
                a = int(face[ei])
                b = int(face[(ei + 1) % 3])
                key = (a, b) if a <= b else (b, a)
                if key not in mid:
                    mid[key] = self.n_nodes + len(mid)

        new_nodes = np.zeros((self.n_nodes + len(mid), 3))
        new_nodes[: self.n_nodes] = self.nodes
        for (a, b), index in mid.items():
            new_nodes[index] = (self.nodes[a] + self.nodes[b]) * 0.5

        new_faces = np.zeros((self.n_faces * 4, 3), dtype=np.int64)
        new_face_labels = np.zeros(self.n_faces * 4, dtype=self.face_labels.dtype)
        for f, face in enumerate(self.faces):
            n0, n1, n2 = (int(face[0]), int(face[1]), int(face[2]))
            m01 = mid[(min(n0, n1), max(n0, n1))]
            m12 = mid[(min(n1, n2), max(n1, n2))]
            m20 = mid[(min(n2, n0), max(n2, n0))]
            f4 = f * 4
            new_faces[f4] = (n0, m01, m20)
            new_faces[f4 + 1] = (n1, m12, m01)
            new_faces[f4 + 2] = (n2, m20, m12)
            new_faces[f4 + 3] = (m01, m12, m20)
            new_face_labels[f4 : f4 + 4] = self.face_labels[f]

        old_boundary_edges = self.n_boundary_edges
        new_edges = np.zeros((old_boundary_edges * 2, 2), dtype=np.int64)
        new_edge_labels = np.zeros(
            old_boundary_edges * 2, dtype=self.edge_labels.dtype
        )
        for e in range(old_boundary_edges):
            n0 = int(self.edges[e, 0])
            n1 = int(self.edges[e, 1])
            m = mid[(min(n0, n1), max(n0, n1))]
            new_edges[2 * e] = (n0, m)
            new_edges[2 * e + 1] = (m, n1)
            new_edge_labels[2 * e] = self.edge_labels[e]
            new_edge_labels[2 * e + 1] = self.edge_labels[e]

        self.nodes = new_nodes
        self.faces = new_faces
        self.face_labels = new_face_labels
        self.edges = new_edges
        self.edge_labels = new_edge_labels
        self.n_boundary_edges = int(new_edges.shape[0])
        # The oracle resets facEdges/edgAdjFac here; the assembler rebuilds
        # them from the refined PLC.
        self.face_edges = None
        self.edge_connectivity_legacy = None

    # ── the oracle's connectivity algorithm ─────────────────────────────────
    def build_2d_edge_connectivity(self, legacy_unsorted_pairs: bool = False) -> None:
        """Build the full 2D edge connectivity used by the DOF map.

        ``legacy_unsorted_pairs=True`` reproduces ``mesh::build_2d_edge_connectivity``
        from v0.1 *exactly*, including a bug: interior edges are inserted with
        their unsorted endpoint pairs while ``facEdges`` looks them up sorted.
        Because :meth:`normalize_order` sorts every face ascending, the lookup
        of ``(n0, n2)`` usually misses and ``std::map::operator[]`` silently
        returns edge 0, collapsing a DOF and leaving an orphan row behind.
        That makes the assembled system singular and every port S-parameter
        collapse to ``-I`` (or a huge unphysical value).

        The default (``False``) sorts the endpoint pair before inserting, which
        is what the v0.1 code clearly intended (the ``std::swap`` is commented
        out next to the comment "each edge pair is individually sorted").  It
        produces one global edge per geometric edge; callers that need
        bug-for-bug parity with the frozen oracle pass ``True``.

        Calling it again with a different flag rebuilds from the original PLC
        edges, so the same mesh can be assembled either way.
        """
        if self.face_edges is not None:
            if self.edge_connectivity_legacy == legacy_unsorted_pairs:
                return
            # Drop the derived edges and start over from the PLC segments.
            keep = self.n_boundary_edges
            self.edges = self.edges[:keep].copy()
            self.edge_labels = self.edge_labels[:keep].copy()
            self.face_edges = None
        if self.n_faces == 0:
            self.edge_connectivity_legacy = legacy_unsorted_pairs
            return

        original_edges = self.n_edges
        emap: Dict[tuple, int] = {}
        for edge_id in range(original_edges):
            a = int(self.edges[edge_id, 0])
            b = int(self.edges[edge_id, 1])
            if a > b:
                a, b = b, a
            emap[(a, b)] = edge_id

        next_edge = original_edges
        for face in self.faces:
            n0, n1, n2 = (int(face[0]), int(face[1]), int(face[2]))
            for a, b in ((n1, n2), (n2, n0), (n0, n1)):
                key = (a, b) if legacy_unsorted_pairs else (min(a, b), max(a, b))
                if key not in emap:
                    emap[key] = next_edge
                    next_edge += 1

        total_edges = next_edge
        edges = np.zeros((total_edges, 2), dtype=np.int64)
        labels = np.zeros(total_edges, dtype=np.uint64)
        if original_edges:
            labels[:original_edges] = self.edge_labels[:original_edges]

        # std::map iterates in ascending key order (a, b).
        for (a, b) in sorted(emap):
            index = emap[(a, b)]
            edges[index, 0] = a
            edges[index, 1] = b
            if index >= original_edges:
                labels[index] = MAX_LABEL

        face_edges = np.zeros((self.n_faces, 3), dtype=np.int64)
        for face_id, face in enumerate(self.faces):
            n0, n1, n2 = (int(face[0]), int(face[1]), int(face[2]))
            for column, key in enumerate(((n0, n1), (n0, n2), (n1, n2))):
                # operator[] inserts a zero-valued entry for a missing key.
                face_edges[face_id, column] = emap.setdefault(key, 0)

        self.edges = edges
        self.edge_labels = labels
        self.face_edges = face_edges
        self.edge_connectivity_legacy = legacy_unsorted_pairs

    # ── material lookup ─────────────────────────────────────────────────────
    def material_for_face(self, face_id: int) -> Optional[Solid]:
        """Return the material the oracle would use, or ``None`` for vacuum."""
        return self.materials.get(int(self.face_labels[face_id]))

    def material_properties(self, face_id: int) -> tuple:
        """Return ``(epsr, mur)`` with the oracle's defaults of 1.0."""
        material = self.material_for_face(face_id)
        if material is None:
            return 1.0, 1.0
        return float(material.epsr), float(material.mur)


def mesh_from_poly(
    filename: str | Path,
    switches: Optional[str] = None,
    *,
    h_ord: int = 0,
    legacy_edge_connectivity: bool = False,
) -> Mesh2D:
    """Read *filename*, mesh it with Triangle, and return a :class:`Mesh2D`.

    The result matches the oracle after preprocessing and ``normalize_order``:
    raw Triangle connectivity, sorted face/edge node indices, and the full 2D
    edge connectivity built from it.  ``h_ord`` applies that many uniform
    refinement levels (``project::project``) before the sort, because the
    oracle's mid-edge numbering depends on the raw Triangle vertex order.
    Pass ``legacy_edge_connectivity=True`` to reproduce the v0.1 unsorted-pair
    bug (see :meth:`Mesh2D.build_2d_edge_connectivity`).
    """
    path = Path(filename)
    metadata = read_poly_metadata(path)
    geometry = read_poly_plc(path)
    triangulation = triangulate_plc(geometry, switches=switches)
    materials = {solid.label: solid for solid in metadata.solids}
    mesh = Mesh2D.from_triangle(
        triangulation,
        materials=materials,
        boundaries=metadata.boundaries,
        name=metadata.name,
    )
    for _ in range(h_ord):
        mesh.refine_homogeneous()
    mesh.normalize_order()
    mesh.build_2d_edge_connectivity(legacy_unsorted_pairs=legacy_edge_connectivity)
    return mesh
