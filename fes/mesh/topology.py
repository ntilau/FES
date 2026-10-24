"""Legacy mesh arrays: the ``.h1.mat`` layout that ``IOrMesh`` wrote.

``py/iormesh/IOrMesh`` was a C wrapper around Triangle.  It ran the command
line binary (``triangle -p<args>DeQ``), read the ``.1.node/.1.ele/.1.edge``
files back, sorted every element and edge, and wrote seven arrays into a
MATLAB file::

    node   (n, 2)   coordinates
    nlab   (n,)     node markers
    ele    (nt, 3)  node indices, each row sorted ascending
    elab   (nt,)    regional attribute
    spig   (nt, 3)  signed 1-based edge id per element, in (1,2),(0,2),(0,1)
                    column order
    spig2  (ne, 2)  edge endpoints, sorted ascending
    slab   (ne,)    edge (boundary) markers

This module rebuilds those arrays directly from a :class:`TriangleMesh`, so
the C tool is no longer needed.  Triangle is still invoked through the same
shared library the solver uses; only the file round-trip and the ``sort`` /
``buildEle2Edge`` post-processing lived in C.

The ``spig`` sign convention looks arbitrary because it is: ``IOrMesh`` sorts
the vertices of each element and each edge, then derives the sign from the
order in which the *sorted* pair appears.  For sorted input the middle column
(edge ``(e0, e2)``) is always negative and the other two are positive.  The
reproduction below keeps the general orientation logic so that unsorted edges
produce the same signs ``IOrMesh`` would have produced.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .geometry import read_poly_plc
from .io_mat import write_cache
from .triangle_backend import TriangleMesh, triangulate_plc

__all__ = [
    "build_ele2edge",
    "derive_edges",
    "legacy_arrays",
    "mesh_arrays_from_poly",
    "write_model_cache",
]

def derive_edges(triangles: np.ndarray) -> np.ndarray:
    """Unique edges of *triangles*, in first-appearance order (0-based)."""
    triangles = np.asarray(triangles, dtype=np.int64)
    seen = set()
    edges = []
    for n0, n1, n2 in triangles:
        for a, b in ((n0, n1), (n1, n2), (n2, n0)):
            key = (int(a), int(b)) if a < b else (int(b), int(a))
            if key not in seen:
                seen.add(key)
                edges.append(key)
    return np.asarray(edges, dtype=np.int64).reshape(-1, 2)


def build_ele2edge(elements: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Signed 1-based element→edge map, as ``IOrMesh::buildEle2Edge``.

    ``elements`` rows must be sorted ascending (the tool sorts them first) and
    ``edges`` must be sorted ascending per row.  Columns hold the edges
    ``(n1, n2)``, ``(n0, n2)`` and ``(n0, n1)`` respectively; a negative sign
    marks the ``(n0, n2)`` column of a sorted element, which is what the C
    scanner produced.  Edges missing from *edges* leave a zero, matching the
    tool's zero-initialised output.
    """
    elements = np.asarray(elements, dtype=np.int64)
    edges = np.sort(np.asarray(edges, dtype=np.int64), axis=1)

    # ``std::map``-style lookup: the scan assigns every match, so the last
    # duplicate wins.
    index: Dict[tuple, int] = {}
    for edge_id, (a, b) in enumerate(edges):
        index[(int(a), int(b))] = edge_id

    spig = np.zeros((elements.shape[0], 3), dtype=np.int64)
    for row, (n0, n1, n2) in enumerate(elements):
        n0, n1, n2 = int(n0), int(n1), int(n2)
        # (col 2) pair (n0, n1): + when stored ascending
        if (n0, n1) in index:
            spig[row, 2] = index[(n0, n1)] + 1
        elif (n1, n0) in index:
            spig[row, 2] = -(index[(n1, n0)] + 1)
        # (col 1) pair (n0, n2): - when stored ascending
        if (n0, n2) in index:
            spig[row, 1] = -(index[(n0, n2)] + 1)
        elif (n2, n0) in index:
            spig[row, 1] = index[(n2, n0)] + 1
        # (col 0) pair (n1, n2): + when stored ascending
        if (n1, n2) in index:
            spig[row, 0] = index[(n1, n2)] + 1
        elif (n2, n1) in index:
            spig[row, 0] = -(index[(n2, n1)] + 1)
    return spig


def legacy_arrays(triangulation: TriangleMesh) -> Dict[str, np.ndarray]:
    """Convert a Triangle result into the ``.h1.mat`` arrays (0-based)."""
    points = np.asarray(triangulation.points, dtype=float).reshape(-1, 2)
    n_nodes = points.shape[0]

    triangles = np.asarray(triangulation.triangles, dtype=np.int64)
    triangles = np.sort(triangles, axis=1)

    attributes = np.asarray(triangulation.triangle_attributes)
    elab = attributes.astype(np.int64).ravel()

    if triangulation.edges is not None and len(triangulation.edges):
        # Triangle's edge list is 1-based, like ``segments``.
        edges = np.sort(np.asarray(triangulation.edges, dtype=np.int64) - 1, axis=1)
        if triangulation.edge_markers is not None:
            slab = np.asarray(triangulation.edge_markers, dtype=np.int64).ravel()
        else:
            slab = np.zeros(edges.shape[0], dtype=np.int64)
    else:
        edges = derive_edges(triangles)
        slab = np.zeros(edges.shape[0], dtype=np.int64)

    if triangulation.point_markers is not None:
        nlab = np.asarray(triangulation.point_markers, dtype=np.int64).ravel()
    else:
        nlab = np.zeros(n_nodes, dtype=np.int64)

    return {
        "node": points,
        "ele": triangles,
        "spig": build_ele2edge(triangles, edges),
        "spig2": edges,
        "nlab": nlab,
        "elab": elab,
        "slab": slab,
    }


def mesh_arrays_from_poly(
    filename: str | Path,
    switches: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """Read *filename*, mesh it, and return the legacy arrays.

    *switches* follows :func:`~fes.mesh.triangle_backend.triangulate_plc`:
    ``None`` selects the oracle's ``pq34AfeQ``; anything else is wrapped as
    ``"p" + switches + "AfeQ"``.
    """
    geometry = read_poly_plc(filename)
    return legacy_arrays(triangulate_plc(geometry, switches=switches))


def write_model_cache(
    filename: str | Path,
    out_dir: str | Path,
    *,
    switches: Optional[str] = None,
    stem: Optional[str] = None,
) -> Path:
    """Mesh *filename* and write ``<stem>.h1.mat.npz`` into *out_dir*."""
    path = Path(filename)
    arrays = mesh_arrays_from_poly(path, switches=switches)
    name = stem or path.stem
    target = Path(out_dir) / f"{name}.h1.mat.npz"
    write_cache(target, **arrays)
    return target
