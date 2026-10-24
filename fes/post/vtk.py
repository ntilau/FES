"""Byte-exact VTK writers ported from the v0.1 C++ binary.

``mesh::Savefield`` (``cpp/src/mesh.cpp``) is called from the ``project``
constructor after ``normalize_order`` and *before* assembly runs.  That timing
is observable: the 2D assembly is what calls
``mesh::build_2d_edge_connectivity``, so the mesh ``.vtk`` carries only the
edges that came from the input PLC, never the derived interior edges.

The writer reproduces the reference text layout exactly -- ``std::ostream`` with
``std::setprecision(16)`` is ``'%.16g'``, and the ``CELLS``/``CELL_DATA``
blocks follow ``mesh::Savefield`` line for line.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

__all__ = ["write_mesh_vtk"]


def _c_int(value: int) -> int:
    """``(int)`` cast of an ``arma::uvec`` element.

    ``mesh::maxLab`` is ``UINT_MAX``; on the two's-complement targets the
    oracle runs on, ``(int)4294967295`` prints ``-1``.  Labels are emitted
    through that cast, so replicate the wrap instead of printing the unsigned
    value.
    """
    wrapped = int(value) & 0xFFFFFFFF
    return wrapped - 0x100000000 if wrapped >= 0x80000000 else wrapped


def write_mesh_vtk(mesh, field_name: str | Path) -> Path:
    """Write ``<field_name>.vtk`` exactly like ``mesh::Savefield``.

    *mesh* is a :class:`~fes.mesh.mesh2d.Mesh2D` or
    :class:`~fes.mesh.mesh3d.Mesh3D`; *field_name* is the C++ binary's
    ``opt->name`` (the ``.vtk`` suffix is appended here).
    """
    target = Path(f"{field_name}.vtk")
    out: List[str] = []
    add = out.append

    add("# vtk DataFile Version 2.0\n")
    add("mesh data\n")
    add("ASCII\n")
    add("DATASET UNSTRUCTURED_GRID\n")

    add(f"POINTS {mesh.n_nodes} double \n")
    for x, y, z in mesh.nodes:
        add(f"{x:.16g} {y:.16g} {z:.16g}\n")

    # ── mixed cells: 3D tetrahedra + faces, 2D triangles + boundary edges ──
    if mesh.n_tetras > 0:
        tetras = np.asarray(mesh.tetras)
        faces = np.asarray(mesh.faces)
        n_tetras = int(tetras.shape[0])
        n_faces = int(faces.shape[0])
        n_cells = n_tetras + n_faces
        n_verts = n_tetras * 5 + n_faces * 4

        add(f"CELLS {n_cells} {n_verts}\n")
        for a, b, c, d in tetras:
            add(f"4 {int(a)} {int(b)} {int(c)} {int(d)}\n")
        for a, b, c in faces:
            add(f"3 {int(a)} {int(b)} {int(c)}\n")

        add(f"CELL_TYPES {n_cells}\n")
        out.extend("10\n" for _ in range(n_tetras))
        out.extend("5\n" for _ in range(n_faces))

        tet_labels = mesh.tet_labels
        face_labels = mesh.face_labels
        have_face_labels = face_labels is not None and len(face_labels) > 0
        n_face_labels = 0 if face_labels is None else len(face_labels)

        add(f"CELL_DATA {n_cells}\n")

        add("SCALARS region_label int 1\n")
        add("LOOKUP_TABLE default\n")
        out.extend(f"{_c_int(tet_labels[i])}\n" for i in range(n_tetras))
        out.extend("0\n" for _ in range(n_faces))

        add("SCALARS boundary_label int 1\n")
        add("LOOKUP_TABLE default\n")
        out.extend("0\n" for _ in range(n_tetras))
        for i in range(n_faces):
            label = _c_int(face_labels[i]) if have_face_labels and i < n_face_labels else 0
            add(f"{label}\n")

    elif mesh.n_faces > 0:
        faces = np.asarray(mesh.faces)
        # ``Savefield`` runs before assembly; only the PLC edges exist yet.
        n_plc_edges = mesh.n_boundary_edges
        edges = np.asarray(mesh.edges[:n_plc_edges])
        n_faces = int(faces.shape[0])
        n_edges = int(edges.shape[0])
        n_cells = n_faces + n_edges
        n_verts = n_faces * 4 + n_edges * 3

        add(f"CELLS {n_cells} {n_verts}\n")
        for a, b, c in faces:
            add(f"3 {int(a)} {int(b)} {int(c)}\n")
        for a, b in edges:
            add(f"2 {int(a)} {int(b)}\n")

        add(f"CELL_TYPES {n_cells}\n")
        out.extend("5\n" for _ in range(n_faces))
        out.extend("3\n" for _ in range(n_edges))

        face_labels = mesh.face_labels
        edge_labels = mesh.edge_labels
        have_edge_labels = edge_labels is not None and len(edge_labels) > 0
        n_edge_labels = 0 if edge_labels is None else len(edge_labels)

        add(f"CELL_DATA {n_cells}\n")

        add("SCALARS region_label int 1\n")
        add("LOOKUP_TABLE default\n")
        out.extend(f"{_c_int(face_labels[i])}\n" for i in range(n_faces))
        out.extend("0\n" for _ in range(n_edges))

        add("SCALARS boundary_label int 1\n")
        add("LOOKUP_TABLE default\n")
        out.extend("0\n" for _ in range(n_faces))
        for i in range(n_edges):
            label = _c_int(edge_labels[i]) if have_edge_labels and i < n_edge_labels else 0
            add(f"{label}\n")

    target.write_text("".join(out))
    return target
