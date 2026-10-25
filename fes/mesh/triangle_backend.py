"""ctypes binding to the Triangle 1.6 mesh generator.

The v0.1 C++ solver meshed 2D ``.poly`` files with Triangle's library API
(``cpp/src/pre_processor.cpp::triangulatemesh2D``).  Reproducing its meshes
bit-for-bit is a prerequisite for S-parameter agreement, so this module binds the
same C library rather than reimplementing a mesher.

Triangle is not vendored with the repository.  Point ``FES_TRIANGLE`` at a
shared library (``libtriangle.dylib`` / ``.so``), or build one into
``build/`` from the v0.1 tree with::

    python tools/build_triangle.py

That script patches one thing in ``dep/triangle/triangle.cpp``: memory
pools are zero-filled.  Triangle leaves the ``vertex->triangle`` slot of
*unused* (duplicate) vertices uninitialised, and the v0.1 C++ binary only worked
because its allocator handed back fresh zero pages; in a Python process the

Building from a stock Triangle 1.6 is accepted for most models but is not
bit-reproducible for geometries with duplicate points.

When no library is available, :func:`triangulate_plc` raises
:class:`TriangleUnavailable`; callers should skip mesh-generation tests in that
case (CI does not ship ``dep/``).
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from ..paths import repo_root

__all__ = [
    "PolyGeometry",
    "TriangleMesh",
    "TriangleUnavailable",
    "library_path",
    "triangulate_plc",
]


class TriangleUnavailable(RuntimeError):
    """Raised when no usable Triangle shared library can be found."""


# ── C interface ─────────────────────────────────────────────────────────────

# Every pointer field is opaque here; the concrete arrays are read through
# ``ctypes.cast`` after the call.  Keeping the layout identical to
# ``dep/include/triangle.h`` is what matters.
class _TriangulateIO(ctypes.Structure):
    _fields_ = [
        ("pointlist", ctypes.c_void_p),
        ("pointattributelist", ctypes.c_void_p),
        ("pointmarkerlist", ctypes.c_void_p),
        ("numberofpoints", ctypes.c_int),
        ("numberofpointattributes", ctypes.c_int),
        ("trianglelist", ctypes.c_void_p),
        ("triangleattributelist", ctypes.c_void_p),
        ("trianglearealist", ctypes.c_void_p),
        ("neighborlist", ctypes.c_void_p),
        ("numberoftriangles", ctypes.c_int),
        ("numberofcorners", ctypes.c_int),
        ("numberoftriangleattributes", ctypes.c_int),
        ("segmentlist", ctypes.c_void_p),
        ("segmentmarkerlist", ctypes.c_void_p),
        ("numberofsegments", ctypes.c_int),
        ("holelist", ctypes.c_void_p),
        ("numberofholes", ctypes.c_int),
        ("regionlist", ctypes.c_void_p),
        ("numberofregions", ctypes.c_int),
        ("edgelist", ctypes.c_void_p),
        ("edgemarkerlist", ctypes.c_void_p),
        ("normlist", ctypes.c_void_p),
        ("numberofedges", ctypes.c_int),
    ]


# The v0.1 C++ binary is a separate translation unit, so Triangle's entry point
# is C++-mangled.  Accept the plain C name as well for system libraries.
_MANGLED = "_Z11triangulatePcP13triangulateioS1_S1_"

_SWITCH_DEFAULT_2D = "pq34AfeQ"


@dataclass
class PolyGeometry:
    """The subset of a ``.poly`` file that Triangle consumes.

    Segment indices are **1-based**, exactly as the C++ binary passes them to
    Triangle (the C++ reader does not enable Triangle's ``z`` switch).
    """

    points: np.ndarray  # (n, 2)
    point_markers: Optional[np.ndarray] = None  # (n,) int
    segments: Optional[np.ndarray] = None  # (m, 2) 1-based
    segment_markers: Optional[np.ndarray] = None  # (m,) int
    holes: Optional[np.ndarray] = None  # (h, 2)
    regions: Optional[np.ndarray] = None  # (r, 4): x y attribute max_area


@dataclass
class TriangleMesh:
    """Triangulation returned by Triangle.

    ``triangles`` is 0-based (the C++ binary stores ``trianglelist - 1``);
    ``segments`` stays 1-based because the C++ binary copies ``segmentlist``
    verbatim.  ``triangle_attributes`` is the per-triangle regional attribute
    that becomes ``mesh->facLab``.

    ``edges``/``edge_markers`` are the full edge list (Triangle's ``-e``
    output).  The reference implementation ignores them, but the legacy cache
    stores exactly this list as ``spig2``/``slab``, so they are kept here.
    """

    points: np.ndarray  # (np, 2)
    point_markers: Optional[np.ndarray]
    triangles: np.ndarray  # (nt, 3) 0-based
    triangle_attributes: np.ndarray  # (nt,)
    segments: np.ndarray  # (ns, 2) 1-based
    segment_markers: Optional[np.ndarray]
    switches: str
    edges: Optional[np.ndarray] = None  # (ne, 2) 1-based
    edge_markers: Optional[np.ndarray] = None  # (ne,)


# ── library discovery ───────────────────────────────────────────────────────


def _candidate_names() -> list[str]:
    return ["libtriangle.dylib", "libtriangle.so", "triangle.dll", "triangle"]


def library_path() -> Path:
    """Locate the Triangle shared library, or raise :class:`TriangleUnavailable`."""
    override = os.environ.get("FES_TRIANGLE")
    if override:
        path = Path(override).expanduser()
        if path.is_file():
            return path.resolve()
        raise TriangleUnavailable(
            f"FES_TRIANGLE={override!r} does not name a file"
        )

    roots = []
    try:
        roots.append(repo_root() / "build")
    except Exception:  # pragma: no cover - defensive for installed copies
        pass
    roots.append(Path(__file__).resolve().parents[2] / "build")

    for directory in roots:
        for name in _candidate_names():
            candidate = directory / name
            if candidate.is_file():
                return candidate.resolve()

    found = ctypes.util.find_library("triangle")
    if found:
        return Path(found)

    raise TriangleUnavailable(
        "Triangle shared library not found; set FES_TRIANGLE or build "
        "build/libtriangle.dylib from dep/triangle/triangle.cpp"
    )


_LIB: Optional[ctypes.CDLL] = None
_FREER = None


def _free():
    global _FREER
    if _FREER is None:
        libc = ctypes.CDLL(None)
        _FREER = libc.free
        _FREER.argtypes = [ctypes.c_void_p]
        _FREER.restype = None
    return _FREER


def _load():  # pragma: no cover - exercised only with a native library
    """Load the shared library once and resolve ``triangulate``."""
    global _LIB
    if _LIB is not None:
        return _LIB

    path = library_path()
    try:
        lib = ctypes.CDLL(str(path))
    except OSError as exc:
        raise TriangleUnavailable(f"could not load Triangle from {path}: {exc}") from exc

    function = None
    for name in (_MANGLED, "triangulate"):
        try:
            function = getattr(lib, name)
            break
        except AttributeError:
            continue
    if function is None:
        raise TriangleUnavailable(
            f"{path} does not export a triangulate entry point"
        )
    function.argtypes = [
        ctypes.c_char_p,
        ctypes.POINTER(_TriangulateIO),
        ctypes.POINTER(_TriangulateIO),
        ctypes.c_void_p,
    ]
    function.restype = None

    lib._fes_triangulate = function  # type: ignore[attr-defined]
    _LIB = lib
    return lib


# ── array helpers ───────────────────────────────────────────────────────────


def _array(values, count: int, ctype, dtype, label: str):
    array = (ctype * max(count, 1))()
    if count and values is not None:
        flat = np.ascontiguousarray(np.asarray(values, dtype=dtype).reshape(-1))
        if flat.size != count:
            raise ValueError(f"expected {count} {label}, got {flat.size}")
        array[:count] = flat.tolist()
    return array


def _double_array(values, count: int):
    return _array(values, count, ctypes.c_double, np.float64, "doubles")


def _int_array(values, count: int):
    return _array(values, count, ctypes.c_int, np.intc, "ints")


def _read(pointer, count: int, ctype, dtype) -> np.ndarray:
    if not pointer or count <= 0:
        return np.zeros(0, dtype=dtype)
    array = ctypes.cast(pointer, ctypes.POINTER(ctype))
    return np.ctypeslib.as_array(array, shape=(count,)).copy()


def _read_doubles(pointer, count: int) -> np.ndarray:
    return _read(pointer, count, ctypes.c_double, np.float64)


def _read_ints(pointer, count: int) -> np.ndarray:
    return _read(pointer, count, ctypes.c_int, np.intc)


# ── public entry point ──────────────────────────────────────────────────────


def triangulate_plc(
    geometry: PolyGeometry,
    switches: Optional[str] = None,
) -> TriangleMesh:
    """Run Triangle on *geometry* and return a copy of its output.

    *switches* follows the oracle: when empty, ``"pq34AfeQ"`` is used;
    otherwise ``"p" + switches + "AfeQ"``.
    """
    lib = _load()
    function = lib._fes_triangulate  # type: ignore[attr-defined]

    points = np.asarray(geometry.points, dtype=np.float64).reshape(-1, 2)
    number_of_points = points.shape[0]
    if number_of_points == 0:
        raise ValueError("a .poly geometry needs at least one point")

    point_markers = geometry.point_markers
    segments = geometry.segments
    segment_markers = geometry.segment_markers
    holes = geometry.holes
    regions = geometry.regions

    number_of_segments = 0 if segments is None else int(np.asarray(segments).shape[0])
    number_of_holes = 0 if holes is None else int(np.asarray(holes).shape[0])
    number_of_regions = 0 if regions is None else int(np.asarray(regions).shape[0])

    in_io = _TriangulateIO()
    out_io = _TriangulateIO()

    # Keep the allocations alive for the duration of the call.
    point_list = _double_array(points, 2 * number_of_points)
    point_marker_list = (
        _int_array(point_markers, number_of_points)
        if point_markers is not None
        else None
    )
    segment_list = (
        _int_array(segments, 2 * number_of_segments)
        if number_of_segments
        else None
    )
    segment_marker_list = (
        _int_array(segment_markers, number_of_segments)
        if number_of_segments and segment_markers is not None
        else None
    )
    hole_list = _double_array(holes, 2 * number_of_holes) if number_of_holes else None
    region_list = (
        _double_array(regions, 4 * number_of_regions) if number_of_regions else None
    )

    in_io.numberofpoints = number_of_points
    in_io.numberofpointattributes = 0
    in_io.pointlist = ctypes.cast(point_list, ctypes.c_void_p)
    if point_marker_list is not None:
        in_io.pointmarkerlist = ctypes.cast(point_marker_list, ctypes.c_void_p)
    in_io.numberofsegments = number_of_segments
    if segment_list is not None:
        in_io.segmentlist = ctypes.cast(segment_list, ctypes.c_void_p)
    if segment_marker_list is not None:
        in_io.segmentmarkerlist = ctypes.cast(segment_marker_list, ctypes.c_void_p)
    in_io.numberofholes = number_of_holes
    if hole_list is not None:
        in_io.holelist = ctypes.cast(hole_list, ctypes.c_void_p)
    in_io.numberofregions = number_of_regions
    if region_list is not None:
        in_io.regionlist = ctypes.cast(region_list, ctypes.c_void_p)

    command = _SWITCH_DEFAULT_2D if not switches else "p" + switches + "AfeQ"
    function(
        command.encode("ascii"),
        ctypes.byref(in_io),
        ctypes.byref(out_io),
        None,
    )

    try:
        if out_io.numberofpoints <= 0 or out_io.numberoftriangles <= 0:
            raise RuntimeError(
                "Triangle produced an empty mesh; check the .poly segments "
                "and the mesher switches"
            )

        n_points = int(out_io.numberofpoints)
        n_triangles = int(out_io.numberoftriangles)
        n_segments = int(out_io.numberofsegments)
        n_attributes = int(out_io.numberoftriangleattributes)
        n_edges = int(out_io.numberofedges)

        points_out = _read_doubles(out_io.pointlist, 2 * n_points).reshape(n_points, 2)
        markers_out = _read_ints(out_io.pointmarkerlist, n_points) if out_io.pointmarkerlist else None

        raw_triangles = _read_ints(out_io.trianglelist, 3 * n_triangles).reshape(
            n_triangles, 3
        )
        if n_attributes > 0 and out_io.triangleattributelist:
            attributes = _read_doubles(
                out_io.triangleattributelist, n_attributes * n_triangles
            ).reshape(n_triangles, n_attributes)[:, 0]
        else:
            attributes = np.zeros(n_triangles)

        segments_out = _read_ints(out_io.segmentlist, 2 * n_segments).reshape(
            n_segments, 2
        )
        segment_markers_out = (
            _read_ints(out_io.segmentmarkerlist, n_segments)
            if out_io.segmentmarkerlist
            else None
        )

        if n_edges > 0 and out_io.edgelist:
            edges_out = _read_ints(out_io.edgelist, 2 * n_edges).reshape(n_edges, 2)
            edge_markers_out = (
                _read_ints(out_io.edgemarkerlist, n_edges)
                if out_io.edgemarkerlist
                else None
            )
        else:
            edges_out = None
            edge_markers_out = None
    finally:
        # ``triangulate`` allocates these with ``trimalloc``.  It does *not*
        # copy ``holelist``/``regionlist``: the output structure aliases the
        # caller's arrays, so freeing them here aborts the process.
        release = _free()
        for pointer in (
            out_io.pointlist,
            out_io.pointattributelist,
            out_io.pointmarkerlist,
            out_io.trianglelist,
            out_io.triangleattributelist,
            out_io.neighborlist,
            out_io.segmentlist,
            out_io.segmentmarkerlist,
            out_io.edgelist,
            out_io.edgemarkerlist,
            out_io.normlist,
        ):
            if pointer:
                release(pointer)

    return TriangleMesh(
        points=points_out,
        point_markers=markers_out,
        triangles=raw_triangles - 1,
        triangle_attributes=attributes,
        segments=segments_out,
        segment_markers=segment_markers_out,
        switches=command,
        edges=edges_out,
        edge_markers=edge_markers_out,
    )
