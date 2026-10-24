"""ctypes binding to the TetGen 1.5 mesh generator.

The v0.1 C++ solver meshed 3D ``.poly`` files with TetGen's library API
(``preprocessing::preprocess3D``): ``tetgenio::load_poly`` reads the PLC and
``tetrahedralize("pqAfeeQ", ...)`` produces the mesh, with ``-e``/``-f``
returning *all* edges and triangular faces.  Reproducing its meshes
bit-for-bit is a prerequisite for S-parameter agreement, so this module binds the
same C library rather than reimplementing a mesher.

TetGen is not vendored as a shared library.  Point ``FES_TETGEN_LIB`` at one,
or build the vendored source into ``build/`` with::

    python tools/build_tetgen.py

The shim it compiles (``tools/tetgen_shim.cpp``) is needed because TetGen's
interface is C++: ``tetgenio`` has a virtual destructor and constructors that
ctypes cannot run.

Indexing: every connectivity array is returned **1-based**, exactly as TetGen
emits it.  ``Createmesh`` subtracts one before storing, and so does
:meth:`fes.mesh.mesh3d.Mesh3D.from_tetgen`; keeping the raw numbering here
means the backend can be compared with TetGen's own output arrays.

When no library is available, :func:`tetrahedralize_poly` raises
:class:`TetGenUnavailable`; callers should skip mesh-generation tests in that
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
    "TetGenError",
    "TetGenMesh",
    "TetGenUnavailable",
    "SWITCH_DEFAULT_3D",
    "library_path",
    "tetrahedralize_poly",
]

#: The oracle's switches: PLC, quality, region attributes, all faces, all
#: edges, quiet.  ``pre_processor.cpp`` prepends nothing when ``+poly`` has no
#: argument and appends ``AfeeQ`` to an explicit switch string.
SWITCH_DEFAULT_3D = "pqAfeeQ"


class TetGenUnavailable(RuntimeError):
    """Raised when no usable TetGen shared library can be found."""


class TetGenError(RuntimeError):
    """Raised when TetGen rejects the PLC or aborts (``terminatetetgen``)."""


@dataclass
class TetGenMesh:
    """The subset of a ``tetgenio`` output that ``Createmesh`` consumes.

    ``edges``/``faces``/``tetrahedra`` use ``first_number`` (1 for file
    input); ``faces`` and ``tetrahedra`` keep TetGen's corner orientation.
    ``tetra_attributes`` is the flat attribute array (``None`` when the run
    produced none), which the oracle reads as one value per tetrahedron.
    """

    points: np.ndarray  # (n, 3)
    edges: np.ndarray  # (m, 2)
    faces: np.ndarray  # (f, 3)
    face_markers: np.ndarray  # (f,)
    tetrahedra: np.ndarray  # (t, 4)
    tetra_attributes: Optional[np.ndarray]  # (t * n_attributes,)
    n_attributes: int
    first_number: int
    switches: str


# ── library discovery ───────────────────────────────────────────────────────


def _candidate_names() -> list[str]:
    return ["libfetgen.dylib", "libfetgen.so", "fetgen.dll"]


def library_path() -> Path:
    """Locate the TetGen shim library, or raise :class:`TetGenUnavailable`."""
    override = os.environ.get("FES_TETGEN_LIB")
    if override:
        path = Path(override).expanduser()
        if path.is_file():
            return path.resolve()
        raise TetGenUnavailable(f"FES_TETGEN_LIB={override!r} does not name a file")

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

    found = ctypes.util.find_library("fetgen")
    if found:
        return Path(found)

    raise TetGenUnavailable(
        "TetGen shared library not found; set FES_TETGEN_LIB or build "
        "build/libfetgen.dylib with tools/build_tetgen.py"
    )


_LIB: Optional[ctypes.CDLL] = None


def _load():
    """Load the shared library once and type every entry point."""
    global _LIB
    if _LIB is not None:
        return _LIB

    path = library_path()
    try:
        lib = ctypes.CDLL(str(path))
    except OSError as exc:
        raise TetGenUnavailable(f"could not load TetGen from {path}: {exc}") from exc

    try:
        new = lib.fes_tetgenio_new
        delete = lib.fes_tetgenio_delete
        load_poly = lib.fes_tetgenio_load_poly
        tetrahedralize = lib.fes_tetgen_tetrahedralize
    except AttributeError as exc:
        raise TetGenUnavailable(
            f"{path} was not built with tools/build_tetgen.py (missing shim)"
        ) from exc

    void = ctypes.c_void_p
    new.argtypes = []
    new.restype = void
    delete.argtypes = [void]
    delete.restype = None
    load_poly.argtypes = [void, ctypes.c_char_p]
    load_poly.restype = ctypes.c_int
    tetrahedralize.argtypes = [ctypes.c_char_p, void, void]
    tetrahedralize.restype = ctypes.c_int

    def integer(name):
        function = getattr(lib, name)
        function.argtypes = [void]
        function.restype = ctypes.c_int
        return function

    def array(name, ctype):
        function = getattr(lib, name)
        function.argtypes = [void]
        function.restype = ctypes.POINTER(ctype)
        return function

    lib._fes = {
        "new": new,
        "delete": delete,
        "load_poly": load_poly,
        "tetrahedralize": tetrahedralize,
        "firstnumber": integer("fes_firstnumber"),
        "numberofpoints": integer("fes_numberofpoints"),
        "pointlist": array("fes_pointlist", ctypes.c_double),
        "numberofedges": integer("fes_numberofedges"),
        "edgelist": array("fes_edgelist", ctypes.c_int),
        "numberoftrifaces": integer("fes_numberoftrifaces"),
        "trifacelist": array("fes_trifacelist", ctypes.c_int),
        "trifacemarkerlist": array("fes_trifacemarkerlist", ctypes.c_int),
        "numberoftetrahedra": integer("fes_numberoftetrahedra"),
        "tetrahedronlist": array("fes_tetrahedronlist", ctypes.c_int),
        "tetrahedronattributelist": array(
            "fes_tetrahedronattributelist", ctypes.c_double
        ),
        "numberoftetrahedronattributes": integer(
            "fes_numberoftetrahedronattributes"
        ),
    }
    _LIB = lib
    return lib


# ── array helpers ───────────────────────────────────────────────────────────


def _copy(pointer, shape, dtype):
    """Copy a TetGen output buffer into a NumPy array."""
    size = int(np.prod(shape))
    if size == 0 or not pointer:
        return np.zeros(shape, dtype=dtype)
    return np.ctypeslib.as_array(pointer, shape=(size,)).reshape(shape).copy()


# ── entry point ─────────────────────────────────────────────────────────────


def _basename(poly: str | Path) -> str:
    """Strip the ``.poly``/``.smesh`` suffix: ``load_poly`` appends its own."""
    path = Path(poly)
    if path.suffix in (".poly", ".smesh"):
        path = path.with_suffix("")
    return str(path)


def tetrahedralize_poly(
    poly: str | Path, switches: str = SWITCH_DEFAULT_3D
) -> TetGenMesh:
    """Mesh *poly* with TetGen, exactly as ``pre_processor.cpp`` does.

    TetGen prints its banner and the ``Opening ...`` line itself; ``Q`` in the
    switches suppresses the banner, matching the oracle's output.
    """
    lib = _load()
    fes = lib._fes

    inp = fes["new"]()
    out = fes["new"]()
    try:
        if not fes["load_poly"](inp, os.fsencode(_basename(poly))):
            raise TetGenError(f"TetGen could not read {poly}")

        code = fes["tetrahedralize"](switches.encode(), inp, out)
        if code != 0:
            raise TetGenError(
                f"TetGen aborted on {poly} with switch "
                f"{switches!r} (terminatetetgen code {code})"
            )

        first = fes["firstnumber"](out)
        n_points = fes["numberofpoints"](out)
        n_edges = fes["numberofedges"](out)
        n_faces = fes["numberoftrifaces"](out)
        n_tets = fes["numberoftetrahedra"](out)
        n_attributes = fes["numberoftetrahedronattributes"](out)

        points = _copy(fes["pointlist"](out), (n_points, 3), np.float64)
        edges = _copy(fes["edgelist"](out), (n_edges, 2), np.int64)
        faces = _copy(fes["trifacelist"](out), (n_faces, 3), np.int64)
        markers = _copy(fes["trifacemarkerlist"](out), (n_faces,), np.int32)
        tetrahedra = _copy(fes["tetrahedronlist"](out), (n_tets, 4), np.int64)
        attributes = None
        if n_attributes > 0:
            attributes = _copy(
                fes["tetrahedronattributelist"](out),
                (n_tets * n_attributes,),
                np.float64,
            )
    finally:
        fes["delete"](inp)
        fes["delete"](out)

    return TetGenMesh(
        points=points,
        edges=edges,
        faces=faces,
        face_markers=markers.astype(np.int64),
        tetrahedra=tetrahedra,
        tetra_attributes=attributes,
        n_attributes=int(n_attributes),
        first_number=int(first),
        switches=switches,
    )
