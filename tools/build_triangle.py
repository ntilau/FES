"""Build the Triangle shared library used by ``fes.mesh.triangle_backend``.

Triangle is not vendored with the repository.  The v0.1 C++ binary linked Triangle
1.6 compiled from ``dep/src/triangle/triangle.cpp``, so reproducing its meshes
means building that same source.  This script does so, applying one documented
deviation: memory pools are zero-filled.

Why zeroing is required
-----------------------
Each vertex record reserves a slot (``m->vertex2triindex``) for a triangle
pointer.  ``formskeleton`` fills it with ``makevertexmap`` for every vertex that
appears in the triangulation.  Duplicate input points are dropped as ``undead``
and keep whatever that slot held.  ``poolinit`` obtains its blocks with
``malloc``, so the slot is uninitialised.

The v0.1 C++ binary received fresh zeroed pages from the allocator, so the slot
read as NULL and ``insertsegment`` located the vertex by search instead.  In a
long-lived Python process the same slot holds recycled heap data, which
``insertsegment`` treats as a triangle pointer; ``org(searchtri1, checkvertex)``
then dereferences a garbage address and ``patch_antenna_2D`` dies with SIGSEGV.
Allocating the pools with ``calloc`` reproduces the zeroed state
deterministically, and the resulting mesh is identical (100 points, 156
triangles, 59 segments).

Usage::

    python tools/build_triangle.py
    python tools/build_triangle.py -o build/libtriangle.dylib --compiler clang++

The patched source is written beside the library (``triangle_patched.cpp``) so
the deviation stays inspectable.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

PY_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PY_DIR

DEFAULT_SOURCE = REPO_ROOT / "dep" / "src" / "triangle" / "triangle.cpp"
DEFAULT_INCLUDE = REPO_ROOT / "dep" / "include"

#: ``poolinit``'s first block: the only allocation a small model needs.
_POOLINIT_OLD = """  pool->firstblock = (VOID **)
    trimalloc(pool->itemsfirstblock * pool->itembytes + (int) sizeof(VOID *) +
              pool->alignbytes);"""
_POOLINIT_NEW = """  pool->firstblock = (VOID **)
    calloc(1, pool->itemsfirstblock * pool->itembytes + (int) sizeof(VOID *) +
              pool->alignbytes);"""

#: ``poolalloc``'s later blocks, used when a pool outgrows its first block.
_POOLALLOC_OLD = """        newblock = (VOID **) trimalloc(pool->itemsperblock * pool->itembytes +
                                       (int) sizeof(VOID *) +
                                       pool->alignbytes);"""
_POOLALLOC_NEW = """        newblock = (VOID **) calloc(1, pool->itemsperblock * pool->itembytes +
                                       (int) sizeof(VOID *) +
                                       pool->alignbytes);"""

_PATCHES = (
    ("poolinit", _POOLINIT_OLD, _POOLINIT_NEW),
    ("poolalloc", _POOLALLOC_OLD, _POOLALLOC_NEW),
)


def default_output() -> Path:
    """Return the platform's shared-library name inside ``build``."""
    if sys.platform == "darwin":
        name = "libtriangle.dylib"
    elif sys.platform.startswith("win"):
        name = "triangle.dll"
    else:
        name = "libtriangle.so"
    return PY_DIR / "build" / name


def apply_patch(source: str) -> str:
    """Return *source* with every pool allocation switched to ``calloc``.

    Raises :class:`ValueError` when a pattern is missing or ambiguous, which
    means the Triangle version in ``dep/`` is not the one this patch targets.
    """
    patched = source
    for label, old, new in _PATCHES:
        count = patched.count(old)
        if count != 1:
            raise ValueError(
                f"{label}: expected exactly one allocation site, found {count}; "
                "refusing to patch an unrecognised Triangle source"
            )
        patched = patched.replace(old, new)
    return patched


def build(
    source: Path = DEFAULT_SOURCE,
    include: Path = DEFAULT_INCLUDE,
    output: Path | None = None,
    compiler: str | None = None,
) -> Path:
    """Patch *source*, compile it, and return the library path."""
    source = Path(source)
    include = Path(include)
    output = Path(output) if output is not None else default_output()
    compiler = compiler or shutil.which("c++") or shutil.which("clang++")

    if not source.is_file():
        raise FileNotFoundError(
            f"Triangle source not found at {source}; the v0.1 tree ships it "
            "under dep/src/triangle/ (see docs/install.md)"
        )
    if not compiler:
        raise FileNotFoundError("no C++ compiler found; pass --compiler")

    patched = apply_patch(source.read_text(encoding="utf-8"))
    output.parent.mkdir(parents=True, exist_ok=True)
    patched_path = output.parent / "triangle_patched.cpp"
    patched_path.write_text(patched, encoding="utf-8")

    command = [
        compiler,
        "-O2",
        "-fPIC",
        "-shared",
        "-DTRILIBRARY",
        "-DANSI_DECLARATORS",
        f"-I{include}",
        str(patched_path),
        "-o",
        str(output),
    ]
    subprocess.run(command, check=True)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--include", type=Path, default=DEFAULT_INCLUDE)
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument("--compiler", default=None)
    args = parser.parse_args(argv)

    try:
        library = build(args.source, args.include, args.output, args.compiler)
    except (FileNotFoundError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {library}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
