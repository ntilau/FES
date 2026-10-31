"""Build the TetGen shared library used by ``fes.mesh.tetgen_backend``.

The v0.1 C++ binary meshes 3D ``.poly`` files by calling ``tetgenio::load_poly``
and ``tetrahedralize("pqAfeeQ", ...)`` from the TetGen sources vendored in
``dep/tetgen``.  Reproducing its meshes bit-for-bit means compiling the
same source, so this script does that and adds a small C shim
(``tools/tetgen_shim.cpp``) because TetGen's interface is C++ and ctypes
cannot own a ``tetgenio``.

Usage::

    python tools/build_tetgen.py
    python tools/build_tetgen.py --compiler clang++ -o build/libtetgen.dylib

TetGen itself is compiled unmodified with ``-DTETLIBRARY`` (the same define as
``dep/tetgen/CMakeLists.txt``); only the shim (a separate translation unit)
is ours.  The library is optional tooling: the Python package runs without it
and mesh-generation tests skip when it is absent.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

PY_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PY_DIR

DEFAULT_SOURCE_DIR = REPO_ROOT / "dep" / "tetgen"
DEFAULT_SHIM = PY_DIR / "tools" / "tetgen_shim.cpp"


def default_output() -> Path:
    """Return the platform's shared-library name inside ``build``."""
    if sys.platform == "darwin":
        name = "libtetgen.dylib"
    elif sys.platform.startswith("win"):
        name = "tetgen.dll"
    else:
        name = "libtetgen.so"
    return PY_DIR / "build" / name


def build(
    source_dir: Path = DEFAULT_SOURCE_DIR,
    shim: Path = DEFAULT_SHIM,
    output: Path | None = None,
    compiler: str | None = None,
) -> Path:
    """Compile TetGen plus the shim and return the library path."""
    source_dir = Path(source_dir)
    shim = Path(shim)
    output = Path(output) if output is not None else default_output()
    compiler = compiler or shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")

    for path in (source_dir / "tetgen.cxx", source_dir / "predicates.cxx", shim):
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} not found; the v0.1 tree ships TetGen under dep/tetgen/"
            )
    if not compiler:
        raise FileNotFoundError("no C++ compiler found; pass --compiler")

    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        compiler,
        "-std=c++11",
        "-O2",
        "-fPIC",
        "-shared",
        "-DTETLIBRARY",
        "-Wno-deprecated-declarations",
        f"-I{source_dir}",
        str(source_dir / "tetgen.cxx"),
        str(source_dir / "predicates.cxx"),
        str(shim),
        "-o",
        str(output),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--shim", type=Path, default=DEFAULT_SHIM)
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument("--compiler", default=None)
    args = parser.parse_args(argv)

    try:
        library = build(args.source, args.shim, args.output, args.compiler)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", "") or exc
        print(f"error: {detail}", file=sys.stderr)
        return 1
    print(f"wrote {library}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
