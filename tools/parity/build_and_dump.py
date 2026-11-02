#!/usr/bin/env python3
"""Build the cFES oracle harness and dump its arrays to a fixture.

The harness compiles the real cFES sources (Shape.cpp, Quad.cpp, EleMat.cpp,
DoF.cpp, ...) and prints every array the Python reference path must reproduce.
The output is committed as ``tests/fixtures/cpp_oracle.json`` so the parity
test does not need a C++ toolchain.

Usage::

    python tools/parity/build_and_dump.py --cfes ../cFES

The cFES tree is only read, never modified. Compilation happens in a
temporary directory, so no build artifacts land in either repository.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PARITY_DIR = Path(__file__).resolve().parent

# Translation units from cFES that the harness links.  Mesh.cpp, Project.cpp
# and the TetGen/METIS importer are replaced by project_stub.cpp.
CFES_SOURCES = [
    "Shape.cpp",
    "Quad.cpp",
    "EleMat.cpp",
    "DoF.cpp",
    "Coupl.cpp",
    "Mtrl.cpp",
]


def find_lib_dir(flag: str, patterns: list[str], explicit: str | None) -> str | None:
    """Return the first existing directory matching *patterns*."""
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_dir():
            raise SystemExit(f"--{flag}: {path} is not a directory")
        return str(path)
    for pattern in patterns:
        path = Path(pattern).expanduser()
        if path.is_dir():
            return str(path)
    return None


def brew_prefix(formula: str) -> str | None:
    brew = shutil.which("brew")
    if brew is None:
        return None
    try:
        out = subprocess.run(
            [brew, "--prefix", formula],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError:
        return None
    prefix = Path(out.stdout.strip())
    return str(prefix / "lib") if (prefix / "lib").is_dir() else None


def gfortran_lib_dir() -> str | None:
    gfortran = shutil.which("gfortran")
    if gfortran is None:
        return None
    try:
        out = subprocess.run(
            [gfortran, "-print-file-name=libgfortran.dylib"],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError:
        return None
    lib = Path(out.stdout.strip())
    return str(lib.parent) if lib.is_file() else None


def link_flags(args: argparse.Namespace) -> list[str]:
    """Locate BLAS/LAPACK and the Fortran runtime the bundled Armadillo needs."""
    lib_dirs: list[str] = []

    openblas = find_lib_dir(
        "openblas-lib",
        [
            "/opt/homebrew/opt/openblas/lib",
            "/usr/local/opt/openblas/lib",
            "/usr/lib/x86_64-linux-gnu/openblas-pthread",
            "/usr/lib/x86_64-linux-gnu",
        ],
        args.openblas_lib,
    )
    if openblas is None:
        openblas = brew_prefix("openblas")
    if openblas is not None:
        lib_dirs.append(openblas)
    elif sys.platform.startswith("linux"):
        print(
            "warning: no OpenBLAS directory found; relying on the default "
            "linker search path",
            file=sys.stderr,
        )

    fortran = find_lib_dir(
        "gfortran-lib",
        [
            "/opt/homebrew/opt/gcc/lib/gcc/current",
            "/usr/local/opt/gcc/lib/gcc/current",
        ],
        args.gfortran_lib,
    )
    if fortran is None:
        fortran = gfortran_lib_dir()
    if fortran is not None:
        lib_dirs.append(fortran)

    flags: list[str] = []
    for directory in lib_dirs:
        flags += [f"-L{directory}", f"-Wl,-rpath,{directory}"]
    flags += ["-lopenblas"]
    if fortran is not None:
        # Homebrew's libopenblas links against the gfortran runtime; on Linux
        # the runtime is usually found automatically.
        flags.append("-lgfortran")
    flags += ["-lm"]
    return flags


def build_command(args: argparse.Namespace, cfes: Path, binary: Path) -> list[str]:
    include_dirs = [
        cfes / "src",
        cfes / "ext" / "arma" / "include",
    ]
    for extra in args.include:
        include_dirs.append(Path(extra).expanduser().resolve())

    sources = [str(PARITY_DIR / "dump_oracle.cpp"),
               str(PARITY_DIR / "project_stub.cpp")]
    sources += [str(cfes / "src" / name) for name in CFES_SOURCES]

    cmd = [
        args.cxx,
        "-O2",
        "-std=c++14",
        "-Wno-deprecated-declarations",
        "-Wno-unused-variable",
    ]
    cmd += [f"-I{directory}" for directory in include_dirs]
    cmd += sources
    cmd += ["-o", str(binary)]
    cmd += link_flags(args)
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cfes",
        default=str(REPO_ROOT.parent / "cFES"),
        help="path to the cFES checkout (default: ../cFES next to this repo)",
    )
    parser.add_argument(
        "--out",
        default=str(REPO_ROOT / "tests" / "fixtures" / "cpp_oracle.json"),
        help="fixture path to write",
    )
    parser.add_argument("--cxx", default=os.environ.get("CXX", "clang++"))
    parser.add_argument("--openblas-lib", default=None)
    parser.add_argument("--gfortran-lib", default=None)
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        help="extra include directory (repeatable)",
    )
    parser.add_argument(
        "--keep-binary",
        metavar="PATH",
        default=None,
        help="copy the compiled harness here for manual runs",
    )
    args = parser.parse_args()

    cfes = Path(args.cfes).expanduser().resolve()
    if not (cfes / "src" / "Shape.cpp").is_file():
        raise SystemExit(f"--cfes: {cfes} does not look like a cFES checkout")

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="fes-parity-") as tmp:
        binary = Path(tmp) / "dump_oracle"
        cmd = build_command(args, cfes, binary)
        print("+ " + " ".join(cmd), file=sys.stderr)
        subprocess.run(cmd, check=True)

        result = subprocess.run(
            [str(binary)], check=True, capture_output=True, text=True
        )
        if result.stderr:
            sys.stderr.write(result.stderr)

        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError as exc:  # pragma: no cover - harness bug
            raise SystemExit(f"harness produced invalid JSON: {exc}") from exc

        # Re-serialise so the committed fixture is stable and readable.
        out_path.write_text(json.dumps(data, indent=1, sort_keys=False) + "\n")

        if args.keep_binary:
            destination = Path(args.keep_binary).expanduser().resolve()
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(binary, destination)

    size = out_path.stat().st_size
    print(f"wrote {out_path} ({size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
