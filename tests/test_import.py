"""The core package must import without the optional ML/visualisation stack.

``torch``, ``gpytorch`` and ``pyvista`` are only needed by the DNN-GP projects
and the field renderer.  A fresh install without those extras has to be able to
``import fes``, run the CLI, and solve 2D models, so this test imports the whole
non-optional surface inside a subprocess whose import hooks refuse those names.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PY_DIR = Path(__file__).resolve().parents[1]

BLOCKER = """
import sys

BLOCKED = {"torch", "gpytorch", "pyvista", "vtk", "sklearn"}


class Blocker:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ImportError(f"{name} is not installed (blocked by the test)")
        return None


sys.meta_path.insert(0, Blocker())

import fes
import fes.api
import fes.cli
import fes.drivers
import fes.mesh.mesh2d
import fes.core.assemblers

assert fes.api.load_drivers() is None
assert "em_ez_fd" in fes.api.DRIVERS
assert "fes.projects" not in sys.modules, "projects must stay lazy"
"""


def test_import_without_optional_dependencies() -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PY_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-c", BLOCKER],
        cwd=str(PY_DIR),
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_projects_package_is_not_imported_by_the_core() -> None:
    code = (
        "import sys, fes, fes.api, fes.drivers\n"
        "assert 'fes.projects' not in sys.modules\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PY_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(PY_DIR),
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
