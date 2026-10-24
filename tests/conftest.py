"""Shared pytest configuration.

Mesh-generation tests are marked ``triangle`` (2D) or ``tetgen`` (3D) and are
skipped when the matching shared library is unavailable.  CI does not ship
``dep/``, so those tests run only where ``python tools/build_triangle.py`` /
``python tools/build_tetgen.py`` has produced a library.
"""

from __future__ import annotations

import pytest

from fes.mesh.tetgen_backend import TetGenUnavailable, library_path as tetgen_path
from fes.mesh.triangle_backend import TriangleUnavailable, library_path


def _triangle_available() -> bool:
    try:
        library_path()
    except TriangleUnavailable:
        return False
    return True


def _tetgen_available() -> bool:
    try:
        tetgen_path()
    except TetGenUnavailable:
        return False
    return True


_MESHERS = {
    "triangle": (
        _triangle_available,
        "Triangle shared library not available (see tools/build_triangle.py)",
    ),
    "tetgen": (
        _tetgen_available,
        "TetGen shared library not available (see tools/build_tetgen.py)",
    ),
}


def pytest_collection_modifyitems(config, items):
    skips = {
        marker: pytest.mark.skip(reason=reason)
        for marker, (available, reason) in _MESHERS.items()
        if not available()
    }
    if not skips:
        return
    for item in items:
        for marker, skip in skips.items():
            if marker in item.keywords:
                item.add_marker(skip)
