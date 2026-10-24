"""Planar geometry I/O: the Triangle ``.poly`` PLC format.

Port of ``IOrPoly.m`` / ``IOwPoly.m`` and the ``.poly`` reader in
``preprocessing::read_poly_for_triangle``.  The result is a
:class:`~fes.mesh.triangle_backend.PolyGeometry`, the exact input the mesher
and the reference consume.

The parser is deliberately line-sequential, like the reference C++: the declared
section counts decide how many lines are consumed, so a file whose counts
disagree with its contents (``patch_antenna_2D`` declares 28 segments but lists
25) shifts the later sections the same way the C++ reader does.  Every missing
or unparsable field becomes zero instead of raising.
"""

from __future__ import annotations

import re
from typing import List, Optional

import numpy as np

from .triangle_backend import PolyGeometry

__all__ = [
    "read_poly_plc",
    "write_poly",
]

_INT_PREFIX = re.compile(r"[+-]?\d+")
_FLOAT_PREFIX = re.compile(
    r"[+-]?(?:(?:\d+\.?\d*)|(?:\.\d+))(?:[eE][+-]?\d+)?"
)


class _Scanner:
    """Character-level emulation of ``std::istringstream`` extraction.

    The oracle reads ``.poly`` records with chained ``iss >> value`` calls.
    Once one extraction fails the stream's failbit is set, and every later
    extraction yields zero; a partial match such as ``"4.5"`` read as an
    ``int`` consumes ``"4"`` and leaves ``".5"`` for the next field.  Both
    behaviours are observable in the shipped models, so they are reproduced
    here rather than approximated with ``str.split``.
    """

    __slots__ = ("text", "pos", "failed")

    def __init__(self, text: str):
        self.text = text or ""
        self.pos = 0
        self.failed = False

    def _skip_space(self) -> None:
        text = self.text
        while self.pos < len(text) and text[self.pos].isspace():
            self.pos += 1

    def _match(self, pattern):
        if self.failed:
            return 0
        self._skip_space()
        match = pattern.match(self.text, self.pos)
        if match is None or match.end() == self.pos:
            self.failed = True
            return 0
        self.pos = match.end()
        return match.group(0)

    def read_int(self) -> int:
        value = self._match(_INT_PREFIX)
        return 0 if value == 0 else int(value)

    def read_float(self) -> float:
        value = self._match(_FLOAT_PREFIX)
        return 0.0 if value == 0 else float(value)


class _LineStream:
    """Sequential line reader matching ``getline`` + the oracle's skip loop."""

    __slots__ = ("lines", "index")

    def __init__(self, lines: List[str]):
        self.lines = lines
        self.index = 0

    def data_line(self) -> str:
        """Return the next non-empty, non-``#`` line, or ``""`` at EOF.

        ``getline`` consumes the line it stops on, and at EOF leaves the
        buffer empty, so a truncated file reads as zeros rather than raising.
        """
        while self.index < len(self.lines):
            line = self.lines[self.index]
            self.index += 1
            if line and line[0] != "#":
                return line
        return ""


def read_poly_plc(filename) -> PolyGeometry:
    """Read a ``.poly`` file the way ``read_poly_for_triangle`` does.

    Returns a :class:`~fes.mesh.triangle_backend.PolyGeometry`.  Segment
    indices remain **1-based** (the oracle does not enable Triangle's ``z``
    switch) and a missing region ``max_area`` becomes ``0.0``.
    """
    with open(filename, "r", encoding="utf-8", errors="replace") as handle:
        lines = handle.read().splitlines()

    stream = _LineStream(lines)

    scanner = _Scanner(stream.data_line())
    number_of_points = scanner.read_int()
    scanner.read_int()  # dimension; the oracle ignores it here
    scanner.read_int()  # number of point attributes (never read)
    number_of_markers = scanner.read_int()

    points = np.zeros((number_of_points, 2))
    point_markers = (
        np.zeros(number_of_points, dtype=int) if number_of_markers else None
    )
    for index in range(number_of_points):
        scanner = _Scanner(stream.data_line())
        scanner.read_int()  # node index (unused)
        points[index, 0] = scanner.read_float()
        points[index, 1] = scanner.read_float()
        scanner.read_float()  # z; discarded, as in the oracle
        if number_of_markers:
            point_markers[index] = scanner.read_int()

    scanner = _Scanner(stream.data_line())
    number_of_segments = scanner.read_int()
    segment_markers_present = scanner.read_int()

    segments = np.zeros((number_of_segments, 2), dtype=int)
    segment_markers = (
        np.zeros(number_of_segments, dtype=int) if segment_markers_present else None
    )
    for index in range(number_of_segments):
        scanner = _Scanner(stream.data_line())
        scanner.read_int()  # segment index (unused)
        segments[index, 0] = scanner.read_int()
        segments[index, 1] = scanner.read_int()
        if segment_markers_present:
            segment_markers[index] = scanner.read_int()

    scanner = _Scanner(stream.data_line())
    number_of_holes = scanner.read_int()
    holes = np.zeros((number_of_holes, 2))
    for index in range(number_of_holes):
        scanner = _Scanner(stream.data_line())
        holes[index, 0] = scanner.read_float()
        holes[index, 1] = scanner.read_float()

    scanner = _Scanner(stream.data_line())
    number_of_regions = scanner.read_int()
    regions = np.zeros((number_of_regions, 4))
    for index in range(number_of_regions):
        scanner = _Scanner(stream.data_line())
        scanner.read_int()  # region index (unused)
        regions[index, 0] = scanner.read_float()
        regions[index, 1] = scanner.read_float()
        regions[index, 2] = scanner.read_float()
        regions[index, 3] = scanner.read_float()

    return PolyGeometry(
        points=points,
        point_markers=point_markers,
        segments=segments if number_of_segments > 0 else None,
        segment_markers=segment_markers,
        holes=holes if number_of_holes > 0 else None,
        regions=regions if number_of_regions > 0 else None,
    )


def write_poly(
    filename,
    corners,
    segments,
    holes: Optional[list] = None,
    regions: Optional[list] = None,
):
    """Write a ``.poly`` file for the Triangle mesh generator.

    Port of ``IOwPoly.m`` (simplified version).

    Parameters
    ----------
    filename : str
        Output filename (without ``.poly``).
    corners : ndarray, shape (n, 2)
        Vertex coordinates.
    segments : ndarray, shape (n, 2)
        Edge connectivity (0-based indices).
    holes : list of ndarray, optional
        Hole coordinates.
    regions : list of tuple, optional
        Region attributes: ``(x, y, attribute, max_area)``.
    """
    with open(filename, "w") as handle:
        number_of_vertices = len(corners)
        handle.write(f"{number_of_vertices} 2 0 1\n")
        for index, (x, y) in enumerate(corners, 1):
            handle.write(f"{index} {x:.15g} {y:.15g} 0 0\n")
        handle.write(f"{len(segments)} 1\n")
        for index, (a, b) in enumerate(segments, 1):
            handle.write(f"{index} {a + 1} {b + 1} 0\n")
        if holes and len(holes) > 0:
            handle.write(f"{len(holes)}\n")
            for index, hole in enumerate(holes):
                handle.write(f"{index + 1} {hole[0]:.15g} {hole[1]:.15g}\n")
        else:
            handle.write("0\n")
        if regions and len(regions) > 0:
            handle.write(f"{len(regions)}\n")
            for index, (x, y, attribute, area) in enumerate(regions, 1):
                handle.write(f"{index} {x:.15g} {y:.15g} {attribute} {area:.15g}\n")
        else:
            handle.write("0\n")
