"""Parse the header and custom sections of FES ``.poly`` files.

This is the metadata half of the C++ preprocessor: it reads the node header,
standard TetGen/Triangle section counts, the ``#Formula`` tag, and the
``#Solids``/``#Regions`` and ``#Boundaries`` tables.  It does not mesh.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

_FORMULA_RE = re.compile(r"^#\s*Formula\s+(\S+)", re.IGNORECASE)
_SOLIDS_RE = re.compile(r"^#\s*(?:Solids|Regions)\s+(\d+)\s*$", re.IGNORECASE)
_BOUNDARIES_RE = re.compile(r"^#\s*Boundaries\s+(\d+)\s*$", re.IGNORECASE)
_NODES_COMMENT_RE = re.compile(r"^#\s*NODES\s*:\s*(\d+)\s+(\d+)", re.IGNORECASE)
_SEGMENTS_COMMENT_RE = re.compile(r"^#\s*SEGMENTS\s*:\s*(\d+)", re.IGNORECASE)
_FACES_COMMENT_RE = re.compile(r"^#\s*(?:FACES|FACETS)\s*:\s*(\d+)", re.IGNORECASE)
_HOLES_COMMENT_RE = re.compile(r"^#\s*HOLES\s*:\s*(\d+)", re.IGNORECASE)
_REGIONS_COMMENT_RE = re.compile(r"^#\s*REGIONS\s*:\s*(\d+)", re.IGNORECASE)

_BC_TYPES = {
    "perfect_e": "PerfectE",
    "perfecte": "PerfectE",
    "perfect_h": "PerfectH",
    "perfecth": "PerfectH",
    "radiation": "Radiation",
    "wave_port": "WavePort",
    "waveport": "WavePort",
    "lumped_port": "LumpedPort",
    "lumpedport": "LumpedPort",
    "const_e": "ConstE",
    "conste": "ConstE",
    "const_h": "ConstH",
    "consth": "ConstH",
}


def _canonical_bc_type(tag: str) -> str:
    return _BC_TYPES.get(tag.strip().lower(), tag.strip())


@dataclass
class Solid:
    name: str
    epsr: float = 1.0
    mur: float = 1.0
    sigma: float = 0.0
    tand: float = 0.0
    material: str = ""
    #: 0-based line index inside ``#Solids``.  The C++ preprocessor uses this
    #: value as the material's ``label``, while elements carry the regional
    #: attribute; the two coincide only when the file lists one solid per
    #: region in order.
    label: int = 0

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "epsr": self.epsr,
            "mur": self.mur,
            "sigma": self.sigma,
            "tand": self.tand,
            "material": self.material,
            "label": self.label,
        }


@dataclass
class Boundary:
    name: str
    label: int = 0
    type: str = ""
    #: ``bc::bc()`` initialises this to 1, and the C++ parser only assigns it
    #: when the ``#Boundaries`` line carries a fourth field.
    num_modes: int = 1
    #: For ``ConstE`` boundaries: the specified tangential E field (Ex, Ey, Ez).
    e_value: Optional[Tuple[float, float, float]] = None
    #: For ``ConstH`` boundaries: the specified tangential H field (Hx, Hy, Hz).
    h_value: Optional[Tuple[float, float, float]] = None
    #: For ``ConstH`` boundaries: the outward normal vector (nx, ny, nz).
    normal_vec: Optional[Tuple[float, float, float]] = None

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "label": self.label,
            "type": self.type,
            "num_modes": self.num_modes,
            "e_value": self.e_value,
            "h_value": self.h_value,
            "normal_vec": self.normal_vec,
        }


@dataclass
class ModelInfo:
    name: str
    path: str
    dimension: int = 0
    formula: Optional[str] = None
    n_nodes: int = 0
    n_segments: int = 0
    n_faces: int = 0
    n_holes: int = 0
    n_regions: int = 0
    solids: List[Solid] = field(default_factory=list)
    boundaries: List[Boundary] = field(default_factory=list)
    bbox: Optional[Tuple[Tuple[float, ...], Tuple[float, ...]]] = None

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "path": self.path,
            "dimension": self.dimension,
            "formula": self.formula,
            "n_nodes": self.n_nodes,
            "n_segments": self.n_segments,
            "n_faces": self.n_faces,
            "n_holes": self.n_holes,
            "n_regions": self.n_regions,
            "solids": [s.as_dict() for s in self.solids],
            "boundaries": [b.as_dict() for b in self.boundaries],
            "bbox": None if self.bbox is None else [list(self.bbox[0]), list(self.bbox[1])],
        }


def _first_data_line(lines: List[str]) -> int:
    for index, line in enumerate(lines):
        if line.strip() and not line.lstrip().startswith("#"):
            return index
    return -1


def read_poly_metadata(filename: str | Path) -> ModelInfo:
    """Read model metadata from *filename*.

    Only the header, the standard section counts, and the custom tables are
    parsed; node coordinates are scanned solely to compute the bounding box.
    """
    path = Path(filename)
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    info = ModelInfo(name=path.stem, path=str(path))

    # Custom tables and comment-style counts can appear anywhere.
    index = 0
    while index < len(lines):
        line = lines[index].strip()

        match = _FORMULA_RE.match(line)
        if match and info.formula is None:
            info.formula = match.group(1).lower()

        match = _SOLIDS_RE.match(line)
        if match:
            count = int(match.group(1))
            for offset in range(1, count + 1):
                if index + offset >= len(lines):
                    break
                parts = lines[index + offset].split()
                if not parts:
                    continue
                values = parts[1:]
                def number(position: int, default: float) -> float:
                    if position < len(values):
                        try:
                            return float(values[position])
                        except ValueError:
                            return default
                    return default
                info.solids.append(
                    Solid(
                        name=parts[0],
                        epsr=number(0, 1.0),
                        mur=number(1, 1.0),
                        sigma=number(2, 0.0),
                        tand=number(3, 0.0),
                        material=parts[5] if len(parts) > 5 else (parts[-1] if len(parts) > 1 else ""),
                        label=len(info.solids),
                    )
                )
            index += count + 1
            continue

        match = _BOUNDARIES_RE.match(line)
        if match:
            count = int(match.group(1))
            for offset in range(1, count + 1):
                if index + offset >= len(lines):
                    break
                parts = lines[index + offset].split()
                if not parts:
                    continue
                boundary = Boundary(name=parts[0])
                if len(parts) > 1:
                    try:
                        boundary.label = int(parts[1])
                    except ValueError:
                        boundary.label = 0
                if len(parts) > 2:
                    boundary.type = _canonical_bc_type(parts[2])
                if len(parts) > 3:
                    try:
                        boundary.num_modes = int(parts[3])
                    except ValueError:
                        pass
                # Parse optional field values for ConstE/ConstH boundaries.
                if boundary.type == "ConstE" and len(parts) >= 7:
                    try:
                        boundary.e_value = (
                            float(parts[4]), float(parts[5]), float(parts[6]),
                        )
                    except ValueError:
                        pass
                elif boundary.type == "ConstH" and len(parts) >= 10:
                    try:
                        boundary.h_value = (
                            float(parts[4]), float(parts[5]), float(parts[6]),
                        )
                        boundary.normal_vec = (
                            float(parts[7]), float(parts[8]), float(parts[9]),
                        )
                    except ValueError:
                        pass
                info.boundaries.append(boundary)
            index += count + 1
            continue

        index += 1

    # Standard section counts; the nodes header also carries the dimension.
    for regex, attr in (
        (_NODES_COMMENT_RE, "n_nodes"),
        (_SEGMENTS_COMMENT_RE, "n_segments"),
        (_FACES_COMMENT_RE, "n_faces"),
        (_HOLES_COMMENT_RE, "n_holes"),
        (_REGIONS_COMMENT_RE, "n_regions"),
    ):
        for line in lines:
            match = regex.match(line.strip())
            if match and not getattr(info, attr):
                setattr(info, attr, int(match.group(1)))
                if attr == "n_nodes":
                    info.dimension = int(match.group(2))
                break

    # Files without comment headers still start with a bare count line.
    first = _first_data_line(lines)
    if first >= 0:
        header = lines[first].split()
        if len(header) >= 2:
            try:
                count, dimension = int(header[0]), int(header[1])
            except ValueError:
                count, dimension = 0, 0
            if count > 0 and dimension in (2, 3):
                if not info.n_nodes:
                    info.n_nodes = count
                if not info.dimension:
                    info.dimension = dimension
                coords = []
                for line in lines[first + 1 : first + 1 + count]:
                    parts = line.split()
                    if len(parts) >= 1 + dimension:
                        try:
                            coords.append(tuple(float(v) for v in parts[1 : 1 + dimension]))
                        except ValueError:
                            pass
                if coords:
                    info.bbox = (
                        tuple(min(c[i] for c in coords) for i in range(dimension)),
                        tuple(max(c[i] for c in coords) for i in range(dimension)),
                    )

                # The next bare count line after the node block is the
                # segment (2D) or facet (3D) header.
                cursor = first + 1 + count
                while cursor < len(lines) and (
                    not lines[cursor].strip() or lines[cursor].lstrip().startswith("#")
                ):
                    cursor += 1
                if cursor < len(lines):
                    parts = lines[cursor].split()
                    if parts:
                        try:
                            section_count = int(parts[0])
                        except ValueError:
                            section_count = -1
                        if section_count >= 0:
                            if dimension == 2 and not info.n_segments:
                                info.n_segments = section_count
                            elif dimension == 3 and not info.n_faces:
                                info.n_faces = section_count

                # Hole count and region count are the next single-integer
                # lines after the segment/facet block.  TetGen facet records
                # span two lines, so search for the counts rather than
                # assuming one line per entry.
                def next_single_int(start: int) -> Tuple[int, int]:
                    for position in range(start, len(lines)):
                        stripped = lines[position].strip()
                        if not stripped or stripped.startswith("#"):
                            continue
                        fields = stripped.split()
                        if len(fields) == 1:
                            try:
                                return int(fields[0]), position
                            except ValueError:
                                continue
                    return -1, -1

                holes, holes_at = next_single_int(cursor + 1)
                if holes >= 0:
                    if not info.n_holes:
                        info.n_holes = holes
                    regions, _ = next_single_int(holes_at + 1 + holes)
                    if regions >= 0 and not info.n_regions:
                        info.n_regions = regions

    return info


__all__ = ["Boundary", "ModelInfo", "Solid", "read_poly_metadata"]
