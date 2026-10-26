"""Extract H(curl) tetrahedral master matrices to NumPy format.

Parses the C++ source files in the external reference that contain
hardcoded master element matrices for H(curl) tetrahedra (degrees 1-4).

Outputs:
  tools/master_matrices/  — one .npz per block (stiffness S**, mass T**)
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

# Root of the external reference C++ source containing the master matrices.
REFERENCE_ROOT = Path("/Users/ntilau/Downloads/LTE/source/fem/lib/FE_Matrix/source")
OUT_DIR = Path(__file__).resolve().parent / "master_matrices"

#: (block name, source file, symmetric) for stiffness blocks
STIFFNESS_BLOCKS = [
    ("S00", "TetraHcurlMasterStiffnessMatrix00.cpp", True),
    ("S01", "TetraHcurlMasterStiffnessMatrix01.cpp", True),
    ("S02", "TetraHcurlMasterStiffnessMatrix02.cpp", True),
    ("S11", "TetraHcurlMasterStiffnessMatrixS11.cpp", True),
    ("S12", "TetraHcurlMasterStiffnessMatrixS12.cpp", True),
    ("S22", "TetraHcurlMasterStiffnessMatrixS22.cpp", True),
]

#: (block name, source file, symmetric) for mass blocks
MASS_BLOCKS = [
    ("T11", "TetraHcurlMasterMassMatrixT11.cpp", True),
    ("T12", "TetraHcurlMasterMassMatrixT12.cpp", True),
    ("T13", "TetraHcurlMasterMassMatrixT13.cpp", True),
    ("T22", "TetraHcurlMasterMassMatrixT22.cpp", True),
    ("T23", "TetraHcurlMasterMassMatrixT23.cpp", True),
    ("T33", "TetraHcurlMasterMassMatrixT33.cpp", True),
]

#: H(curl) tetra DOF count per degree: 6*n + 4*n*(n-1) + n*(n-1)*(n-2)/2
def hcurl_tetra_dim(degree: int) -> int:
    n = degree
    return 6 * n + 4 * n * (n - 1) + n * (n - 1) * (n - 2) // 2


#: Match lines like: s00[ 1][ 2] =   -2.0/      3.0;
_ASSIGN_RE = re.compile(
    r"[sm]\w+\[\s*(\d+)\s*\]\[\s*(\d+)\s*\]\s*=\s*([^;]+);"
)


def _eval_rational(expr: str) -> float:
    """Evaluate a rational expression like ``2.0/3.0`` or ``16.0/315.0``."""
    expr = expr.strip().replace(" ", "")
    return float(eval(expr, {"__builtins__": {}}))


def _parse_block(file_path: Path) -> dict[int, list[tuple[int, int, float]]]:
    """Parse one master matrix block file.

    Returns {degree: [(i, j, value), ...]} using 0-based indices.
    """
    text = file_path.read_text()
    assignments: dict[int, list[tuple[int, int, float]]] = {1: [], 2: [], 3: [], 4: []}

    # Determine if indexing is 0-based or 1-based by looking at the first index used.
    first_indices: list[int] = []
    for m in _ASSIGN_RE.finditer(text):
        expr = m.group(3).strip().replace(" ", "")
        if "{" in expr or "}" in expr:
            continue
        first_indices.append(int(m.group(1)))
    offset = 0 if first_indices and min(first_indices) == 0 else 1

    for m in _ASSIGN_RE.finditer(text):
        expr = m.group(3).strip().replace(" ", "")
        if "{" in expr or "}" in expr:
            continue  # skip array initialization line
        i = int(m.group(1)) - offset
        j = int(m.group(2)) - offset
        val = _eval_rational(expr)
        # The same array is used for all degrees; entries beyond the degree's
        # dimension are just unused.  We record them all and slice per degree.
        for deg in (1, 2, 3, 4):
            dim = hcurl_tetra_dim(deg)
            if i < dim and j < dim:
                assignments[deg].append((i, j, val))

    return assignments


def _build_matrix(assignments: list[tuple[int, int, float]], dim: int) -> np.ndarray:
    """Build a symmetric dim×dim matrix from (i, j, value) triplets."""
    M = np.zeros((dim, dim), dtype=float)
    for i, j, val in assignments:
        M[i, j] = val
        if i != j:
            M[j, i] = val
    return M


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_blocks = [
        ("stiffness", STIFFNESS_BLOCKS),
        ("mass", MASS_BLOCKS),
    ]

    for family, blocks in all_blocks:
        for block_name, fname, _symmetric in blocks:
            src = REFERENCE_ROOT / fname
            if not src.exists():
                print(f"WARNING: {src} not found, skipping")
                continue
            assignments_by_deg = _parse_block(src)
            for deg in (1, 2, 3, 4):
                dim = hcurl_tetra_dim(deg)
                M = _build_matrix(assignments_by_deg[deg], dim)
                out = OUT_DIR / f"{family}_{block_name}_deg{deg}.npz"
                np.savez(out, matrix=M, degree=deg, dim=dim)
                nnz = int(np.count_nonzero(M))
                print(f"  {out}: {dim}x{dim}, {nnz} nonzeros")

    print(f"\nAll matrices written to {OUT_DIR}")


if __name__ == "__main__":
    main()
