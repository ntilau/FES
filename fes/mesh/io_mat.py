"""Read and write the legacy ``.<h>h.mat`` mesh caches.

The v0.1 ``IOrMesh`` binary wrote MATLAB v5 files holding a linear mesh plus
its boundary and region labels.  The Python port writes ``.h1.mat.npz`` files
with the same array names, so both formats are accepted here.

Two conventions differ between the formats and are normalised on read:

* MATLAB stores 1-based connectivity; the arrays are shifted to 0-based when
  their smallest index is 1.
* MATLAB files load labels as 2-D row vectors, while ``.npz`` files store them
  flat.  Both are returned flat.

The arrays are:

===========  =====================================================
``node``     node coordinates, ``(n_nodes, dim)``
``ele``      triangle / element connectivity, 1- or 0-based on disk
``spig``     segment-to-edge map used by the port code
``spig2``    segment-to-edge map, second convention
``nlab``     node labels
``elab``     element labels
``slab``     segment labels
===========  =====================================================
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np

#: Arrays a cache must provide.
MESH_KEYS = ("node", "ele", "spig", "spig2", "nlab", "elab", "slab")


def cache_paths(
    data_dir: Union[str, Path], filename: str, h_ord: int = 1
) -> Tuple[Path, Path]:
    """The ``.npz`` and MATLAB cache paths for one model."""
    directory = Path(data_dir)
    return (
        directory / f"{filename}.h{h_ord}.mat.npz",
        directory / f"{filename}.h{h_ord}.mat",
    )


def load_cache(path: Union[str, Path]) -> Dict[str, np.ndarray]:
    """Load one cache file (``.npz`` or MATLAB) and return its mesh arrays."""
    path = Path(path)
    if path.suffix == ".npz":
        with np.load(path) as data:
            arrays = {key: np.asarray(data[key]) for key in MESH_KEYS}
    else:
        import scipy.io as sio

        raw = sio.loadmat(str(path))
        arrays = {key: np.asarray(raw[key]) for key in MESH_KEYS}
    for key in ("nlab", "elab", "slab"):
        arrays[key] = arrays[key].ravel()
    # Connectivity and labels are integers on disk but MATLAB may load them as
    # doubles; indexing with floats fails, so normalise here for both formats.
    for key in ("ele", "spig", "spig2", "nlab", "elab", "slab"):
        arrays[key] = arrays[key].astype(np.int64)
    # MATLAB stores 1-based connectivity; a cache that already starts at 0 is
    # left alone (the legacy loader used the same test).
    for key in ("ele", "spig2"):
        if arrays[key].size and np.min(arrays[key]) == 1:
            arrays[key] = arrays[key] - 1
    return arrays


def read_cache(
    data_dir: Union[str, Path],
    filename: str,
    h_ord: int = 1,
    scale: float = 1.0,
) -> Dict[str, np.ndarray]:
    """Load the first existing cache for *filename*, scaling node coordinates.

    Raises :class:`FileNotFoundError` when neither format exists, which is the
    legacy behaviour callers rely on to fall back to meshing.
    """
    npz_path, mat_path = cache_paths(data_dir, filename, h_ord)
    for path in (npz_path, mat_path):
        if path.is_file():
            try:
                arrays = load_cache(path)
            except Exception as exc:
                raise FileNotFoundError(
                    f"Could not load mesh data from {path}: {exc}"
                ) from exc
            arrays["node"] = arrays["node"] * scale
            return arrays
    raise FileNotFoundError(
        f"Mesh file not found for {filename}. "
        f"None of {npz_path} or {mat_path} exists."
    )


def write_cache(
    path: Union[str, Path], **arrays: np.ndarray
) -> Path:
    """Write a ``.npz`` cache; unknown keys are rejected."""
    unknown = set(arrays) - set(MESH_KEYS)
    if unknown:
        raise KeyError(f"unknown mesh arrays: {', '.join(sorted(unknown))}")
    missing = [key for key in MESH_KEYS if key not in arrays]
    if missing:
        raise KeyError(f"missing mesh arrays: {', '.join(missing)}")
    target = Path(path)
    os.makedirs(target.parent, exist_ok=True)
    np.savez(target, **{key: np.asarray(value) for key, value in arrays.items()})
    return target


__all__ = [
    "MESH_KEYS",
    "cache_paths",
    "load_cache",
    "read_cache",
    "write_cache",
]
