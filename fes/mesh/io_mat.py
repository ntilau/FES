"""Read and write ``.npz`` mesh caches.

The v0.1 ``IOrMesh`` binary wrote MATLAB v5 files holding a linear mesh plus
its boundary and region labels.  The Python port reads and writes ``.npz``
files with the same array names; the legacy MATLAB format is no longer
supported.

The arrays are:

===========  =====================================================
``node``     node coordinates, ``(n_nodes, dim)``
``ele``      triangle / element connectivity, 0-based
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
from typing import Dict, Union

import numpy as np

#: Arrays a cache must provide.
MESH_KEYS = ("node", "ele", "spig", "spig2", "nlab", "elab", "slab")


def cache_paths(
    data_dir: Union[str, Path], filename: str, h_ord: int = 1
) -> Path:
    """The ``.npz`` cache path for one model."""
    directory = Path(data_dir)
    return directory / f"{filename}.npz"


def load_cache(path: Union[str, Path]) -> Dict[str, np.ndarray]:
    """Load one ``.npz`` cache file and return its mesh arrays."""
    path = Path(path)
    with np.load(path) as data:
        arrays = {key: np.asarray(data[key]) for key in MESH_KEYS}
    for key in ("nlab", "elab", "slab"):
        arrays[key] = arrays[key].ravel()
    # Connectivity and labels are integers on disk; indexing with floats
    # fails, so normalise here.
    for key in ("ele", "spig", "spig2", "nlab", "elab", "slab"):
        arrays[key] = arrays[key].astype(np.int64)
    return arrays


def read_cache(
    data_dir: Union[str, Path],
    filename: str,
    h_ord: int = 1,
    scale: float = 1.0,
) -> Dict[str, np.ndarray]:
    """Load the ``.npz`` cache for *filename*, scaling node coordinates.

    Raises :class:`FileNotFoundError` when the cache does not exist, which is
    the behaviour callers rely on to fall back to meshing.
    """
    path = cache_paths(data_dir, filename, h_ord)
    if not path.is_file():
        raise FileNotFoundError(
            f"Mesh file not found for {filename}. {path} does not exist."
        )
    try:
        arrays = load_cache(path)
    except Exception as exc:
        raise FileNotFoundError(
            f"Could not load mesh data from {path}: {exc}"
        ) from exc
    arrays["node"] = arrays["node"] * scale
    return arrays


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
