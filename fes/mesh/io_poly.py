"""Mesh I/O for Triangle ``.poly`` geometry and ``.npz`` caches.

The ``.poly`` format is the Triangle mesh generator format:
  https://www.cs.cmu.edu/~quake/triangle.poly.html

The ``.npz`` caches hold the arrays the legacy ``IOrMesh`` tool wrote
(``node``, ``ele``, ``spig``, ``spig2``, ``nlab``, ``elab``, ``slab``).  They
are read and written by :mod:`fes.mesh.io_mat`; :mod:`fes.mesh.topology`
rebuilds them from a Triangle result, so a model without a cache is meshed on
the fly instead of requiring the C tool.

Geometry parsing lives in :mod:`fes.mesh.geometry` and is re-exported here for
backwards compatibility.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .geometry import read_poly_plc, write_poly
from .io_mat import cache_paths, read_cache, write_cache
from .topology import mesh_arrays_from_poly

__all__ = [
    "read_poly",
    "read_poly_plc",
    "write_poly",
]


def _cached_or_meshed(
    filename: str,
    args: str,
    h_ord: int,
    data_dir,
    remesh: bool,
    persist: bool,
):
    """Load the cache for *filename*, or mesh its ``.poly`` when absent."""
    npz_path = cache_paths(data_dir, filename, h_ord)
    has_cache = npz_path.is_file()
    if has_cache and not remesh:
        # A corrupt cache is an error: silently remeshing would hide the
        # problem and change every downstream result.
        return read_cache(data_dir, filename, h_ord)

    poly_path = Path(data_dir) / f"{filename}.poly"
    if not poly_path.is_file():
        raise FileNotFoundError(
            f"Mesh file not found for {filename}. None of {npz_path} "
            f"or {poly_path} exists."
        )
    if h_ord != 1:
        raise NotImplementedError(
            f"h-ord {h_ord} requires a pre-generated {npz_path.name}; "
            "automatic meshing writes linear meshes only"
        )

    # Reproduce ``triangle -p<args>DeQ``: the tool's default switch set is
    # ``q34A``, and ``D``/``e`` are mandatory.  ``triangulate_plc`` appends
    # ``AfeQ`` on top.
    switches = (args or "q34A") + "D"
    arrays = mesh_arrays_from_poly(poly_path, switches=switches)
    if persist:
        write_cache(npz_path, **arrays)
    return arrays


def read_poly(filename, args="", h_ord=1, scale=1.0, data_dir=".", *,
              remesh=False, persist=False):
    """Read a Triangle mesh for *filename*.

    A pre-generated cache (``<name>.npz``) short-circuits meshing.  When it does
    not exist, the ``<name>.poly`` file in *data_dir* is meshed with Triangle,
    reproducing the arrays the legacy ``IOrMesh`` tool wrote.

    Parameters
    ----------
    filename : str
        Base name of the model (without extension).
    args : str
        Additional Triangle switches, in ``IOrMesh`` syntax (``q34A`` when
        empty, ``D`` and ``e`` are always added).
    h_ord : int
        Mesh order.  Automatic meshing supports ``1`` only.
    scale : float
        Scaling factor applied to node coordinates.
    data_dir : str or Path
        Directory containing the mesh and ``.poly`` files.
    remesh : bool
        Ignore an existing cache and mesh from ``.poly``.
    persist : bool
        Write the generated mesh to ``<name>.npz`` in *data_dir*.

    Returns
    -------
    dict
        Mesh data with keys: node, ele, spig, spig2, nlab, elab, slab,
        NNODE, NELE, NSPIG.
    """
    arrays = _cached_or_meshed(
        filename, args, h_ord, data_dir, remesh, persist
    )
    node = arrays["node"] * scale
    ele = arrays["ele"]
    spig = arrays["spig"]
    spig2 = arrays["spig2"]
    nlab = arrays["nlab"]
    elab = arrays["elab"]
    slab = arrays["slab"]

    mesh = {
        "node": node,
        "ele": ele,
        "spig": spig,
        "spig2": spig2,
        "nlab": nlab.ravel() if isinstance(nlab, np.ndarray) else np.array(nlab).ravel(),
        "elab": elab.ravel() if isinstance(elab, np.ndarray) else np.array(elab).ravel(),
        "slab": slab.ravel() if isinstance(slab, np.ndarray) else np.array(slab).ravel(),
        "NNODE": node.shape[0],
        "NELE": ele.shape[0],
        "NSPIG": spig2.shape[0],
    }

    return mesh
