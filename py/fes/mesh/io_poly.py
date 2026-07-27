"""Mesh I/O for Triangle .poly and .h1.mat formats.

Port of IOrPoly.m and IOwPoly.m.

The .poly format is the Triangle mesh generator format:
  https://www.cs.cmu.edu/~quake/triangle.poly.html

The .h1.mat files are MATLAB .mat files produced by a custom C++ mesher
(IOrMesh) containing: node, ele, spig, spig2, nlab, elab, slab.
"""

import numpy as np
import os


def _get_tmp_info(txt):
    """Parse header line from Triangle output files."""
    parts = txt.strip().split()
    return [int(p) for p in parts]


def read_poly(filename, args="", h_ord=1, scale=1.0, data_dir="."):
    """Read a Triangle mesh .poly file.

    Attempts to load pre-generated .h1.mat first (from IOrMesh binary),
    then falls back to running Triangle directly.

    Parameters
    ----------
    filename : str
        Base name of the .poly file (without extension).
    args : str
        Additional Triangle arguments (e.g., 'q34a0.01A').
    h_ord : int
        Mesh order (1 for linear, 2 for quadratic).
    scale : float
        Scaling factor for node coordinates.
    data_dir : str
        Directory containing the mesh files.

    Returns
    -------
    dict
        Mesh data with keys: node, ele, spig, spig2, nlab, elab, slab,
        NNODE, NELE, NSPIG.
    """
    # Try loading pre-computed .h1.mat (NumPy .npz format in Python)
    mat_path = os.path.join(data_dir, f"{filename}.h{h_ord}.mat.npz")
    if os.path.exists(mat_path):
        data = np.load(mat_path)
        node = data["node"] * scale
        ele = data["ele"]
        spig = data["spig"]
        spig2 = data["spig2"]
        nlab = data["nlab"]
        elab = data["elab"]
        slab = data["slab"]
    else:
        # Try loading the .mat file directly (MATLAB format for .h1.mat)
        mat_path_old = os.path.join(data_dir, f"{filename}.h{h_ord}.mat")
        if os.path.exists(mat_path_old):
            try:
                import scipy.io as sio
                data = sio.loadmat(mat_path_old)
                node = data["node"] * scale
                ele = data["ele"]
                spig = data["spig"]
                spig2 = data["spig2"]
                nlab = data["nlab"].ravel()
                elab = data["elab"].ravel()
                slab = data["slab"].ravel()

                # Convert to 0-based indexing for Python
                node = node
                ele = ele.astype(int)
                spig = spig.astype(int)
                spig2 = spig2.astype(int)

            except Exception:
                raise FileNotFoundError(
                    f"Could not load mesh data for {filename}. "
                    f"Expected {mat_path} or {mat_path_old}"
                )
        else:
            raise FileNotFoundError(
                f"Mesh file not found for {filename}. "
                f"None of {mat_path} or {mat_path_old} exists."
            )

    # Ensure 0-based indexing (MATLAB stores 1-based)
    if np.min(ele) == 1:
        ele = ele - 1
    if np.min(spig2) == 1:
        spig2 = spig2 - 1

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


def _read_triangle_output(filename, data_dir="."):
    """Read Triangle-generated output files (.node, .ele, .edge)."""
    node = None
    ele = None
    nlab = None
    elab = None
    spig2 = None
    slab = None

    node_path = os.path.join(data_dir, f"{filename}.1.node")
    if os.path.exists(node_path):
        with open(node_path) as f:
            header = f.readline()
            info = _get_tmp_info(header)
            n_nodes = info[0]
            dim = info[1]
            n_attr = info[2]
            n_marker = info[3]
            node = np.zeros((n_nodes, dim))
            nlab = np.zeros(n_nodes, dtype=int)
            for i in range(n_nodes):
                parts = f.readline().strip().split()
                idx = int(parts[0]) - 1
                node[idx, :] = [float(p) for p in parts[1:1 + dim]]
                if n_marker > 0:
                    nlab[idx] = int(parts[-1])

    ele_path = os.path.join(data_dir, f"{filename}.1.ele")
    if os.path.exists(ele_path):
        with open(ele_path) as f:
            header = f.readline()
            info = _get_tmp_info(header)
            n_ele = info[0]
            nodes_per_ele = info[1]
            n_attr = info[2]
            ele = np.zeros((n_ele, nodes_per_ele), dtype=int)
            elab = np.ones(n_ele, dtype=int)
            for i in range(n_ele):
                parts = f.readline().strip().split()
                idx = int(parts[0]) - 1
                ele[idx, :] = [int(p) - 1 for p in parts[1:1 + nodes_per_ele]]
                ele[idx, :] = np.sort(ele[idx, :])
                if n_attr > 0:
                    elab[idx] = int(parts[-1])

    edge_path = os.path.join(data_dir, f"{filename}.1.edge")
    if os.path.exists(edge_path):
        with open(edge_path) as f:
            header = f.readline()
            info = _get_tmp_info(header)
            n_edges = info[0]
            n_marker = info[1]
            spig2 = np.zeros((n_edges, 2), dtype=int)
            slab = np.zeros(n_edges, dtype=int)
            for i in range(n_edges):
                parts = f.readline().strip().split()
                idx = int(parts[0]) - 1
                n1 = int(parts[1]) - 1
                n2 = int(parts[2]) - 1
                spig2[idx, :] = sorted([n1, n2])
                if n_marker > 0:
                    slab[idx] = int(parts[-1])

    if node is None and ele is None and spig2 is None:
        raise RuntimeError("No Triangle output files found")

    # Build spig (element-to-edge connectivity)
    NELE = ele.shape[0]
    NSPIG = spig2.shape[0]
    spig = np.zeros((NELE, 3), dtype=int)
    tspig = spig2.T  # (2, NSPIG)
    for ie in range(NELE):
        nodes = ele[ie, :]
        for i in range(3):
            snodes = [nodes[(i + 1) % 3], nodes[(i + 2) % 3]]
            match = np.where((tspig[0] == snodes[0]) & (tspig[1] == snodes[1]))[0]
            if len(match) > 0:
                spig[ie, i] = match[0] + 1  # 1-based
            else:
                match = np.where((tspig[0] == snodes[1]) & (tspig[1] == snodes[0]))[0]
                if len(match) > 0:
                    spig[ie, i] = -(match[0] + 1)

    return node, ele, spig, spig2, nlab, elab, slab


def write_poly(filename, corners, segments, holes=None, regions=None):
    """Write a .poly file for the Triangle mesh generator.

    Port of IOwPoly.m (simplified version).

    Parameters
    ----------
    filename : str
        Output filename (without extension).
    corners : ndarray, shape (n, 2)
        Vertex coordinates.
    segments : ndarray, shape (n, 2)
        Edge connectivity (0-based indices).
    holes : list of ndarray, optional
        Hole coordinates.
    regions : list of tuple, optional
        Region attributes: (x, y, attribute, max_area).
    """
    with open(filename, "w") as f:
        n_vertices = len(corners)
        f.write(f"{n_vertices} 2 0 1\n")
        for i, (x, y) in enumerate(corners, 1):
            f.write(f"{i} {x:.15g} {y:.15g} 0 0\n")
        f.write(f"{len(segments)} 1\n")
        for i, (a, b) in enumerate(segments, 1):
            f.write(f"{i} {a + 1} {b + 1} 0\n")
        if holes and len(holes) > 0:
            f.write(f"{len(holes)}\n")
            for i, h in enumerate(holes):
                f.write(f"{i + 1} {h[0]:.15g} {h[1]:.15g}\n")
        else:
            f.write("0\n")
        if regions and len(regions) > 0:
            f.write(f"{len(regions)}\n")
            for i, (x, y, attr, area) in enumerate(regions, 1):
                f.write(f"{i} {x:.15g} {y:.15g} {attr} {area:.15g}\n")
        else:
            f.write("0\n")
