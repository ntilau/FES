"""Regular triangular mesh generation.

Port of BuildRegularSquare.m.
"""

import numpy as np


def _regular_triangular_mesh(nx, ny):
    """Generate a regular triangular mesh on the unit square [0,1]x[0,1].

    Parameters
    ----------
    nx : int
        Number of vertices in x-direction.
    ny : int
        Number of vertices in y-direction.

    Returns
    -------
    ele : ndarray, shape (2*(nx-1)*(ny-1), 3)
        Element connectivity (0-based indices).
    node : ndarray, shape (nx*ny, 2)
        Node coordinates.
    """
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)  # note: meshgrid returns shape (ny, nx)
    node = np.column_stack([X.ravel(), Y.ravel()])

    # Build element connectivity
    idx = np.arange(nx * ny).reshape(ny, nx)

    v1 = idx[:-1, :-1].ravel()  # top-left
    v2 = idx[:-1, 1:].ravel()   # top-right
    v3 = idx[1:, :-1].ravel()   # bottom-left
    v4 = idx[1:, 1:].ravel()    # bottom-right

    ele = np.vstack([
        np.column_stack([v1, v2, v4]),
        np.column_stack([v1, v3, v4])
    ])

    return ele, node


def build_regular_square(nptsx, nptsy):
    """Build a regular triangular mesh for the unit square.

    Port of BuildRegularSquare.m. Generates node coordinates,
    element connectivity, edge connectivity, and boundary labels.

    Parameters
    ----------
    nptsx : int
        Number of vertices in x-direction.
    nptsy : int
        Number of vertices in y-direction.

    Returns
    -------
    dict
        Mesh data with keys: node, ele, spig, spig2, nlab, elab, slab,
        NNODE, NELE, NSPIG.
    """
    ele, node = _regular_triangular_mesh(nptsx, nptsy)
    NNODE = node.shape[0]
    NELE = ele.shape[0]

    # Node labels: 1 if on boundary
    nlab = np.zeros(NNODE, dtype=int)
    on_bnd = (
        (node[:, 0] == 0) | (node[:, 0] == 1)
        | (node[:, 1] == 0) | (node[:, 1] == 1)
    )
    nlab[on_bnd] = 1

    elab = np.ones(NELE, dtype=int)

    # Build edge list and labels
    tspig = []
    slab = []
    for ie in range(NELE):
        nodes = np.sort(ele[ie, :])
        for i in range(3):
            snodes = sorted([nodes[i], nodes[(i + 1) % 3]])
            is_boundary_edge = False
            if nlab[snodes[0]] == 1 and nlab[snodes[1]] == 1:
                x0, y0 = node[snodes[0]]
                x1, y1 = node[snodes[1]]
                is_boundary_edge = ((x0 == 0 and x1 == 0) or (x0 == 1 and x1 == 1)
                                    or (y0 == 0 and y1 == 0) or (y0 == 1 and y1 == 1))
            if not tspig:
                tspig.append(snodes)
                slab.append(1 if is_boundary_edge else 0)
            else:
                found = False
                for e in tspig:
                    if e[0] == snodes[0] and e[1] == snodes[1]:
                        found = True
                        break
                    if e[0] == snodes[1] and e[1] == snodes[0]:
                        found = True
                        break
                if not found:
                    tspig.append(snodes)
                    slab.append(1 if is_boundary_edge else 0)

    spig2 = np.array(tspig, dtype=int)
    NSPIG = len(spig2)
    slab = np.array(slab, dtype=int)

    # Build element-to-edge mapping (spig)
    spig = np.zeros((NELE, 3), dtype=int)
    for ie in range(NELE):
        nodes = ele[ie, :]
        for i in range(3):
            snodes = [nodes[(i + 1) % 3], nodes[(i + 2) % 3]]
            match = np.where(
                (spig2[:, 0] == snodes[0]) & (spig2[:, 1] == snodes[1])
            )[0]
            if len(match) > 0:
                spig[ie, i] = match[0] + 1  # 1-based
            else:
                match = np.where(
                    (spig2[:, 0] == snodes[1]) & (spig2[:, 1] == snodes[0])
                )[0]
                if len(match) > 0:
                    spig[ie, i] = -(match[0] + 1)

    mesh = {
        "node": node,
        "ele": ele,
        "spig": spig,
        "spig2": spig2,
        "nlab": nlab,
        "elab": elab,
        "slab": slab,
        "NNODE": NNODE,
        "NELE": NELE,
        "NSPIG": NSPIG,
    }

    return mesh
