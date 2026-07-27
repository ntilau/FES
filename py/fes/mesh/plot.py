"""Mesh visualization utilities.

Port of PlotPoly.m and PlotMesh.m.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_poly(filename, fig_handle=1, scale=1.0, data_dir="."):
    """Plot a .poly geometry file.

    Port of PlotPoly.m.

    Parameters
    ----------
    filename : str
        Base name of the .poly file.
    fig_handle : int
        Figure number.
    scale : float
        Scale factor for coordinates.
    data_dir : str
        Directory containing the .poly file.
    """
    import os

    poly_path = os.path.join(data_dir, f"{filename}.poly")
    with open(poly_path) as f:
        header = f.readline()
        info = header.strip().split()
        n_nodes = int(info[0])
        dim = int(info[1])

        node = np.zeros((n_nodes, dim))
        nlab = np.zeros(n_nodes)
        for i in range(n_nodes):
            parts = f.readline().strip().split()
            idx = int(parts[0]) - 1
            node[idx, :] = [float(p) for p in parts[1:1 + dim]]
            nlab[idx] = float(parts[-1])

        node *= scale

        header = f.readline()
        info = header.strip().split()
        n_spig = int(info[0])

        spig = np.zeros((n_spig, 2), dtype=int)
        slab = np.zeros(n_spig)
        for i in range(n_spig):
            parts = f.readline().strip().split()
            idx = int(parts[0]) - 1
            spig[idx, :] = [int(parts[1]) - 1, int(parts[2]) - 1]
            slab[idx] = float(parts[-1])

    plt.figure(fig_handle)
    for i in range(n_spig):
        pts = node[spig[i, :], :]
        plt.plot(pts[:, 0], pts[:, 1], "k-")
    plt.axis("equal")
    plt.axis("tight")


def plot_mesh(mesh, plot_type=0):
    """Plot the finite element mesh.

    Port of PlotMesh.m.

    Parameters
    ----------
    mesh : dict
        Mesh data.
    plot_type : int
        0: Show element, edge, and node numbering.
        1: Show element and edge labels.
    """
    ele = mesh["ele"]
    node = mesh["node"]
    spig2 = mesh["spig2"]

    if plot_type == 0:
        # Elements
        plt.figure(1)
        for tri in ele:
            pts = node[tri, :]
            pts = np.vstack([pts, pts[0]])
            plt.plot(pts[:, 0], pts[:, 1], "k-")
        plt.title("Elements")
        plt.axis("equal")
        plt.axis("tight")

        # Element numbers
        centers = np.column_stack([
            (node[ele[:, 0], 0] + node[ele[:, 1], 0] + node[ele[:, 2], 0]) / 3,
            (node[ele[:, 0], 1] + node[ele[:, 1], 1] + node[ele[:, 2], 1]) / 3,
        ])
        for i, (x, y) in enumerate(centers):
            plt.text(x, y, str(i + 1), color="b", fontsize=8, ha="center")

        # Edges
        plt.figure(2)
        for e in spig2:
            pts = node[e, :]
            plt.plot(pts[:, 0], pts[:, 1], "k-")
        plt.title("Edges")
        plt.axis("equal")
        plt.axis("tight")

        midpoints = np.column_stack([
            (node[spig2[:, 0], 0] + node[spig2[:, 1], 0]) / 2,
            (node[spig2[:, 0], 1] + node[spig2[:, 1], 1]) / 2,
        ])
        for i, (x, y) in enumerate(midpoints):
            plt.text(x, y, str(i + 1), color="b", fontsize=8, ha="center")

        # Nodes
        plt.figure(3)
        plt.plot(node[:, 0], node[:, 1], "b.")
        plt.title("Nodes")
        plt.axis("equal")
        plt.axis("tight")
        for i, (x, y) in enumerate(node):
            plt.text(x, y, str(i + 1), fontsize=8, ha="center")

    elif plot_type == 1:
        elab = mesh["elab"]
        nlab = mesh["nlab"]
        slab = mesh["slab"]

        # Elements with labels
        plt.figure(1)
        for tri in ele:
            pts = node[tri, :]
            pts = np.vstack([pts, pts[0]])
            plt.plot(pts[:, 0], pts[:, 1], "k-")
        plt.title("Labels")
        plt.axis("equal")
        plt.axis("tight")

        centers = np.column_stack([
            (node[ele[:, 0], 0] + node[ele[:, 1], 0] + node[ele[:, 2], 0]) / 3,
            (node[ele[:, 0], 1] + node[ele[:, 1], 1] + node[ele[:, 2], 1]) / 3,
        ])
        for i, (x, y) in enumerate(centers):
            plt.text(x, y, str(elab[i]), color="k", fontsize=8, ha="center")

        midpoints = np.column_stack([
            (node[spig2[:, 0], 0] + node[spig2[:, 1], 0]) / 2,
            (node[spig2[:, 0], 1] + node[spig2[:, 1], 1]) / 2,
        ])
        for i, (x, y) in enumerate(midpoints):
            plt.text(x, y, str(slab[i]), color="r", fontsize=8, ha="center")
