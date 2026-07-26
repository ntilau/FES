"""pyVista-based visualization of FE solutions.

Replaces the IOwVTK.m / IOwVTKH.m VTK file writer approach with
in-process rendering via pyVista.
"""

import numpy as np
import pyvista as pv


def _build_grid(mesh):
    """Build a pyVista UnstructuredGrid from the refined FE mesh.

    Parameters
    ----------
    mesh : dict
        Must contain 'refNode' (N, 3) and 'refEle' (M, 3) arrays.

    Returns
    -------
    pyvista.UnstructuredGrid
    """
    ref_node = mesh["refNode"]
    ref_ele = mesh["refEle"]

    # pyVista expects cells as flat array: [n_pts, i, j, k, n_pts, i, j, k, ...]
    n_cells = ref_ele.shape[0]
    cells = np.column_stack([
        np.full(n_cells, 3, dtype=np.int64),
        ref_ele.astype(np.int64),
    ]).ravel()
    cell_types = np.full(n_cells, pv.CellType.TRIANGLE, dtype=np.uint8)

    # Ensure 3D points
    if ref_node.shape[1] == 2:
        points = np.column_stack([ref_node[:, :2], np.zeros(len(ref_node))])
    else:
        points = ref_node.copy()

    return pv.UnstructuredGrid(cells, cell_types, points)


def plot_field(sys, mesh, field="u", component="abs", cmap="jet",
               show_edges=True, show=True, notebook=None, **kwargs):
    """Plot a scalar field on the FE mesh using pyVista.

    Parameters
    ----------
    sys : dict
        System data containing the field array.
    mesh : dict
        Mesh data with 'refNode' and 'refEle'.
    field : str
        Key in ``sys`` for the solution array.
    component : {"abs", "real", "imag"}
        Which component to plot when the field is complex.
    cmap : str
        Matplotlib colormap name.
    show_edges : bool
        Whether to draw mesh edges.
    show : bool
        Whether to display the plot immediately.
    notebook : bool or None
        Whether to use notebook plotting (None = auto-detect).
    **kwargs
        Additional keyword arguments passed to ``plot()``.

    Returns
    -------
    pyvista.UnstructuredGrid
        The mesh with field data attached (for further use).
    """
    u = sys[field]
    grid = _build_grid(mesh)

    if np.iscomplexobj(u):
        if component == "abs":
            data = np.abs(u)
        elif component == "real":
            data = u.real
        elif component == "imag":
            data = u.imag
        else:
            raise ValueError(f"Unknown component: {component}")
    else:
        data = u

    label = f"{field}_{component}" if np.iscomplexobj(u) else field
    grid.point_data[label] = data

    plotter = pv.Plotter(notebook=notebook)
    plotter.add_mesh(grid, cmap=cmap, show_edges=show_edges,
                     scalar_bar_args={"title": label}, **kwargs)
    if show:
        plotter.show()

    return grid


def plot_electric_field(sys, mesh, component="abs", **kwargs):
    """Convenience wrapper around :func:`plot_field`."""
    return plot_field(sys, mesh, field="u", component=component, **kwargs)


def plot_magnetic_field(sys, mesh, **kwargs):
    """Plot the H field as vectors on the mesh.

    Requires ``sys["H"]`` to be set (M, 3) array.

    Parameters
    ----------
    sys : dict
        Must contain ``H`` (N, 3) magnetic field vectors.
    mesh : dict
        Mesh data.
    **kwargs
        Passed to ``plot()``.
    """
    H = sys.get("H")
    if H is None:
        raise KeyError("sys['H'] is not set. Compute it first.")

    grid = _build_grid(mesh)
    grid.point_data["H"] = H

    plotter = pv.Plotter()
    plotter.add_mesh(grid, style="wireframe", color="grey", opacity=0.3)
    plotter.add_arrows(grid.points, H, mag=0.5)
    plotter.show()
    return grid


def plot_mesh(mesh_pv, **kwargs):
    """Display a previously returned pyVista grid.

    Parameters
    ----------
    mesh_pv : pyvista.UnstructuredGrid
        Grid returned by :func:`plot_field`.
    **kwargs
        Passed to ``plot()``.
    """
    plotter = pv.Plotter()
    plotter.add_mesh(mesh_pv, **kwargs)
    plotter.show()
