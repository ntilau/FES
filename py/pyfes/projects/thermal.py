"""Heat conduction simulation.

Port of ProjectThermalDistro.m, ProjectThermalDistroDG.m, and ThermalFE.m.
"""

import numpy as np
from scipy.sparse.linalg import spsolve

from ..mesh.io_poly import read_poly
from ..fem.assembly import assemble_linear
from ..post.plot import plot_field
from ._utils import apply_dirichlet_bc


def thermal_distribution(K=52, ht=750, T_edge=(300.0, 200.0), q0=100.0,
                         plot=False):
    """Solve a steady-state heat conduction problem.

    Parameters
    ----------
    K : float
        Thermal conductivity (W/m·K).
    ht : float
        Heat transfer coefficient (W/m²·K).
    T_edge : tuple
        Dirichlet temperatures (K).
    q0 : float
        Heat flux (W/m²).
    plot : bool
        Whether to display the result.

    Returns
    -------
    u : ndarray
        Temperature field.
    """
    sys = {"pOrd": 1, "hOrd": 1}
    mesh = read_poly("ModelHeatEquationDD", data_dir="./data")
    mesh["BC"] = {"Dir": [2, 3], "Neu": 3}

    sys, mesh = assemble_linear(sys, mesh)

    A = K * sys["S"]
    b = q0 * sys["fs"]

    for ibc in range(len(np.atleast_1d(mesh["BC"]["Dir"]))):
        dir_key = f"Dir_{ibc}"
        if dir_key in sys and len(sys[dir_key]) > 0:
            A, b = apply_dirichlet_bc(A, b, sys[dir_key], T_edge[ibc])

    u = spsolve(A, b)
    sys["u"] = u
    sys["u_abs"] = np.abs(u)

    print(f"T min: {u.min():.2f} K, T max: {u.max():.2f} K")

    if plot:
        plot_field(sys, mesh, field="u_abs", component="real", cmap="hot")

    return u


def thermal_distribution_dg():
    """Discontinuous Galerkin thermal simulation.

    Port of ProjectThermalDistroDG.m.
    """
    sys = {"pOrd": 1, "hOrd": 1}
    mesh = read_poly("ModelHeatEquationDD", data_dir="./data")
    mesh["BC"] = {"Dir": [2, 3], "Neu": 3}

    sys, mesh = assemble_linear(sys, mesh)

    A = 52 * sys["S"]
    b = 100 * sys["fs"]

    for ibc in range(len(np.atleast_1d(mesh["BC"]["Dir"]))):
        dir_key = f"Dir_{ibc}"
        if dir_key in sys and len(sys[dir_key]) > 0:
            A, b = apply_dirichlet_bc(A, b, sys[dir_key],
                                      [300.0, 200.0][ibc])

    sys["u"] = spsolve(A, b)
    print(f"DG thermal: T min={sys['u'].min():.2f}, T max={sys['u'].max():.2f}")
    return sys["u"]
