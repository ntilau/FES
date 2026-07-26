"""Capacitive sensor simulation.

Port of ProjectCapacitiveClearance.m, PrjCableParassitics.m,
and ProjectCapacitiveClearanceCorrelated.m.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from ..mesh.io_poly import read_poly
from ..fem.assembly import assemble_linear
from ..post.plot import plot_field
from ._utils import apply_dirichlet_bc


def coaxial_capacitance(plot=False):
    """Compute capacitance per unit length of a coaxial cable.

    Port of PrjCableParassitics.m.

    Parameters
    ----------
    plot : bool
        Whether to display the potential field.

    Returns
    -------
    Cap : float
        Capacitance per unit length (pF/m).
    u : ndarray
        Potential field.
    """
    sys = {"pOrd": 4, "hOrd": 1}
    mesh = read_poly("coax", data_dir="./data")

    epsr_dielectric = (1 / 0.66)**2
    mesh["epsr"] = [1, epsr_dielectric]
    mesh["BC"] = {"Dir": [1, 2]}

    sys, mesh = assemble_linear(sys, mesh)

    A = sys["S"]
    b = np.zeros(sys["NDOFs"], dtype=complex)

    A, b = apply_dirichlet_bc(A, b, sys["Dir_0"], 1.0)
    A, b = apply_dirichlet_bc(A, b, sys["Dir_1"], 0.0)

    sys["u"] = spsolve(A, b)

    eps0 = 8.854187817e-12
    W = 0.5 * eps0 * np.dot(sys["u"].real, sys["S"].dot(sys["u"].real))
    Cap = 2 * W
    print(f"Capacitance = {Cap:.6g} pF/m")

    if plot:
        plot_field(sys, mesh, cmap="jet")

    return Cap, sys["u"]


def capacitive_clearance(plot=False):
    """Capacitive clearance sensor simulation.

    Port of ProjectCapacitiveClearance.m.

    Parameters
    ----------
    plot : bool
        Whether to plot the field.

    Returns
    -------
    u : ndarray
        Potential field.
    """
    sys = {"pOrd": 4, "hOrd": 1}
    mesh = read_poly("CapSense", data_dir="./data")
    mesh["BC"] = {"Dir": [1]}

    sys, mesh = assemble_linear(sys, mesh)

    A = sys["S"]
    b = np.zeros(sys["NDOFs"], dtype=complex)

    A, b = apply_dirichlet_bc(A, b, sys["Dir_0"], 1.0)

    sys["u"] = spsolve(A, b)

    if plot:
        plot_field(sys, mesh, cmap="jet")

    return sys["u"]
