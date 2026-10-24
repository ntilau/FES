"""Electrostatic simulation example.

Port of ProjectElectrostatics.m.
"""

import numpy as np
from scipy.sparse.linalg import spsolve

from ..constants import get_constants
from ..mesh.io_poly import read_poly
from ..paths import data_dir
from ..core.assembly import assemble_linear
from ..post.plot import plot_field
from ._utils import apply_dirichlet_bc


def run_electrostatics():
    """Run an electrostatic simulation with Dirichlet boundaries."""
    consts = get_constants()

    sys = {"pOrd": 4, "hOrd": 1}

    # Mesh and boundary conditions
    mesh = read_poly("ModelScatteringDD_py", data_dir=data_dir())
    mesh["BC"] = {"Dir": [1, 133]}
    sys["V"] = [1.0, 0.0]

    # Assemble system
    sys, mesh = assemble_linear(sys, mesh)

    # Apply Dirichlet BCs and solve
    A = sys["S"]
    b = np.zeros(sys["NDOFs"], dtype=complex)

    for dir_dofs, value in zip(sys.get("Dir", []), sys.get("V", [])):
        A, b = apply_dirichlet_bc(A, b, dir_dofs, value)

    sys["u"] = spsolve(A, b)
    print(f"Solved, |u|_inf = {np.max(np.abs(sys['u'])):.6g}")

    plot_field(sys, mesh, cmap="jet")
    return sys, mesh


if __name__ == "__main__":
    sys, mesh = run_electrostatics()
