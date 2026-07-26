"""Electrostatic simulation example.

Port of ProjectElectrostatics.m.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from ..constants import get_constants
from ..mesh.io_poly import read_poly
from ..mesh.plot import plot_mesh
from ..fem.assembly import assemble_linear
from ..post.plot import plot_field


def run_electrostatics():
    """Run an electrostatic simulation with Dirichlet boundaries."""
    consts = get_constants()

    sys = {"pOrd": 4, "hOrd": 1}

    # Mesh and boundary conditions
    mesh = read_poly("ModelScatteringDD", data_dir="./data")
    mesh["BC"] = {"Dir": [1, 133]}
    sys["V"] = [1.0, 0.0]

    # Assemble system
    sys, mesh = assemble_linear(sys, mesh)

    # Apply Dirichlet BCs and solve
    A = sys["S"]
    b = np.zeros(sys["NDOFs"], dtype=complex)

    if "Dir_0" in sys:
        dir_dofs = sys["Dir_0"]
        b = b - A[:, dir_dofs].dot(np.ones(len(dir_dofs))) * sys["V"][0]
        A = A.tolil()
        A[dir_dofs, :] = 0
        A[:, dir_dofs] = 0
        A[dir_dofs, dir_dofs] = 1.0
        A = A.tocsr()
        b[dir_dofs] = sys["V"][0]

    if "Dir_1" in sys:
        dir_dofs = sys["Dir_1"]
        b = b - A[:, dir_dofs].dot(np.ones(len(dir_dofs))) * sys["V"][1]
        A = A.tolil()
        A[dir_dofs, :] = 0
        A[:, dir_dofs] = 0
        A[dir_dofs, dir_dofs] = 1.0
        A = A.tocsr()
        b[dir_dofs] = sys["V"][1]

    sys["u"] = spsolve(A, b)
    print(f"Solved, |u|_inf = {np.max(np.abs(sys['u'])):.6g}")

    plot_field(sys, mesh, cmap="jet")
    return sys, mesh


if __name__ == "__main__":
    sys, mesh = run_electrostatics()
