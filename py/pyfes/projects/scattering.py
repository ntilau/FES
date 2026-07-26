"""Wave scattering simulations with domain decomposition.

Ports of ProjectWaveScatteringDD.m, PrjNewDDscattering.m,
ProjectWaveScatteringEincEs.m, ProjectWaveScatteringFullField.m.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from ..mesh.io_poly import read_poly
from ..fem.assembly import assemble_linear, assemble_domain_decomposition
from ..post.plot import plot_field
from ._utils import apply_dirichlet_bc


def _build_scattering_system(sys, mesh):
    """Build the Helmholtz scattering system: (S - k²T + ik·ABC) u = f."""
    k = sys["k"]
    A = sys["S"] - k**2 * sys["T"] + 1j * k * sys.get("ABC", 0)
    B_mat = sys["S"] - k**2 * sys["T"]
    return A, B_mat


def _apply_scattering_bcs(A, B_mat, sys):
    """Apply Dirichlet BCs for the scattered field formulation."""
    rhs = np.zeros(sys["NDOFs"], dtype=complex)
    if "Dir_0" in sys:
        dir_dofs = sys["Dir_0"]
        rhs = B_mat[:, dir_dofs].dot(sys["fsEinc"][dir_dofs])
        rhs[dir_dofs] = -sys["fsEinc"][dir_dofs]
        A = A.tolil()
        A[dir_dofs, :] = 0
        A[:, dir_dofs] = 0
        for d in dir_dofs:
            A[d, d] = 1.0
        A = A.tocsr()
    return A, rhs


def scattering_dd(theta=0, p_ord=4, plot=False):
    """Wave scattering with absorbing boundary conditions.

    Port of the direct solve from ProjectWaveScatteringDD.m.

    Parameters
    ----------
    theta : float
        Incidence angle (degrees).
    p_ord : int
        Polynomial order.
    plot : bool
        Whether to display the scattered field.

    Returns
    -------
    sys : dict
        System with solution field.
    """
    sys = {"pOrd": p_ord, "hOrd": 1}
    sys["k"] = 2 * np.pi
    sys["kEinc"] = np.array([np.cos(np.deg2rad(theta)),
                             np.sin(np.deg2rad(theta))])

    mesh = read_poly("ModelScatteringDD", data_dir="./data")
    mesh["BC"] = {"Dir": [1], "ABC": 133}
    mesh["epsr"] = [1, 1]

    sys, mesh = assemble_linear(sys, mesh)

    A, B_mat = _build_scattering_system(sys, mesh)

    if "Dir_0" in sys:
        A, rhs = _apply_scattering_bcs(A, B_mat, sys)
        sys["u"] = sys["fsEinc"] + spsolve(A, rhs)[:sys["NDOFs"]]
    else:
        rhs = np.zeros(sys["NDOFs"], dtype=complex)
        sys["u"] = sys["fsEinc"] + spsolve(A, rhs)[:sys["NDOFs"]]

    if plot:
        plot_field(sys, mesh, cmap="jet")

    return sys


def scattering_dd_iterative(theta=0, p_ord=4, toll=0.001, max_iter=1000, plot=False):
    """Iterative Schwarz DD for wave scattering (two-subdomain).

    Port of the iterative DD solver from ProjectWaveScatteringDD.m.

    Solves the scattering problem by iterating between two overlapping
    subdomains using Robin (ABC) transmission conditions at the DD interface.
    The iteration is: u1^{n+1} = A1^{-1}(b1 + T12 * u2^n), and vice versa.

    Parameters
    ----------
    theta : float
        Incidence angle (degrees).
    p_ord : int
        Polynomial order.
    toll : float
        Convergence tolerance.
    max_iter : int
        Maximum iterations.
    plot : bool
        Whether to display results.

    Returns
    -------
    sys : dict
        System with combined DD solution.
    """
    sys_base = {"pOrd": p_ord, "hOrd": 1}
    k = 2 * np.pi
    sys_base["k"] = k
    sys_base["kEinc"] = np.array([np.cos(np.deg2rad(theta)),
                                  np.sin(np.deg2rad(theta))])

    mesh_base = read_poly("ModelScatteringDD", data_dir="./data")
    mesh_base["BC"] = {"Dir": [1], "ABC": 133, "DD": 13}
    mesh_base["epsr"] = [1, 1]

    # Build subdomain 1 (elab == 1)
    sys1 = dict(sys_base)
    mesh1 = dict(mesh_base)
    sys1, mesh1 = assemble_linear(sys1, mesh1)

    A1, B1 = _build_scattering_system(sys1, mesh1)
    if "Dir_0" in sys1:
        dir_dofs = sys1["Dir_0"]
        rhs1 = B1[:, dir_dofs].toarray().ravel() * sys1["fsEinc"][dir_dofs]
        rhs1[dir_dofs] = -sys1["fsEinc"][dir_dofs]
        A1 = A1.tolil()
        A1[dir_dofs, :] = 0
        A1[:, dir_dofs] = 0
        for d in dir_dofs:
            A1[d, d] = 1.0
        A1 = A1.tocsr()

    # Build subdomain 2 (elab == 2)
    sys2 = dict(sys_base)
    mesh2 = dict(mesh_base)
    sys2, mesh2 = assemble_linear(sys2, mesh2)

    A2, B2 = _build_scattering_system(sys2, mesh2)
    if "Dir_0" in sys2:
        dir_dofs = sys2["Dir_0"]
        rhs2 = B2[:, dir_dofs].toarray().ravel() * sys2["fsEinc"][dir_dofs]
        rhs2[dir_dofs] = -sys2["fsEinc"][dir_dofs]
        A2 = A2.tolil()
        A2[dir_dofs, :] = 0
        A2[:, dir_dofs] = 0
        for d in dir_dofs:
            A2[d, d] = 1.0
        A2 = A2.tocsr()

    # DD coupling matrices
    D1 = sys1.get("DD", sparse.csr_matrix((sys1["NDOFs"], sys1["NDOFs"])))
    D2 = sys2.get("DD", sparse.csr_matrix((sys2["NDOFs"], sys2["NDOFs"])))
    T12 = sparse.bmat([
        [sparse.csr_matrix((sys1["NDOFs"], sys1["NDOFs"])),
         sparse.csr_matrix((sys1["NDOFs"], len(sys2.get("DirDD", []))))],
        [D2.T, -sparse.csr_matrix(
            (len(sys2.get("DirDD", [])), len(sys2.get("DirDD", []))))]
    ])
    T21 = sparse.bmat([
        [sparse.csr_matrix((sys2["NDOFs"], sys2["NDOFs"])),
         sparse.csr_matrix((sys2["NDOFs"], len(sys1.get("DirDD", []))))],
        [D1.T, -sparse.csr_matrix(
            (len(sys1.get("DirDD", [])), len(sys1.get("DirDD", []))))]
    ])

    # Iterative solve
    u1 = np.zeros(A1.shape[0], dtype=complex)
    u2 = np.zeros(A2.shape[0], dtype=complex)
    error_history = []

    for i in range(max_iter):
        u1_new = spsolve(A1, rhs1 + T12 @ u2)
        u2_new = spsolve(A2, rhs2 + T21 @ u1_new)

        err1 = np.linalg.norm(u1_new - u1) / max(np.linalg.norm(u1_new), 1e-16)
        err2 = np.linalg.norm(u2_new - u2) / max(np.linalg.norm(u2_new), 1e-16)
        err = max(err1, err2)
        error_history.append(err)

        u1, u2 = u1_new, u2_new

        if i % 10 == 0 or i == max_iter - 1:
            print(f"  DD iter {i+1}: error = {err:.6e}")

        if err < toll:
            print(f"DD converged in {i+1} iterations")
            break

    # Combine fields
    ndofs = sys1["NDOFs"]
    u1_field = np.zeros(ndofs, dtype=complex)
    u2_field = np.zeros(ndofs, dtype=complex)

    u1_field[sys1.get("DirReg", np.arange(ndofs))] = u1[:len(sys1.get("DirReg", []))]
    u2_field[sys2.get("DirReg", np.arange(ndofs))] = u2[:len(sys2.get("DirReg", []))]

    # Subtract incident field (scattered field formulation)
    sys1["u"] = sys1.get("fsEinc", np.zeros(ndofs)) + u1_field
    sys2["u"] = sys2.get("fsEinc", np.zeros(ndofs)) + u2_field
    combined = sys1["u"] + sys2["u"]

    sys_out = dict(sys_base)
    sys_out["u"] = combined
    sys_out["error_history"] = error_history

    if plot:
        plot_field(sys_out, mesh_base, cmap="jet")

    return sys_out


def scattering_full_field(theta=0, p_ord=3, plot=False):
    """Full-field scattering formulation.

    Port of ProjectWaveScatteringFullField.m.

    Parameters
    ----------
    theta : float
        Incidence angle (degrees).
    p_ord : int
        Polynomial order.
    plot : bool
        Whether to display the field.

    Returns
    -------
    sys : dict
        System with solution.
    """
    sys = {"pOrd": p_ord, "hOrd": 1}
    sys["k"] = 2 * np.pi
    sys["kEinc"] = np.array([np.cos(np.deg2rad(theta)),
                             np.sin(np.deg2rad(theta))])

    mesh = read_poly("ModelScatteringDD", data_dir="./data")
    mesh["BC"] = {"Dir": [1], "ABC": 133}
    mesh["epsr"] = [1, 1]

    sys, mesh = assemble_linear(sys, mesh)

    A, B_mat = _build_scattering_system(sys, mesh)

    if "Dir_0" in sys:
        dir_dofs = sys["Dir_0"]
        A = A.tolil()
        A[dir_dofs, :] = 0
        A[:, dir_dofs] = 0
        for d in dir_dofs:
            A[d, d] = 1.0
        A = A.tocsr()
        rhs = B_mat[:, dir_dofs].toarray().ravel() * sys["fsEinc"][dir_dofs]
        rhs[dir_dofs] = -sys["fsEinc"][dir_dofs]
    else:
        rhs = np.zeros(sys["NDOFs"], dtype=complex)

    sys["u"] = sys["fsEinc"] + spsolve(A, rhs)[:sys["NDOFs"]]

    if plot:
        plot_field(sys, mesh, cmap="jet")

    return sys
