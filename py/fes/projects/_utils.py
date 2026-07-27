"""Common solver utilities used across projects."""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve, eigs


def apply_dirichlet_bc(A, b, dir_dofs, values=None):
    """Apply Dirichlet boundary conditions to the linear system.

    Zeroes out rows/cols and sets diagonal to 1 for constrained DOFs.

    Parameters
    ----------
    A : sparse matrix
        System matrix.
    b : ndarray
        RHS vector.
    dir_dofs : ndarray
        Indices of Dirichlet DOFs (0-based).
    values : float or ndarray, optional
        Dirichlet values.

    Returns
    -------
    A : csr_matrix
        Modified system matrix.
    b : ndarray
        Modified RHS.
    """
    from scipy.sparse import coo_matrix, csr_matrix
    n_dir = len(dir_dofs)
    n = A.shape[0]

    b = b.copy()
    A = A.tocoo()

    if values is not None:
        vals = np.full(n_dir, values) if np.isscalar(values) else np.asarray(values, dtype=float)
        A_csc = A.tocsc()
        bc_term = A_csc[:, dir_dofs] * vals
        b = np.asarray(b - bc_term).ravel()

    # Build mask for entries NOT touching Dirichlet rows/cols
    free_mask = ~np.isin(A.row, dir_dofs) & ~np.isin(A.col, dir_dofs)

    # Build the modified matrix
    rows = A.row[free_mask].copy()
    cols = A.col[free_mask].copy()
    vals = A.data[free_mask].copy()

    # Add diagonal entries for Dirichlet DOFs
    rows = np.concatenate([rows, dir_dofs])
    cols = np.concatenate([cols, dir_dofs])
    vals = np.concatenate([vals, np.ones(n_dir, dtype=A.dtype)])

    A_new = coo_matrix((vals, (rows, cols)), shape=(n, n), dtype=A.dtype)

    if values is not None:
        vals = np.full(n_dir, values) if np.isscalar(values) else np.asarray(values, dtype=float)
        b[dir_dofs] = vals

    return A_new.tocsr(), b


def solve_eigenvalue(S, T, n_modes, sigma=None):
    """Solve the generalized eigenvalue problem S u = λ T u.

    Parameters
    ----------
    S, T : sparse matrix
        System matrices.
    n_modes : int
        Number of eigenvalues to compute.
    sigma : float, optional
        Target value for shift-invert mode.

    Returns
    -------
    eigenvalues : ndarray
        Sorted eigenvalues.
    eigenvectors : ndarray
        Corresponding eigenvectors.
    """
    if sigma is not None:
        evalues, evectors = eigs(S, k=n_modes, M=T, sigma=sigma, which="LM")
    else:
        evalues, evectors = eigs(S, k=n_modes, M=T, which="SM")
    idx = np.argsort(np.abs(evalues))
    return evalues[idx], evectors[:, idx]


def reconstruct_wp_field(X, sys, port_idx=0):
    """Reconstruct the full field from waveguide port solve.

    Parameters
    ----------
    X : ndarray
        Solution from waveguide port system solve.
    sys : dict
        System data with WP, WPgvec, nnWP keys.
    port_idx : int
        Port excitation index.

    Returns
    -------
    u : ndarray
        Reconstructed field on all DOFs.
    """
    if hasattr(X, 'toarray'):
        X = X.toarray()
    n_wp = len(sys["WP"])
    n_modes = sys.get("WPnModes", 1)
    if n_modes is None and "WPgvec" in sys:
        n_modes = sys["WPgvec"][0].shape[0] if hasattr(sys["WPgvec"][0], "shape") else 1

    u = np.zeros(sys["NDOFs"], dtype=complex)
    u[sys["nnWP"]] = X[n_wp * n_modes:, port_idx].ravel()

    for ip in range(n_wp):
        if "WPgvec" in sys:
            vec_block = sys["WPgvec"][ip]
            if hasattr(vec_block, "shape") and vec_block.ndim == 2:
                u[sys["WP"][ip]] = vec_block @ X[ip * n_modes:(ip + 1) * n_modes, port_idx].ravel()
        elif "WPvec" in sys:
            u[sys["WP"][ip]] = sys["WPvec"][ip] @ X[ip * n_modes:(ip + 1) * n_modes, port_idx].ravel()

    return u


def scattering_parameters(X, sys):
    """Extract scattering parameters from waveguide port solve.

    Parameters
    ----------
    X : ndarray
        Solution matrix from waveguide port system solve.
    sys : dict
        System data.

    Returns
    -------
    sp : ndarray
        Scattering parameter matrix.
    """
    if hasattr(X, 'toarray'):
        X = X.toarray()
    n_wp = len(sys["WP"])
    n_modes = sys.get("WPnModes", 1)
    n_tot = n_wp * n_modes
    sp = X[:n_tot, :n_tot] - np.eye(n_tot)
    return sp
