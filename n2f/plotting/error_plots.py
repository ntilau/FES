"""Plotting functions for error analysis and SVD."""

import numpy as np
import matplotlib.pyplot as plt
from .plot_properties import get_figure_properties


def plot_svd_error(s_psi, nbr_vectors, n_error, f_error, f_ref_error=None,
                   nbr_elems_x=None, plot_error_only=False):
    """
    Error induced to the pattern and singular values plot after low rank
    approximation of the near field.

    Parameters
    ----------
    s_psi : ndarray (N, N)
        Diagonal matrix of the singular values
    nbr_vectors : ndarray
        Number of near field pictures collected
    n_error : ndarray
        L2 error in the near field relative to non-tested
    f_error : ndarray
        L2 error in the transformed far field relative to non-tested near fields
    f_ref_error : ndarray, optional
        L2 error in the transformed far field relative to directly computed
        far field
    nbr_elems_x : int, optional
        Number of point sources in the x direction (scanning direction)
    plot_error_only : bool, optional
        If True, avoid plotting the singular values

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    """
    fig_prop = get_figure_properties()

    # Extract error values at nbr_vectors indices
    n_err_sel = n_error[nbr_vectors]
    f_err_sel = f_error[nbr_vectors]

    if plot_error_only:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        ax = ax1

    # Plot errors on first subplot
    ax.semilogy(nbr_vectors, n_err_sel, '+-b', linewidth=fig_prop['lw'],
                markersize=fig_prop['ms'], label='Near field')
    ax.semilogy(nbr_vectors, f_err_sel, '*-r', linewidth=fig_prop['lw'],
                markersize=fig_prop['ms'], label='N2F far field')

    max_error = max(n_err_sel[-1], f_err_sel[-1])
    min_error = min(n_err_sel[-1], f_err_sel[-1])

    if f_ref_error is not None:
        f_ref_err_sel = f_ref_error[nbr_vectors]
        ax.semilogy(nbr_vectors, f_ref_err_sel, 'x-k',
                    linewidth=fig_prop['lw'], markersize=fig_prop['ms'],
                    label='Direct far field')

    ax.set_xlim([nbr_vectors[0], nbr_vectors[-1]])
    ax.set_ylim([1e-15, 1e5])

    # Determine legend location based on error magnitudes
    log_min = np.log10(min_error)
    log_max = np.log10(max_error)
    loc_value = 10**((log_min + log_max) / 2)
    if loc_value > 10**((np.log10(1e-15) + np.log10(1e5)) / 2):
        location = 'lower right'
    else:
        location = 'upper right'

    ax.set_xlabel('No of vectors', fontsize=fig_prop['fs'])
    ax.set_ylabel('Relative error', fontsize=fig_prop['fs'])
    ax.legend(loc=location)
    ax.grid(True, alpha=0.3)

    # Plot singular values on second subplot (if not plot_error_only)
    if not plot_error_only:
        s_diag = np.diag(s_psi)
        ax2.semilogy(s_diag, 'o-b', linewidth=fig_prop['lw'],
                     markersize=fig_prop['ms'])

        if nbr_elems_x is not None:
            if nbr_vectors[-1] > nbr_elems_x:
                ax2.text(nbr_elems_x + 1, s_psi[nbr_elems_x, nbr_elems_x],
                        f'$\\sigma_{{{nbr_elems_x}}}$ = {s_psi[nbr_elems_x, nbr_elems_x]:.4g}',
                        fontsize=fig_prop['fs'])
            else:
                ax2.text(2, s_psi[nbr_elems_x, nbr_elems_x],
                        f'$\\sigma_{{{nbr_elems_x}}}$ = {s_psi[nbr_elems_x, nbr_elems_x]:.4g}',
                        fontsize=fig_prop['fs'])

        ax2.set_xlim([s_diag.size * 0.01, s_diag.size])
        ax2.set_ylim([1e-15, 1e5])
        ax2.set_xlabel('n', fontsize=fig_prop['fs'])
        ax2.set_ylabel('Singular values $\\sigma_n$', fontsize=fig_prop['fs'])
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
