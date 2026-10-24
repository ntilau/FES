"""Plotting functions for near and far fields."""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ..utils import vector2matrix
from .plot_properties import get_figure_properties


def sf_plot_box_nf(matrix_size, box_pos, psi, chosen_title=''):
    """
    Plots the near field distribution on the box.

    Parameters
    ----------
    matrix_size : ndarray (N_faces, 2)
        Matrix sizes for each box face [n_theta, n_phi]
    box_pos : ndarray (3, N_samples)
        Near field sampling points on the box
    psi : ndarray (N_samples,)
        Vector of near field sampled values
    chosen_title : str, optional
        Additional title information

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    nbr_faces = matrix_size.shape[0]
    val_tot = 0

    for k in range(nbr_faces):
        m_size = matrix_size[k, :]
        val = m_size[0] * m_size[1]

        # Extract data for this face
        psi_face = psi[val_tot:val_tot + val]
        pos_x_face = box_pos[0, val_tot:val_tot + val]
        pos_y_face = box_pos[1, val_tot:val_tot + val]
        pos_z_face = box_pos[2, val_tot:val_tot + val]

        # Reshape to matrix form
        m_psi = vector2matrix(m_size, psi_face)
        pos_x = vector2matrix(m_size, pos_x_face)
        pos_y = vector2matrix(m_size, pos_y_face)
        pos_z = vector2matrix(m_size, pos_z_face)

        # Plot as surface
        surf = ax.plot_surface(pos_x, pos_y, pos_z, facecolors=plt.cm.viridis(
            np.abs(m_psi) / np.max(np.abs(m_psi))), alpha=0.6, edgecolor='k', linewidth=0.3)

        val_tot += val

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title(f'|Ψ| {chosen_title}')
    ax.view_init(elev=30, azim=45)
    return fig


def sf_plot_sph_nf(matrix_size, nf_data, title='Spherical Near Field'):
    """
    Plots scalar near field on spherical surface.

    Parameters
    ----------
    matrix_size : tuple
        Shape of the field data matrix (theta, phi)
    nf_data : ndarray
        Near field data to plot
    title : str, optional
        Plot title

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Reshape data
    nf_reshaped = nf_data.reshape(matrix_size)
    theta_idx, phi_idx = np.meshgrid(np.arange(matrix_size[0]),
                                      np.arange(matrix_size[1]), indexing='ij')

    # Magnitude as radius
    R = np.abs(nf_reshaped)

    # Convert spherical to Cartesian
    X = R * np.sin(np.pi * theta_idx / matrix_size[0]) * np.cos(2 * np.pi * phi_idx / matrix_size[1])
    Y = R * np.sin(np.pi * theta_idx / matrix_size[0]) * np.sin(2 * np.pi * phi_idx / matrix_size[1])
    Z = R * np.cos(np.pi * theta_idx / matrix_size[0])

    surf = ax.plot_surface(X, Y, Z, facecolors=plt.cm.viridis(
        np.abs(nf_reshaped) / np.max(np.abs(nf_reshaped))), alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_box_aspect([1, 1, 1])
    return fig


def sf_plot_sph_nf_solid(matrix_size, nf_data, title='Spherical Near Field'):
    """
    Plots scalar near field on spherical surface with solid rendering.

    Parameters
    ----------
    matrix_size : tuple
        Shape of the field data matrix (theta, phi)
    nf_data : ndarray
        Near field data to plot
    title : str, optional
        Plot title

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    """
    return sf_plot_sph_nf(matrix_size, nf_data, title)


def sf_plot_ff_cut_planes(phi_cuts, theta, pattern, title='Far Field Cut Planes'):
    """
    Plots far field cut planes.

    Parameters
    ----------
    phi_cuts : ndarray
        Phi values for cut planes
    theta : ndarray
        Theta values [degrees]
    pattern : ndarray
        Field pattern magnitude for each (theta, phi_cut)
    title : str, optional
        Plot title
    """
    fig, axes = plt.subplots(1, len(phi_cuts), figsize=(5*len(phi_cuts), 4))
    
    if len(phi_cuts) == 1:
        axes = [axes]
    
    for idx, phi_val in enumerate(phi_cuts):
        ax = axes[idx]
        ax.plot(theta, 20 * np.log10(np.abs(pattern[:, idx]) + 1e-10))
        ax.set_xlabel('Theta [deg]')
        ax.set_ylabel('Magnitude [dB]')
        ax.set_title(f'φ = {phi_val}°')
        ax.grid(True)
    
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def sf_plot_array_geom(array_pos):
    """
    Plots the array point sources geometry (2D plot on current figure).

    Parameters
    ----------
    array_pos : ndarray (3, N_array)
        Point source positions in Cartesian coordinates

    Returns
    -------
    fig : matplotlib.figure.Figure
        Current figure object
    """
    fig = plt.gcf()
    ax = fig.add_subplot(111)

    for i in range(array_pos.shape[1]):
        ax.plot(array_pos[0, i], array_pos[1, i], '.k', markersize=20)

    ax.set_xlabel('x [λ]')
    ax.set_ylabel('y [λ]')
    ax.set_title('Array geometry (2D Projection)')
    ax.axis('equal')
    ax.axis('tight')
    ax.view_init(elev=45, azim=45)
    return fig


def vf_plot_ff_3d_radiation(theta, phi, gain_theta, gain_phi, offset=0, infos=None):
    """
    Plots 3D radiation solid.

    Parameters
    ----------
    theta : ndarray (meshgrid)
        Elevation angles [radians]
    phi : ndarray (meshgrid)
        Azimuth angles [radians]
    gain_theta : ndarray
        Gain for theta polarization
    gain_phi : ndarray
        Gain for phi polarization
    offset : float, optional
        Pattern range in [dB]
    infos : str, optional
        Additional information for title

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Combine gains
    gain = gain_theta + gain_phi
    max_gain = np.max(gain)

    # Convert to dB scale with offset
    r = 10 * np.log10(gain / max_gain) + offset
    r[r < 0] = 0

    # Convert to Cartesian coordinates
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    # Plot surface
    surf = ax.plot_surface(x, y, z, facecolors=plt.cm.viridis(r / r.max()),
                          alpha=1.0, edgecolor='k', linewidth=0.1)

    # Set viewing angle
    ax.view_init(elev=45, azim=135)
    ax.axis('off')
    ax.set_box_aspect([1, 1, 1])

    # Add coordinate axes
    axis_length = 3 + offset
    ax.plot([0, 0], [0, 0], [0, axis_length], 'k-', linewidth=2)
    ax.plot([0, 0], [0, axis_length], [0, 0], 'k-', linewidth=2)
    ax.plot([0, axis_length], [0, 0], [0, 0], 'k-', linewidth=2)

    ax.text(0, 0, axis_length + 0.5, 'z', fontsize=12)
    ax.text(0, axis_length + 0.5, 0, 'y', fontsize=12)
    ax.text(axis_length + 0.5, 0, 0, 'x', fontsize=12)

    # Title
    title_str = f'Radiation solid. Max {10*np.log10(max_gain):.4g} [dBi]'
    if infos:
        title_str += f'\n{infos}'
    ax.set_title(title_str, fontsize=14)

    return fig


def vf_plot_ff_3d(theta, phi, pattern, offset=0, title='3D Far Field Radiation Pattern'):
    """
    Plots 3D far field radiation solid.

    Parameters
    ----------
    theta : ndarray (meshgrid)
        Elevation angles [radians]
    phi : ndarray (meshgrid)
        Azimuth angles [radians]
    pattern : ndarray
        Field pattern magnitude
    offset : float, optional
        Pattern range in [dB]
    title : str, optional
        Plot title

    Returns
    -------
    handle : int
        Figure handle
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    max_gain = np.max(np.abs(pattern))
    r = 10 * np.log10(np.abs(pattern) / max_gain) + offset
    r[r < 0] = 0

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    surf = ax.plot_surface(x, y, z, facecolors=plt.cm.viridis(r / r.max()),
                          alpha=1.0, edgecolor='k', linewidth=0.1)
    ax.view_init(elev=45, azim=135)
    ax.axis('off')
    ax.set_box_aspect([1, 1, 1])

    # Add coordinate axes
    axis_length = 3 + offset
    ax.plot([0, 0], [0, 0], [0, axis_length], 'k-', linewidth=2)
    ax.plot([0, 0], [0, axis_length], [0, 0], 'k-', linewidth=2)
    ax.plot([0, axis_length], [0, 0], [0, 0], 'k-', linewidth=2)

    ax.text(0, 0, axis_length, 'z', fontsize=12)
    ax.text(0, axis_length, 0, 'y', fontsize=12)
    ax.text(axis_length, 0, 0, 'x', fontsize=12)

    ax.set_title(title)
    return fig.number


def vf_plot_ff_cut_planes(phi_cuts, theta, et_pattern, ep_pattern, title='Vector Far Field Cut Planes'):
    """
    Plots vector far field cut planes.

    Parameters
    ----------
    phi_cuts : ndarray
        Phi values for cut planes
    theta : ndarray
        Theta values [degrees]
    et_pattern : ndarray
        Theta component field pattern
    ep_pattern : ndarray
        Phi component field pattern
    title : str, optional
        Plot title
    """
    fig, axes = plt.subplots(len(phi_cuts), 2, figsize=(10, 4*len(phi_cuts)))
    
    if len(phi_cuts) == 1:
        axes = axes.reshape(1, 2)
    
    for idx, phi_val in enumerate(phi_cuts):
        axes[idx, 0].plot(theta, 20 * np.log10(np.abs(et_pattern[:, idx]) + 1e-10))
        axes[idx, 0].set_ylabel('Magnitude [dB]')
        axes[idx, 0].set_title(f'Theta Pol., φ = {phi_val}°')
        axes[idx, 0].grid(True)
        
        axes[idx, 1].plot(theta, 20 * np.log10(np.abs(ep_pattern[:, idx]) + 1e-10))
        axes[idx, 1].set_ylabel('Magnitude [dB]')
        axes[idx, 1].set_title(f'Phi Pol., φ = {phi_val}°')
        axes[idx, 1].grid(True)
    
    axes[-1, 0].set_xlabel('Theta [deg]')
    axes[-1, 1].set_xlabel('Theta [deg]')
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def vf_plot_ff_polar_cut_planes(phi_cuts, theta, pattern, title='Polar Far Field Pattern'):
    """
    Plots far field in polar coordinates.

    Parameters
    ----------
    phi_cuts : ndarray
        Phi values for cut planes
    theta : ndarray
        Theta values [radians]
    pattern : ndarray
        Field pattern magnitude
    title : str, optional
        Plot title

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    """
    fig = plt.figure()
    
    for idx, phi_val in enumerate(phi_cuts):
        ax = fig.add_subplot(1, len(phi_cuts), idx+1, projection='polar')
        r = np.abs(pattern[:, idx])
        theta_deg = np.degrees(theta)
        ax.plot(theta_deg, 20 * np.log10(r + 1e-10))
        ax.set_title(f'φ = {phi_val}°')
        ax.grid(True)
    
    fig.suptitle(title)
    plt.tight_layout()
    return fig


def vf_plot_surface_power_density(matrix_size, power_density, title='Surface Power Density'):
    """
    Plots surface power density.

    Parameters
    ----------
    matrix_size : tuple
        Shape of power density matrix
    power_density : ndarray
        Power density on surface
    title : str, optional
        Plot title
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    X, Y = np.meshgrid(range(matrix_size[1]), range(matrix_size[0]))
    Z = power_density.reshape(matrix_size)
    
    surf = ax.plot_surface(X, Y, Z, cmap='hot')
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Power Density')
    fig.colorbar(surf)
    plt.show()


def plot_svd_error(errors, title='SVD Truncation Error'):
    """
    Plots SVD truncation error.

    Parameters
    ----------
    errors : ndarray
        Error values
    title : str, optional
        Plot title

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    """
    fig, ax = plt.subplots()
    ax.semilogy(errors, 'o-')
    ax.set_xlabel('Coefficient Index')
    ax.set_ylabel('Error [V]')
    ax.set_title(title)
    ax.grid(True)
    return fig


def vf_plot_array_geom_3d(array_pos):
    """
    Plots the vector array element geometry in 3D.

    Parameters
    ----------
    array_pos : ndarray (3, N_array)
        Array element positions in Cartesian coordinates

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(array_pos[0, :], array_pos[1, :], array_pos[2, :], '.k', markersize=20)
    ax.set_xlabel('x [λ]')
    ax.set_ylabel('y [λ]')
    ax.set_zlabel('z [λ]')
    ax.set_title('Array Geometry')
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=30, azim=45)
    return fig


def sf_plot_ff_cut_planes(theta, gain, ref_theta=None, ref_gain=None, planes=None,
                          infos=None, filename=None):
    """
    Plots far field pattern on cut planes (constant phi).

    Parameters
    ----------
    theta : ndarray
        Theta look angles on the constant phi plane [radians]
    gain : ndarray
        Directivity values [dBi]
    ref_theta : ndarray, optional
        Reference theta look angles
    ref_gain : ndarray, optional
        Reference directivity values [dBi]
    planes : ndarray, optional
        Vector of plane indices to plot
    infos : list, optional
        List of dicts with plot information (markers, title, legends)
    filename : list, optional
        List of filenames for saving plots

    Returns
    -------
    figs : list
        List of figure objects
    """
    fig_prop = get_figure_properties()

    if planes is None:
        planes = np.arange(gain.shape[0])

    figs = []

    for i, plane_idx in enumerate(planes):
        fig, ax = plt.subplots()

        theta_deg = np.degrees(theta.flatten()) if theta.ndim > 1 else np.degrees(theta)
        gain_vals = gain[plane_idx, :] if gain.ndim > 1 else gain

        ax.plot(theta_deg, 10 * np.log10(np.abs(gain_vals) + 1e-10), '-b',
                linewidth=fig_prop['lw'], markersize=fig_prop['ms'], label='N2F')

        if ref_gain is not None:
            ref_theta_deg = (np.degrees(ref_theta.flatten()) if ref_theta.ndim > 1
                           else np.degrees(ref_theta))
            ref_gain_vals = ref_gain[plane_idx, :] if ref_gain.ndim > 1 else ref_gain
            ax.plot(ref_theta_deg, 10 * np.log10(np.abs(ref_gain_vals) + 1e-10), '+r',
                    linewidth=fig_prop['lw'], markersize=fig_prop['ms'], label='Direct')

        v = ax.axis()
        ax.axvline(x=-90, color='k', linestyle='-.', alpha=0.5)
        ax.axvline(x=90, color='k', linestyle='-.', alpha=0.5)
        ax.axvline(x=270, color='k', linestyle='-.', alpha=0.5)

        if ref_gain is not None and np.min(ref_gain) < 1e-6:
            ax.set_ylim([-60, v[3]])

        ax.axis('tight')
        ax.set_xlabel(f'θ [°]', fontsize=fig_prop['fs'])
        ax.set_ylabel('Directivity [dBi]', fontsize=fig_prop['fs'])

        if infos is not None and i < len(infos):
            if 'title' in infos[i]:
                ax.set_title(infos[i]['title'], fontsize=fig_prop['fs'])
            if 'legend1' in infos[i]:
                legend_labels = [infos[i].get('legend1', 'N2F')]
                if 'legend2' in infos[i]:
                    legend_labels.append(infos[i]['legend2'])
                ax.legend(legend_labels, loc='lower right')
            else:
                ax.legend(loc='lower right')
        else:
            ax.legend(loc='lower right')

        ax.grid(True, alpha=0.3)
        figs.append(fig)

    return figs
