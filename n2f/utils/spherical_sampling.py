"""Spherical sampling angle functions."""

import numpy as np
from .deg2rad import deg2rad


def get_sph_smpl_angles(d_theta, d_phi):
    """
    Computes the sphere sampling angles for near field to far field transformation.

    Parameters
    ----------
    d_theta : float
        Theta sampling resolution in degrees
    d_phi : float
        Phi sampling resolution in degrees

    Returns
    -------
    theta : ndarray (N_theta*N_phi,)
        Column vector of theta angles in radians
    phi : ndarray (N_theta*N_phi,)
        Column vector of phi angles in radians
    matrix_size : tuple (N_phi, N_theta)
        Size of the angles meshgrid for plots
    """
    theta_grid, phi_grid = np.meshgrid(
        deg2rad(np.arange(d_theta/2, 180, d_theta)),
        deg2rad(np.arange(0, 360, d_phi))
    )
    matrix_size = theta_grid.shape
    theta = theta_grid.flatten()
    phi = phi_grid.flatten()
    
    return theta, phi, matrix_size


def get_sph_smpl_angles_for_plots(d_theta, d_phi):
    """
    Computes the sphere sampling angles for plot purposes. As the initial and
    final angles coincide, this introduces an error in the near field to far
    field transformation.

    Parameters
    ----------
    d_theta : float
        Theta sampling resolution in degrees
    d_phi : float
        Phi sampling resolution in degrees

    Returns
    -------
    theta : ndarray (N_theta*N_phi,)
        Column vector of theta angles in radians
    phi : ndarray (N_theta*N_phi,)
        Column vector of phi angles in radians
    matrix_size : tuple (N_phi, N_theta)
        Size of the angles meshgrid for plots
    """
    theta_grid, phi_grid = np.meshgrid(
        deg2rad(np.arange(0, 180 + d_theta, d_theta)),
        deg2rad(np.arange(0, 360 + d_phi, d_phi))
    )
    matrix_size = theta_grid.shape
    theta = theta_grid.flatten()
    phi = phi_grid.flatten()
    
    return theta, phi, matrix_size


def get_sph_smpl_res(radius, sph_smpl_res):
    """
    Computes the resolution in degrees given the desired resolution in wavelengths.

    Parameters
    ----------
    radius : float
        Radius of the sphere in metres
    sph_smpl_res : float
        Resolution in metres

    Returns
    -------
    d_theta : float
        Resolution in degrees for theta
    d_phi : float
        Resolution in degrees for phi
    """
    n_theta = int(np.floor(radius * np.pi / sph_smpl_res - 1))
    n_phi = int(np.floor(radius * 2 * np.pi / sph_smpl_res))
    
    d_theta = 180 / n_theta
    d_phi = 360 / n_phi
    
    return d_theta, d_phi
