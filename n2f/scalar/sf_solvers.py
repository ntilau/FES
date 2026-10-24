"""Scalar near-field and far-field computation functions."""

import numpy as np


def sf_compute_gain(f_psi):
    """
    Returns the directivity in [dBi] for properly normalized far field.

    Parameters
    ----------
    f_psi : ndarray
        Power density normalized far field detectors

    Returns
    -------
    gain : ndarray
        Directivity in [dBi]
    """
    gain = np.abs(f_psi) ** 2
    return gain


def sf_nf_solver(wavelength, excit_phasor, rmag, n_dot_rv):
    """
    Computes the scalar near field and its normal derivative over a
    surface used in a near-field to far-field transformation.
    Uses point-source superposition and scalar Huygens' formulation.

    Parameters
    ----------
    wavelength : float
        Wavelength [m]
    excit_phasor : ndarray (N_s,)
        Complex source excitation phasors
    rmag : ndarray (N_s, N_pts)
        Distances from each source to each surface sampling point
    n_dot_rv : ndarray (N_s, N_pts)
        Normal-direction dot products used for normal derivative computation

    Returns
    -------
    psi : ndarray (N_pts,)
        Near-field values on the surface
    del_psi : ndarray (N_pts,)
        Normal derivative of the near field on the surface

    Notes
    -----
    psi and delPsi are computed using the free-space Green's function
    for scalar point sources.
    """
    k_wn = 2 * np.pi / wavelength  # wavenumber
    
    # Computing psi, the near field
    psi = np.sum(excit_phasor[:, np.newaxis] * (np.exp(-1j * k_wn * rmag) / (4 * np.pi * rmag)), axis=0)
    
    # Computing delPsi, the near field normal derivative
    del_psi = np.sum(excit_phasor[:, np.newaxis] * 
                     (-(1j * k_wn + 1 / rmag) * np.exp(-1j * k_wn * rmag) / (4 * np.pi * rmag) * n_dot_rv), axis=0)
    
    return psi, del_psi


def sf_nf2ff_solver(wavelength, theta, phi, surf_pos, n, dS, psi, del_psi):
    """
    Computes the scalar near-field to far-field transformation using
    Huygens' principle over a sampled bounding surface.

    Parameters
    ----------
    wavelength : float
        Wavelength [m]
    theta : ndarray (N_theta,)
        Far-field elevation angles [rad]
    phi : ndarray (N_phi,)
        Far-field azimuth angles [rad]
    surf_pos : ndarray (3, N_pts)
        Coordinates of surface sampling points [m]
    n : ndarray (3, N_pts)
        Outward unit normal vectors at each surface point
    dS : ndarray (N_pts,)
        Surface patch areas
    psi : ndarray (N_pts,)
        Scalar near-field values at surface points
    del_psi : ndarray (N_pts,)
        Scalar normal derivative values at surface points

    Returns
    -------
    f_psi : ndarray (N_phi, N_theta)
        Far-field pattern matrix for specified angles
    """
    k_wn = 2 * np.pi / wavelength  # wavenumber
    
    f_psi = np.zeros((len(phi), len(theta)))
    
    for i in range(len(theta)):
        for j in range(len(phi)):
            # Unit direction vector for the far-field observation
            rx_v = np.sin(theta[i]) * np.cos(phi[j])
            ry_v = np.sin(theta[i]) * np.sin(phi[j])
            rz_v = np.cos(theta[i])
            R = np.array([rx_v, ry_v, rz_v])
            
            # Projection of surface normals onto the far-field direction
            n_dot_r = np.sum(n * R[:, np.newaxis], axis=0)
            
            # Scalar Green's function for each surface point in the far-field direction
            green = 1 / (4 * np.pi) * np.exp(1j * k_wn * (R @ surf_pos))
            
            # Surface integral for the far-field pattern component
            f_psi[j, i] = np.sum(green * ((1j * k_wn * n_dot_r) * psi - del_psi) * dS)
    
    return f_psi


def sf_direct_ff_solver(wavelength, theta, phi, excit_phasor, array_pos):
    """
    Computes the direct far-field pattern of a set of point sources.
    This reference solution is used to validate the near-field-based N2F transformation.

    Parameters
    ----------
    wavelength : float
        Wavelength [m]
    theta : ndarray (N_theta,)
        Far-field elevation angles [rad]
    phi : ndarray (N_phi,)
        Far-field azimuth angles [rad]
    excit_phasor : ndarray (N_a,)
        Complex excitation phasors for each source
    array_pos : ndarray (3, N_a)
        Cartesian coordinates of the point sources [m]

    Returns
    -------
    f_psi_ref : ndarray (N_phi, N_theta)
        Far-field pattern from the point source array
    """
    k_wn = 2 * np.pi / wavelength  # wavenumber
    
    f_psi_ref = np.zeros((len(phi), len(theta)))
    
    # Reshape theta for proper broadcasting: (N_theta, 1) to broadcast with (N_a,)
    theta_reshaped = theta[:, np.newaxis]
    
    for i in range(len(phi)):
        phase = k_wn * (array_pos[0, :] * np.sin(theta_reshaped) * np.cos(phi[i]) +
                        array_pos[1, :] * np.sin(theta_reshaped) * np.sin(phi[i]) +
                        array_pos[2, :] * np.cos(theta_reshaped))
        # phase has shape (N_theta, N_a), excit_phasor has shape (N_a,)
        # multiply element-wise and sum over sources
        f_psi_ref[i, :] = np.sum(excit_phasor / (4 * np.pi) * np.exp(1j * phase), axis=1)
    
    return f_psi_ref
