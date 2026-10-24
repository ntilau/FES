"""Scalar field excitations and operators."""

import numpy as np
from ..utils import deg2rad


def sf_excitations(wavelength, array_pos, steering_t, steering_p):
    """
    Computes the required linear phase excitations for a beam steering 
    process in (theta, phi). The phasors are normalized such that the total
    power emitted is unitary.

    Parameters
    ----------
    wavelength : float
        Wavelength [m]
    array_pos : ndarray (3, N_a)
        Cartesian coordinates of the array elements [m]
    steering_t : float
        Steering theta [°] (spherical coordinates)
    steering_p : float
        Steering phi [°] (spherical coordinates)

    Returns
    -------
    excit_phasor : ndarray (N_a,)
        Vector of the excitations for unit power radiation and
        normalized with respect to a single isotropic radiator
    """
    k_wn = 2 * np.pi / wavelength  # wavenumber
    
    excit_phasor = (4 * np.pi / np.sqrt(array_pos.shape[1]) * 
                    np.exp(-1j * k_wn * 
                           (array_pos[0, :] * np.sin(deg2rad(steering_t)) * np.cos(deg2rad(steering_p)) +
                            array_pos[1, :] * np.sin(deg2rad(steering_t)) * np.sin(deg2rad(steering_p)) +
                            array_pos[2, :] * np.cos(deg2rad(steering_t)))))
    
    return excit_phasor


def sf_nf2ff_operator(wavelength, theta, phi, surf_pos, n, dS):
    """
    Computes scalar near-field-to-far-field operator matrices using
    Huygens' principle.

    Parameters
    ----------
    wavelength : float
        Wavelength [m]
    theta : ndarray (N_theta,)
        Far-field elevation angles [rad]
    phi : ndarray (N_phi,)
        Far-field azimuth angles [rad]
    surf_pos : ndarray (3, N_pts)
        Cartesian coordinates of bounding surface points
    n : ndarray (3, N_pts)
        Outward normal unit vectors at each point
    dS : ndarray (N_pts,)
        Surface patch areas

    Returns
    -------
    lpsi : ndarray (N_theta, N_pts, N_phi)
        Operator for scalar field values
    ldel_psi : ndarray (N_theta, N_pts, N_phi)
        Operator for normal derivatives

    Notes
    -----
    The far-field pattern can be computed as:
    fPsi(:,i) = Lpsi(:,:,i) @ psi^T + LdelPsi(:,:,i) @ delPsi^T
    for each azimuth slice phi(i).
    """
    k_wn = 2 * np.pi / wavelength  # wavenumber
    
    lpsi = np.zeros((len(theta), surf_pos.shape[1], len(phi)), dtype=complex)
    ldel_psi = np.zeros((len(theta), surf_pos.shape[1], len(phi)), dtype=complex)
    
    for i in range(len(phi)):
        # Direction cosines for the far-field observation directions
        rx_v = np.sin(theta) * np.cos(phi[i])
        ry_v = np.sin(theta) * np.sin(phi[i])
        rz_v = np.cos(theta)
        R = np.column_stack([rx_v, ry_v, rz_v])
        
        # Green's function times surface element area for each direction and point
        green_dS = (1 / (4 * np.pi) * 
                    np.exp(1j * k_wn * (R @ surf_pos)) * 
                    (np.ones((len(theta), 1)) @ dS[np.newaxis, :]))
        
        # Operator components for psi and normal derivative contributions
        n_dot_r = R @ n
        lpsi[:, :, i] = green_dS * (1j * k_wn * n_dot_r)
        ldel_psi[:, :, i] = -green_dS
    
    return lpsi, ldel_psi
