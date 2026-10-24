"""Vector field near-field to far-field operators."""

import numpy as np
from ..utils import vector2matrix
from ..transforms import cross_operator


def vf_n2f_op_fields(k0, z0, surf_pos, surf_n, dS, theta_ff, phi_ff):
    """
    Computes the nf2ff operators for arbitrarily shaped surface.

    Parameters
    ----------
    k0 : float
        Free-space wavenumber [rad/m]
    z0 : float
        Free-space impedance [ohm]
    surf_pos : ndarray (3, N_pts)
        Surface sampling points locations
    surf_n : ndarray (3, N_pts)
        Outwardly directed normal unit vectors to the surface
    dS : ndarray (N_pts,)
        Surface patches areas
    theta_ff : ndarray (N_phi, N_theta)
        Look angles for pattern shape - theta component
    phi_ff : ndarray (N_phi, N_theta)
        Look angles for pattern shape - phi component

    Returns
    -------
    opt : ndarray (N_theta, N_pts*6, N_phi)
        Near fields to far electric field operator for theta polarization
    opp : ndarray (N_theta, N_pts*6, N_phi)
        Near fields to far electric field operator for phi polarization
    """
    opt = np.zeros((theta_ff.shape[0], surf_pos.shape[1] * 6, theta_ff.shape[1]), dtype=complex)
    opp = np.zeros((theta_ff.shape[0], surf_pos.shape[1] * 6, theta_ff.shape[1]), dtype=complex)
    
    dim = [theta_ff.shape[0], surf_pos.shape[1] * 3]
    
    for j in range(theta_ff.shape[1]):
        rff_vx = np.sin(theta_ff[:, j]) * np.cos(phi_ff[:, j])
        rff_vy = np.sin(theta_ff[:, j]) * np.sin(phi_ff[:, j])
        rff_vz = np.cos(theta_ff[:, j])
        
        t_x = np.cos(theta_ff[:, j]) * np.cos(phi_ff[:, j])
        t_y = np.cos(theta_ff[:, j]) * np.sin(phi_ff[:, j])
        t_z = -np.sin(theta_ff[:, j])
        p_x = -np.sin(phi_ff[:, j])
        p_y = np.cos(phi_ff[:, j])
        
        green = -1j * k0 / (4 * np.pi) * np.exp(1j * k0 * (
            rff_vx[:, np.newaxis] * surf_pos[0, :] +
            rff_vy[:, np.newaxis] * surf_pos[1, :] +
            rff_vz[:, np.newaxis] * surf_pos[2, :]))
        
        # Build theta polarization operator
        opt_e_component = np.vstack([
            green * (-(t_x[:, np.newaxis] * surf_n[2, :]) * (rff_vz[:, np.newaxis] * dS) +
                     (t_x[:, np.newaxis] * surf_n[1, :]) * (-rff_vy[:, np.newaxis] * dS)),
            green * (-(t_y[:, np.newaxis] * surf_n[2, :]) * np.zeros_like(rff_vz[:, np.newaxis]) +
                     (t_y[:, np.newaxis] * surf_n[1, :]) * (rff_vx[:, np.newaxis] * dS)),
            green * (-(t_z[:, np.newaxis] * surf_n[2, :]) * (-rff_vx[:, np.newaxis] * dS) +
                     (t_z[:, np.newaxis] * surf_n[1, :]) * np.zeros_like(rff_vy[:, np.newaxis]))
        ])
        
        opt_h_component = np.vstack([
            green * ((t_x[:, np.newaxis] * surf_n[2, :]) * z0 * (-rff_vy[:, np.newaxis] * rff_vx[:, np.newaxis] * dS) +
                     (t_y[:, np.newaxis] * surf_n[2, :]) * z0 * ((1 - rff_vy[:, np.newaxis] * rff_vy[:, np.newaxis]) * dS)),
            green * ((t_x[:, np.newaxis] * surf_n[0, :]) * z0 * (-rff_vz[:, np.newaxis] * rff_vx[:, np.newaxis] * dS) +
                     (t_y[:, np.newaxis] * surf_n[0, :]) * z0 * (-rff_vz[:, np.newaxis] * rff_vy[:, np.newaxis] * dS)),
            green * ((t_x[:, np.newaxis] * surf_n[1, :]) * z0 * ((1 - rff_vx[:, np.newaxis] * rff_vx[:, np.newaxis]) * dS) +
                     (t_y[:, np.newaxis] * surf_n[1, :]) * z0 * (-rff_vx[:, np.newaxis] * rff_vy[:, np.newaxis] * dS))
        ])
        
        opt[:, :, j] = np.hstack([opt_h_component.reshape(theta_ff.shape[0], -1),
                                   opt_e_component.reshape(theta_ff.shape[0], -1)])
        
        # Build phi polarization operator (similar structure)
        opp_e_component = np.vstack([
            green * (-(p_x[:, np.newaxis] * surf_n[2, :]) * (rff_vz[:, np.newaxis] * dS) +
                     (p_x[:, np.newaxis] * surf_n[1, :]) * (-rff_vy[:, np.newaxis] * dS)),
            green * (-(p_y[:, np.newaxis] * surf_n[2, :]) * np.zeros_like(rff_vz[:, np.newaxis]) +
                     (p_y[:, np.newaxis] * surf_n[1, :]) * (rff_vx[:, np.newaxis] * dS))
        ])
        
        opp_h_component = np.vstack([
            green * ((p_x[:, np.newaxis] * surf_n[2, :]) * z0 * (-rff_vy[:, np.newaxis] * rff_vx[:, np.newaxis] * dS) +
                     (p_y[:, np.newaxis] * surf_n[2, :]) * z0 * ((1 - rff_vy[:, np.newaxis] * rff_vy[:, np.newaxis]) * dS)),
            green * ((p_x[:, np.newaxis] * surf_n[0, :]) * z0 * (-rff_vz[:, np.newaxis] * rff_vx[:, np.newaxis] * dS) +
                     (p_y[:, np.newaxis] * surf_n[0, :]) * z0 * (-rff_vz[:, np.newaxis] * rff_vy[:, np.newaxis] * dS))
        ])
        
        opp[:, :, j] = np.hstack([opp_h_component.reshape(theta_ff.shape[0], -1),
                                   opp_e_component.reshape(theta_ff.shape[0], -1)])
    
    return opt, opp


def vf_n2f_op_fields_fft(k0, z0, surf_pos, surf_n, dS, theta_ff, phi_ff, nbr_coeffs):
    """
    Computes the nf2ff operators for arbitrarily shaped surface using FFT.

    Parameters
    ----------
    k0 : float
        Free-space wavenumber [rad/m]
    z0 : float
        Free-space impedance [ohm]
    surf_pos : ndarray (3, N_pts)
        Surface sampling points locations
    surf_n : ndarray (3, N_pts)
        Outwardly directed normal unit vectors to the surface
    dS : ndarray (N_pts,)
        Surface patches areas
    theta_ff : ndarray (N_theta, N_phi)
        Look angles for pattern shape - theta component
    phi_ff : ndarray (N_theta, N_phi)
        Look angles for pattern shape - phi component
    nbr_coeffs : int
        Coefficients to retain in the DFT-truncation

    Returns
    -------
    opt : ndarray (2*nbr_coeffs+1, N_pts*6, N_phi)
        Near fields to far electric field operator Fourier coefficients for theta polarization
    opp : ndarray (2*nbr_coeffs+1, N_pts*6, N_phi)
        Near fields to far electric field operator Fourier coefficients for phi polarization
    nbr_coeffs : int
        Effective number of coefficients retained (odd number)
    """
    size_t = theta_ff.shape[0]
    nbr_coeffs_half = int(np.floor(nbr_coeffs / 2))
    coeffs_ind = np.concatenate([np.arange(nbr_coeffs_half + 1), 
                                  np.arange(size_t - nbr_coeffs_half, size_t)])
    
    opt = np.zeros((2 * nbr_coeffs_half + 1, surf_pos.shape[1] * 6, theta_ff.shape[1]), dtype=complex)
    opp = np.zeros((2 * nbr_coeffs_half + 1, surf_pos.shape[1] * 6, theta_ff.shape[1]), dtype=complex)
    
    # Compute full operators
    opt_full, opp_full = vf_n2f_op_fields(k0, z0, surf_pos, surf_n, dS, theta_ff, phi_ff)
    
    # Apply FFT and coefficient truncation
    for phi_idx in range(theta_ff.shape[1]):
        opt_fft = np.fft.fft(opt_full[:, :, phi_idx], axis=0)
        opp_fft = np.fft.fft(opp_full[:, :, phi_idx], axis=0)
        
        opt[:, :, phi_idx] = opt_fft[coeffs_ind, :]
        opp[:, :, phi_idx] = opp_fft[coeffs_ind, :]
    
    nbr_coeffs_effective = 2 * nbr_coeffs_half + 1
    
    return opt, opp, nbr_coeffs_effective
