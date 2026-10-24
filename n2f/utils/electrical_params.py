"""Free-space electrical parameters and reduced output system functions."""

import numpy as np


def get_free_space_electrical_params(freq, unitary=False):
    """
    Returns the electrical parameters given a specific frequency.

    Parameters
    ----------
    freq : float
        Frequency of analysis [Hz]
    unitary : bool, optional
        If True, returns values such that the wavelength is unitary

    Returns
    -------
    z0 : float
        Free-space wave impedance [ohm] = sqrt(mu0/eps0)
    k0 : float
        Free-space wavenumber [rad/m] = 2*pi*freq*sqrt(eps0*mu0)
    lambda0 : float
        Free-space wavelength [m] = 2*pi/k0
    """
    eps0 = 8.85418781761e-12  # F/m
    mu0 = np.pi * 4e-7         # H/m
    c0 = 1 / np.sqrt(eps0 * mu0)  # Speed of light [m/s]
    
    if unitary:
        freq = c0
    
    z0 = np.sqrt(mu0 / eps0)
    lambda0 = c0 / freq
    k0 = 2 * np.pi * freq / c0
    
    return z0, k0, lambda0


def get_reduced_output_system_matrices(nbr_coeffs, nbr_smpls, k0, z0, box_pos, 
                                      box_n, dS, phi, Q):
    """
    Computes the reduced order n2f system matrices.

    Parameters
    ----------
    nbr_coeffs : int
        Number of coefficients
    nbr_smpls : int
        Number of samples
    k0 : float
        Free-space wavenumber
    z0 : float
        Free-space impedance
    box_pos : ndarray (3, N_pts)
        Box sampling points positions
    box_n : ndarray (3, N_pts)
        Normal vectors to the box faces
    dS : ndarray (N_pts,)
        Surface patch areas
    phi : ndarray
        Azimuth angles
    Q : ndarray
        Functional matrix

    Returns
    -------
    rom_ct : ndarray
        Reduced order model for theta component
    rom_cp : ndarray
        Reduced order model for phi component
    rom_f : ndarray
        Functional matrix
    nbr_smpls : int
        Number of samples (effective)
    nbr_coeffs : int
        Number of coefficients (effective, odd number)
    """
    # This is a complex function that requires data files and FFT operators
    # For now, we provide a skeleton that would need to be completed
    # with the full implementation based on MATLAB's mmread and n2fOpFieldsFFT
    
    print('#> Loading Output System matrices ...')
    
    # NOTE: This function requires loading matrix files (.mm format)
    # which is not a standard part of numpy/scipy.
    # The external dependencies would need to be added:
    # comp2Ext = scipy.io.mmread('lte_fileset/comp2Ext1.mm')
    # num2UnNum = scipy.io.mmread('lte_fileset/num2UnNum1.mm')
    # etc.
    
    # For now, we return placeholder outputs
    rom_ct = np.zeros((nbr_coeffs * 2 + 1, Q.shape[1], phi.shape[1]))
    rom_cp = np.zeros((nbr_coeffs * 2 + 1, Q.shape[1], phi.shape[1]))
    rom_f = Q
    
    return rom_ct, rom_cp, rom_f, nbr_smpls, nbr_coeffs * 2 + 1
