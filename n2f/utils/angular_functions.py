"""Angular and look angle functions."""

import numpy as np


def get_spanning_angles(max_nbr_angles, angle_range):
    """
    Theta exponential angles' selection.
    Builds the sequence 1 2 3 5 9 17 33 65 129 ...

    Parameters
    ----------
    max_nbr_angles : int
        Truncation of the sequence
    angle_range : float
        Extension around broadside in degrees in [0°, 180°]

    Returns
    -------
    angles : ndarray
        Theta values in degrees
    nbr_angles : ndarray
        Number of angles effectively computed at each step
    """
    angles = [angle_range / 2, 0, angle_range]
    order = int(np.floor(np.log2(max_nbr_angles))) - 1
    data_inc = [1, 2]
    nbr_angles = [1, 3]
    
    for i in range(order):
        temp = sorted(angles)
        new = temp[1] / 2
        angles.append(new)
        data = 1
        parts = int(angle_range / new)
        
        for j in range(1, parts + 1):
            if new * j not in angles:
                angles.append(new * j)
                data += 1
        
        data_inc.append(data)
        nbr_angles.append(sum(data_inc))
    
    return np.array(angles), np.array(nbr_angles)


def get_look_angle_poly(phi, size_t, size_t_ref, nbr_coeffs):
    """
    Returns the complex trigonometric polynomials values for selected look
    angles and related to the proper eigenfrequency.

    Parameters
    ----------
    phi : ndarray
        Angles of cut planes selected
    size_t : int
        Number of theta look angles for pattern plot
    size_t_ref : int
        Size of the initial DFT dimension
    nbr_coeffs : int
        Number of coefficients retained in DFT-truncation

    Returns
    -------
    ifft_op : ndarray (size_t, 2*nbrCoeffs+1)
        Complex trigonometric polynomials sampled values matrix
    phi_ff : ndarray
        Sampled look angles phi
    theta_ff : ndarray
        Sampled look angles theta
    """
    nbr_coeffs = int(np.floor(nbr_coeffs / 2))
    phi_ff, theta_ff = np.meshgrid(phi, np.linspace(0, 2 * np.pi * size_t / size_t, size_t))
    
    angles = np.linspace(0, (size_t - 1) / size_t, size_t)
    ifft_op = np.zeros((size_t, 2 * nbr_coeffs + 1), dtype=complex)
    
    for i in range(nbr_coeffs + 1):
        ifft_op[:, i] = np.exp(1j * 2 * np.pi * angles * (i - 1)) / size_t_ref
    
    for i in range(1, nbr_coeffs + 1):
        ifft_op[:, nbr_coeffs + i] = np.exp(-1j * 2 * np.pi * angles * (nbr_coeffs - i + 1)) / size_t_ref
    
    return ifft_op, phi_ff, theta_ff
