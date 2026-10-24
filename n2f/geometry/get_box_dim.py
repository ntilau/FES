"""Get box dimension functions."""

import numpy as np


def get_box_dim(wavelength, array_pos, wl_ranging, wl_spacing, ext):
    """
    Returns the box dimensions to encompass the array dimensions.

    Parameters
    ----------
    wavelength : float
        Wavelength
    array_pos : ndarray (3, N)
        Array elements positions of a planar array on XY plane
    wl_ranging : float
        Distance between the array and the top and bottom faces
        of the enclosing box in wavelengths
    wl_spacing : float
        Sampling resolution on the box faces in wavelengths
    ext : float
        Distance between the edge array elements and the lateral faces
        in wavelengths

    Returns
    -------
    x_min, x_max, y_min, y_max, z_min, z_max : float
        Dimensions of the box
    x_pts, y_pts, z_pts : int
        Number of sampling points along x, y and z directions
    """
    spacing = wl_spacing * wavelength
    ranging = wl_ranging * wavelength
    
    x_min = np.min(array_pos[0, :]) - ext * wavelength
    x_max = np.max(array_pos[0, :]) + ext * wavelength
    
    y_min = np.min(array_pos[1, :]) - ext * wavelength
    y_max = np.max(array_pos[1, :]) + ext * wavelength
    
    z_min = np.min(array_pos[2, :]) - ranging
    z_max = np.max(array_pos[2, :]) + ranging
    
    x_pts = int(np.floor((x_max - x_min) / spacing))
    y_pts = int(np.floor((y_max - y_min) / spacing))
    z_pts = int(np.floor(2 * ranging / spacing))
    
    return x_min, x_max, y_min, y_max, z_min, z_max, x_pts, y_pts, z_pts
