"""Build array functions for planar radiating arrays."""

import numpy as np


def build_array(wavelength, nbr_elems_x, wl_spacing_x, nbr_elems_y, wl_spacing_y):
    """
    Returns the positions of a planar radiating array defined on the XY plane, 
    centered at the origin.

    Parameters
    ----------
    wavelength : float
        Wavelength [m]
    nbr_elems_x : int
        Number of elements along x
    wl_spacing_x : float
        Element spacing in wavelengths along x
    nbr_elems_y : int
        Number of elements along y
    wl_spacing_y : float
        Element spacing in wavelengths along y

    Returns
    -------
    array_pos : ndarray (3, N)
        3xN matrix of element coordinates [x; y; z], with z=0
        and N = nbr_elems_x * nbr_elems_y

    Examples
    --------
    >>> array_pos = build_array(0.01, 4, 0.5, 2, 0.7)
    Creates a 4x2 planar array with 0.5 lambda spacing along x and
    0.7 lambda spacing along y. The array is centered at the origin.
    """
    spacing_x = wl_spacing_x * wavelength
    spacing_y = wl_spacing_y * wavelength
    
    # Check odd or even for array centering
    if nbr_elems_x % 2:
        dim_x = nbr_elems_x // 2
    else:
        dim_x = (nbr_elems_x - 1) // 2
    
    if nbr_elems_y % 2:
        dim_y = nbr_elems_y // 2
    else:
        dim_y = (nbr_elems_y - 1) // 2
    
    # Create planar grid points for array elements positions
    x0, y0 = np.meshgrid(spacing_x * np.arange(-dim_x, dim_x + 1),
                         spacing_y * np.arange(-dim_y, dim_y + 1))
    
    array_pos = np.vstack([
        x0.flatten(),
        y0.flatten(),
        np.zeros_like(x0.flatten())
    ])
    
    return array_pos
