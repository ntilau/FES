"""Get box vector functions."""

import numpy as np


def get_box_vectors(array_pos, box_pos, box_n):
    """
    Computes the relevant vectors for scalar near field computations.

    Parameters
    ----------
    array_pos : ndarray (3, N_array)
        Array elements positions
    box_pos : ndarray (3, N_box)
        Box sampling points positions
    box_n : ndarray (3, N_box)
        Outwardly directed normal unit vectors to the faces

    Returns
    -------
    rhon_mag : ndarray (N_array, N_box)
        Vector magnitude from the array positions to the box near field sampling points
    box_n_dot_rhon_v : ndarray (N_array, N_box)
        Dot product between the normal unit vectors and the
        unit vectors in the near field sampling directions
    """
    # Array related vectors to the patches
    # For each elements vector to the integration surface, computation of the
    # distance to surface patches and the scalar product to their normal versor
    rhon_mag = np.zeros((array_pos.shape[1], box_pos.shape[1]))
    box_n_dot_rhon_v = np.zeros((array_pos.shape[1], box_pos.shape[1]))
    
    for i in range(array_pos.shape[1]):
        R = np.vstack([
            box_pos[0, :] - array_pos[0, i],
            box_pos[1, :] - array_pos[1, i],
            box_pos[2, :]
        ])
        rhon_mag[i, :] = np.sqrt(R[0, :]**2 + R[1, :]**2 + R[2, :]**2)
        # Scalar product of the versor to surface normal
        box_n_dot_rhon_v[i, :] = np.sum(box_n * R, axis=0) / rhon_mag[i, :]
    
    return rhon_mag, box_n_dot_rhon_v


def get_sph_vectors(array_pos, sphere_pos):
    """
    Computes the relevant vectors for near field to far field computations
    from a bounding sphere.

    Parameters
    ----------
    array_pos : ndarray (3, N_array)
        Cartesian coordinates of the array elements
    sphere_pos : ndarray (3, N_sphere)
        Cartesian coordinates of the sphere sampling points

    Returns
    -------
    rhon_mag : ndarray (N_array, N_sphere)
        Vectors from each array element to the sphere sampling points
    n_dot_rhon_v : ndarray (N_array, N_sphere)
        For each array element, dot product between the normal unit vector
        of the surface patch and the unit vector pointing from the point 
        source to the surface patch
    n : ndarray (3, N_sphere)
        Normal unit vector to the patches, outwardly directed from the sphere
    """
    # Normal versor of the patches in cartesian coordinates
    radius = np.sqrt(sphere_pos[0, :]**2 + sphere_pos[1, :]**2 + sphere_pos[2, :]**2)
    n = np.vstack([
        sphere_pos[0, :] / radius,
        sphere_pos[1, :] / radius,
        sphere_pos[2, :] / radius
    ])
    
    # Array related vectors to the patches
    # For each elements vector to the integration surface, computation of the
    # distance to surface patches and the scalar product to their normal versor
    rhon_mag = np.zeros((array_pos.shape[1], sphere_pos.shape[1]))
    n_dot_rhon_v = np.zeros((array_pos.shape[1], sphere_pos.shape[1]))
    
    for i in range(array_pos.shape[1]):
        R = np.vstack([
            sphere_pos[0, :] - array_pos[0, i],
            sphere_pos[1, :] - array_pos[1, i],
            sphere_pos[2, :]
        ])
        rhon_mag[i, :] = np.sqrt(R[0, :]**2 + R[1, :]**2 + R[2, :]**2)
        # Scalar product of the versor to surface normal
        n_dot_rhon_v[i, :] = np.sum(n * R, axis=0) / rhon_mag[i, :]
    
    return rhon_mag, n_dot_rhon_v, n


def get_sph_radius(wavelength, array_pos, ext):
    """
    Computes the radius of the sphere provided the array elements positions
    and the extension in wavelengths from the edge array element.

    Parameters
    ----------
    wavelength : float
        Wavelength in [m]
    array_pos : ndarray (3, N)
        Coordinates of the array elements
    ext : float
        Distance in wavelengths between the sphere and the edge array elements

    Returns
    -------
    radius : float
        Sphere radius in [m]
    """
    radius = (np.sqrt(np.max(np.abs(array_pos[0, :]))**2 + 
                      np.max(np.abs(array_pos[1, :]))**2 +
                      np.max(np.abs(array_pos[2, :]))**2) / wavelength + ext) * wavelength
    
    return radius
