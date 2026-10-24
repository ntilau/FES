"""Coordinate transformation functions."""

import numpy as np


def cartesian2spherical(ax, ay, az, theta, phi):
    """
    Transform from cartesian components to spherical components of
    arbitrary structured (scalar, vector, matrix) vectorial field A.

    Parameters
    ----------
    ax, ay, az : float or ndarray
        Cartesian components of A
    theta, phi : float or ndarray
        Look angles (scalar or array)

    Returns
    -------
    ar, at, ap : float or ndarray
        Spherical components of A (Ar, At, Ap)
    """
    ar = ax * np.sin(theta) * np.cos(phi) + ay * np.sin(theta) * np.sin(phi) + az * np.cos(theta)
    at = ax * np.cos(theta) * np.cos(phi) + ay * np.cos(theta) * np.sin(phi) - az * np.sin(theta)
    ap = -ax * np.sin(phi) + ay * np.cos(phi)
    
    return ar, at, ap


def spherical2cartesian(ar, at, ap, theta, phi):
    """
    Transform from spherical components to cartesian components of
    arbitrary structured vectorial field A.

    Parameters
    ----------
    ar, at, ap : float or ndarray
        Spherical components of A
    theta, phi : float or ndarray
        Look angles (scalar or array)

    Returns
    -------
    ax, ay, az : float or ndarray
        Cartesian components of A (Ax, Ay, Az)
    """
    ax = ar * np.sin(theta) * np.cos(phi) + at * np.cos(theta) * np.cos(phi) - ap * np.sin(phi)
    ay = ar * np.sin(theta) * np.sin(phi) + at * np.cos(theta) * np.sin(phi) + ap * np.cos(phi)
    az = ar * np.cos(theta) - at * np.sin(theta)
    
    return ax, ay, az


def cross_operator(ax, ay, az, bx, by, bz):
    """
    Cross product of vectorial fields A and B.

    Parameters
    ----------
    ax, ay, az : float or ndarray
        Scalar or matrix of components (vector field A)
    bx, by, bz : float or ndarray
        Scalar or matrix of components (vector field B)

    Returns
    -------
    cx, cy, cz : float or ndarray
        Cross product components of C = A x B

    Notes
    -----
    - Can combine scalars to matrices: e.g. A is a single vector and B
      is a set of vectors
    - Does not care of the size of the matrices as the components are
      already split
    """
    cx = ay * bz - az * by
    cy = az * bx - ax * bz
    cz = ax * by - ay * bx
    
    return cx, cy, cz


def get_rotation_matrix(rot_angle, rot_axis):
    """
    Computes the rotation matrix for a rotation of angle 'a' around an axis.

    Uses Rodrigues' rotation formula:
    R = cos(a)I + (1-cos(a))uu^T + sin(a)[u]_x

    Parameters
    ----------
    rot_angle : float
        Rotation angle in degrees
    rot_axis : ndarray (3,)
        Unit vector pointing along the rotation axis [x, y, z]

    Returns
    -------
    rot_matrix : ndarray (3, 3)
        Rotation matrix for linear transformation of vectors in Cartesian coordinates
    """
    from ..utils.deg2rad import deg2rad
    
    # Normalize axis
    rot_axis_norm = rot_axis / np.linalg.norm(rot_axis)
    rot_angle_rad = deg2rad(rot_angle)
    
    # Rodrigues' formula components
    c1 = np.cos(rot_angle_rad)
    c2 = 1 - c1
    s = np.sin(rot_angle_rad)
    
    # Outer product: u*u^T
    outer = np.outer(rot_axis_norm, rot_axis_norm)
    
    # Skew-symmetric matrix [u]_x
    skew = np.array([
        [0, -rot_axis_norm[2], rot_axis_norm[1]],
        [rot_axis_norm[2], 0, -rot_axis_norm[0]],
        [-rot_axis_norm[1], rot_axis_norm[0], 0]
    ])
    
    # Build rotation matrix
    rot_matrix = c1 * np.eye(3) + c2 * outer + s * skew
    
    return rot_matrix
