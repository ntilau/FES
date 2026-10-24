"""Build sphere functions for near-field sampling."""

import numpy as np
from .get_tri_sph_mesh import get_tri_sph_mesh
from ..utils import get_sph_smpl_angles, get_sph_smpl_angles_for_plots, get_sph_smpl_res
from ..transforms import get_rotation_matrix


def build_sphere(radius, sph_smpl_res, d_theta, d_phi, flag, rot_angle=None, rot_axis=None):
    """
    Builds the near field sampling points location and the area of the patches.

    Parameters
    ----------
    radius : float
        Radius of the sphere in [m]
    sph_smpl_res : float
        Sampling resolution in wavelengths (depends on flag)
    d_theta : float
        Theta sampling resolution in degrees (depends on flag)
    d_phi : float
        Phi sampling resolution in degrees (depends on flag)
    flag : int
        0: dTheta-dPhi  
        1: sphSmplRes 
        2: dTheta-dPhi for nf plots (chosen in a range that allows 3D plots without missing patches)
        n.b. 2: induces errors in the n2f computation
        3: combined
    rot_angle : float, optional
        Rotation angle in degrees of the sphere points for nf rotation
    rot_axis : ndarray (3,), optional
        Rotation axis defined by the vector with components [x; y; z]

    Returns
    -------
    sphere_pos : ndarray (3, N)
        Position [x; y; z] of the sphere sampling points
    dS : ndarray (N,)
        Area of the patch in which the fields are sampled
    theta : ndarray (N,)
        Theta direction of the sphere sampling points
    phi : ndarray (N,)
        Phi direction of the sphere sampling points
    matrix_size : tuple
        For the collection of the sampling points in terms of
        the sampling direction [theta, phi]
    """
    if flag == 1 or flag == 3:
        d_theta, d_phi = get_sph_smpl_res(radius, sph_smpl_res)
    
    if flag == 2 or flag == 3:
        # for plots (leading to erroneous surface integration)
        theta, phi, matrix_size = get_sph_smpl_angles_for_plots(d_theta, d_phi)
    else:
        theta, phi, matrix_size = get_sph_smpl_angles(d_theta, d_phi)
    
    # Sphere patches areas
    dS = radius**2 * np.sin(theta) * 2 * np.pi**2 / len(theta) / len(phi)
    
    # Sphere patches vectors in cartesian coordinates
    sphere_pos = np.vstack([
        radius * np.sin(theta) * np.cos(phi),
        radius * np.sin(theta) * np.sin(phi),
        radius * np.cos(theta)
    ])
    
    # Rotation
    if rot_angle is not None and rot_axis is not None:
        rot_matrix = get_rotation_matrix(rot_angle, rot_axis)
        sphere_pos = rot_matrix @ sphere_pos
    
    return sphere_pos, dS, theta, phi, matrix_size


def build_tri_sphere(radius, tri_order, plot_tri_sphere=False):
    """
    Builds the triangulated sphere encompassing the array and the normal unit
    vectors to the triangles and outwardly directed from the array for near
    field to far field computation.

    Parameters
    ----------
    radius : float
        Radius of the sphere in [m]
    tri_order : int
        Triangulation order
    plot_tri_sphere : bool, optional
        Plots the sphere if True

    Returns
    -------
    tri_cent : ndarray (3, N_tri)
        Cartesian components of the vectors pointing to the centroids of the triangles
    tri_dS : ndarray (N_tri,)
        Areas of the triangles
    tri_n : ndarray (3, N_tri)
        Normals to the triangles
    """
    # Sphere structure construction
    p, t = get_tri_sph_mesh(tri_order)
    p = (radius * p)  # Scaling unit sphere radius to radius
    
    nbr_tri = t.shape[0]  # Number of triangles
    
    # Centroids & patches area & normal outwardly directed unit vectors
    cent_x = np.zeros(nbr_tri)
    cent_y = np.zeros(nbr_tri)
    cent_z = np.zeros(nbr_tri)
    n_vx = np.zeros(nbr_tri)
    n_vy = np.zeros(nbr_tri)
    n_vz = np.zeros(nbr_tri)
    tri_dS = np.zeros(nbr_tri)
    
    for i in range(nbr_tri):
        # Each vertex is a column vector of x, y, z components
        vert1 = p[t[i, 0], :]
        vert2 = p[t[i, 1], :]
        vert3 = p[t[i, 2], :]
        
        tri = np.column_stack([vert1, vert2, vert3])
        
        # Centroids
        cent_x[i] = np.sum(tri[0, :]) / 3
        cent_y[i] = np.sum(tri[1, :]) / 3
        cent_z[i] = np.sum(tri[2, :]) / 3
        
        # Area of the triangle using cross product
        v = vert3 - vert2
        w = vert2 - vert1
        
        cross = np.cross(w, v)
        tri_dS[i] = 0.5 * np.linalg.norm(cross)
        
        # Normals
        nx = v[1] * w[2] - w[1] * v[2]
        ny = v[2] * w[0] - w[2] * v[0]
        nz = v[0] * w[1] - w[0] * v[1]
        
        n_mag = np.sqrt(nx**2 + ny**2 + nz**2)
        n_vx[i] = nx / n_mag
        n_vy[i] = ny / n_mag
        n_vz[i] = nz / n_mag
    
    tri_cent = np.vstack([cent_x, cent_y, cent_z])
    tri_n = np.vstack([n_vx, n_vy, n_vz])
    
    if plot_tri_sphere:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        for i in range(nbr_tri):
            vertices = np.array([p[t[i, 0]], p[t[i, 1]], p[t[i, 2]]])
            poly = Poly3DCollection([vertices], alpha=0.3, edgecolor='blue')
            ax.add_collection3d(poly)
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_xlim([-radius, radius])
        ax.set_ylim([-radius, radius])
        ax.set_zlim([-radius, radius])
        ax.set_box_aspect([1, 1, 1])
        plt.show()
    
    return tri_cent, tri_dS, tri_n
