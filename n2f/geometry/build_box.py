"""Build box functions for sampling near fields."""

import numpy as np


def build_box(faces, x_min, x_max, y_min, y_max, z_min, z_max,
              x_pts, y_pts, z_pts, scale, plot_flag=False, for_plot=False):
    """
    Builds a box that encompasses the array over which will be computed the
    near fields on the selected faces and returns the near field sampling
    points position and the patches areas.

    Parameters
    ----------
    faces : ndarray of bool (6,)
        Boolean vector of the faces to select [XY face up (top), 
        XY face down (bottom), XZ face up, XZ face down,
        YZ face up, YZ face down]
    x_min, x_max, y_min, y_max, z_min, z_max : float
        Box dimensions
    x_pts, y_pts, z_pts : int
        Number of sampling points along x, y, z directions
    scale : float
        Reduction of the box dimensions in [m] that scales
        proportionally the box within the initial dimensions
    plot_flag : bool, optional
        Plots the sampling grid if True
    for_plot : bool, optional
        Allows proper box dimensions plot if True

    Returns
    -------
    box_pos : ndarray (3, N_pts)
        Positions of the sampling points or grid intersections
    box_n : ndarray (3, N_faces)
        Outwardly directed normal unit vectors to the faces
    dS : ndarray
        Surface patches areas
    m_size : ndarray
        Matrix sizes collected for each face selected
    """
    nbr_faces = np.sum(faces)
    
    x_vect = np.linspace(x_min, x_max, x_pts + 1)
    y_vect = np.linspace(y_min, y_max, y_pts + 1)
    z_vect = np.linspace(z_min, z_max, z_pts + 1)
    
    dx = abs(x_vect[0] - x_vect[1])
    dy = abs(y_vect[0] - y_vect[1])
    dz = abs(z_vect[0] - z_vect[1])
    
    if not for_plot:
        x_min = x_min * scale
        y_min = y_min * scale
        z_min = z_min * scale
        x_max = x_max * scale
        y_max = y_max * scale
        z_max = z_max * scale
        
        x_min_tmp = x_min + dx / 2
        y_min_tmp = y_min + dy / 2
        z_min_tmp = z_min + dz / 2
        x_max_tmp = x_max - dx / 2
        y_max_tmp = y_max - dy / 2
        z_max_tmp = z_max - dz / 2
        
        x_vect = np.linspace(x_min_tmp, x_max_tmp, x_pts)
        y_vect = np.linspace(y_min_tmp, y_max_tmp, y_pts)
        z_vect = np.linspace(z_min_tmp, z_max_tmp, z_pts)
    
    # Building box
    face_list = []
    
    if faces[0]:
        # XY face up
        pos_x, pos_y = np.meshgrid(x_vect, y_vect)
        pos_z = z_max * np.ones_like(pos_x)
        face_list.append({
            'pos_x': pos_x, 'pos_y': pos_y, 'pos_z': pos_z,
            'dS': dx * dy * np.ones_like(pos_x),
            'nVx': 0, 'nVy': 0, 'nVz': 1
        })
    
    if faces[1]:
        # XY face down
        pos_x, pos_y = np.meshgrid(x_vect, y_vect)
        pos_z = z_min * np.ones_like(pos_x)
        face_list.append({
            'pos_x': pos_x, 'pos_y': pos_y, 'pos_z': pos_z,
            'dS': dx * dy * np.ones_like(pos_x),
            'nVx': 0, 'nVy': 0, 'nVz': -1
        })
    
    if faces[2]:
        # XZ face up
        pos_x, pos_z = np.meshgrid(x_vect, z_vect)
        pos_y = y_max * np.ones_like(pos_x)
        face_list.append({
            'pos_x': pos_x, 'pos_y': pos_y, 'pos_z': pos_z,
            'dS': dx * dz * np.ones_like(pos_x),
            'nVx': 0, 'nVy': 1, 'nVz': 0
        })
    
    if faces[3]:
        # XZ face down
        pos_x, pos_z = np.meshgrid(x_vect, z_vect)
        pos_y = y_min * np.ones_like(pos_x)
        face_list.append({
            'pos_x': pos_x, 'pos_y': pos_y, 'pos_z': pos_z,
            'dS': dx * dz * np.ones_like(pos_x),
            'nVx': 0, 'nVy': -1, 'nVz': 0
        })
    
    if faces[4]:
        # YZ face up
        pos_y, pos_z = np.meshgrid(y_vect, z_vect)
        pos_x = x_max * np.ones_like(pos_y)
        face_list.append({
            'pos_x': pos_x, 'pos_y': pos_y, 'pos_z': pos_z,
            'dS': dy * dz * np.ones_like(pos_y),
            'nVx': 1, 'nVy': 0, 'nVz': 0
        })
    
    if faces[5]:
        # YZ face down
        pos_y, pos_z = np.meshgrid(y_vect, z_vect)
        pos_x = x_min * np.ones_like(pos_y)
        face_list.append({
            'pos_x': pos_x, 'pos_y': pos_y, 'pos_z': pos_z,
            'dS': dy * dz * np.ones_like(pos_y),
            'nVx': -1, 'nVy': 0, 'nVz': 0
        })
    
    # Stack all positions and normals
    pos_x_all = np.concatenate([f['pos_x'].flatten() for f in face_list])
    pos_y_all = np.concatenate([f['pos_y'].flatten() for f in face_list])
    pos_z_all = np.concatenate([f['pos_z'].flatten() for f in face_list])
    box_pos = np.vstack([pos_x_all, pos_y_all, pos_z_all])
    
    n_vx_all = np.concatenate([np.full_like(f['pos_x'].flatten(), f['nVx']) for f in face_list])
    n_vy_all = np.concatenate([np.full_like(f['pos_y'].flatten(), f['nVy']) for f in face_list])
    n_vz_all = np.concatenate([np.full_like(f['pos_z'].flatten(), f['nVz']) for f in face_list])
    box_n = np.vstack([n_vx_all, n_vy_all, n_vz_all])
    
    dS = np.concatenate([f['dS'].flatten() for f in face_list])
    
    m_size = np.array([f['pos_x'].shape for f in face_list])
    
    return box_pos, box_n, dS, m_size
