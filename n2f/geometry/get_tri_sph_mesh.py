"""Get triangulated sphere mesh function."""

import numpy as np


def get_tri_sph_mesh(n):
    """
    Returns the triangulated model of a sphere using the icosahedron subdivision method.

    Parameters
    ----------
    n : int
        Number of subdivisions. Can assume values between 0-inf. 
        The greater n the better the surface but more computation time.

    Returns
    -------
    p : ndarray (N_points, 3)
        Points of the triangulated sphere model
    t : ndarray (N_triangles, 3)
        Triangle indexes (1-indexed in MATLAB, 0-indexed in Python)

    Notes
    -----
    The sphere is supposed to be of unit radius and centered in
    (0,0,0). To obtain spheres centered in different locations or with
    different radius, just apply translation and scaling transformations.
    """
    # Twelve vertices of icosahedron on unit sphere
    tau = 0.8506508084  # t=(1+sqrt(5))/2, tau=t/sqrt(1+t^2)
    one = 0.5257311121  # one=1/sqrt(1+t^2), unit sphere
    
    p = np.array([
        [tau, one, 0],      # ZA  - 0
        [-tau, one, 0],     # ZB  - 1
        [-tau, -one, 0],    # ZC  - 2
        [tau, -one, 0],     # ZD  - 3
        [one, 0, tau],      # YA  - 4
        [one, 0, -tau],     # YB  - 5
        [-one, 0, -tau],    # YC  - 6
        [-one, 0, tau],     # YD  - 7
        [0, tau, one],      # XA  - 8
        [0, -tau, one],     # XB  - 9
        [0, -tau, -one],    # XC  - 10
        [0, tau, -one]      # XD  - 11
    ])
    
    # Structure for unit icosahedron
    t = np.array([
        [4, 7, 8],
        [4, 9, 7],
        [5, 11, 6],
        [5, 6, 10],
        [0, 3, 4],
        [0, 5, 3],
        [2, 1, 7],
        [2, 6, 1],
        [8, 11, 0],
        [8, 1, 11],
        [9, 3, 10],
        [9, 10, 2],
        [8, 0, 4],
        [11, 5, 0],
        [4, 3, 9],
        [5, 10, 3],
        [7, 1, 8],
        [6, 11, 1],
        [9, 2, 10],
        [7, 2, 4]
    ], dtype=int)
    
    # Subdivide
    for _ in range(n):
        t_new = []
        for tri in t:
            # Get the 3 vertices
            v0, v1, v2 = tri
            
            # Get the 3 edge midpoints (or find if already exists)
            # Create new vertices at edge midpoints
            p = np.vstack([p, (p[v0] + p[v1]) / 2])
            m01_idx = len(p) - 1
            
            p = np.vstack([p, (p[v1] + p[v2]) / 2])
            m12_idx = len(p) - 1
            
            p = np.vstack([p, (p[v2] + p[v0]) / 2])
            m20_idx = len(p) - 1
            
            # Normalize new vertices to unit sphere
            p[m01_idx] = p[m01_idx] / np.linalg.norm(p[m01_idx])
            p[m12_idx] = p[m12_idx] / np.linalg.norm(p[m12_idx])
            p[m20_idx] = p[m20_idx] / np.linalg.norm(p[m20_idx])
            
            # Create 4 new triangles
            t_new.append([v0, m01_idx, m20_idx])
            t_new.append([v1, m12_idx, m01_idx])
            t_new.append([v2, m20_idx, m12_idx])
            t_new.append([m01_idx, m12_idx, m20_idx])
        
        t = np.array(t_new, dtype=int)
    
    return p, t
