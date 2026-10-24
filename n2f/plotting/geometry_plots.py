"""Plotting functions for array and field geometry."""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ..utils import vector2matrix
from .plot_properties import get_color_map, get_figure_properties


def plot_sph_geom(matrix_size, array_pos, sphere_pos):
    """
    Plots the sphere and array geometry.

    Parameters
    ----------
    matrix_size : tuple
        For theta-phi assembly
    array_pos : ndarray (3, N_array)
        Array positions for plot
    sphere_pos : ndarray (3, N_sphere)
        Sphere grid (surface patches) positions
    """
    X = vector2matrix(matrix_size, sphere_pos[0, :])
    Y = vector2matrix(matrix_size, sphere_pos[1, :])
    Z = vector2matrix(matrix_size, sphere_pos[2, :])
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    color = get_color_map()
    ax.plot_surface(X, Y, Z, alpha=0.3, edgecolor=color, facecolor='none')
    
    ax.set_xlabel('x [λ]')
    ax.set_ylabel('y [λ]')
    ax.set_zlabel('z [λ]')
    ax.set_title('Sphere geometry')
    
    # Plot array elements
    for i in range(array_pos.shape[1]):
        ax.plot([array_pos[0, i]], [array_pos[1, i]], [array_pos[2, i]], 
                '.k', markersize=20)
    
    ax.set_box_aspect([1, 1, 1])
    plt.show()


def plot_selected_angles(span_t, span_p, test_t=None, test_p=None):
    """
    Plots the unit vectors pointing in the scan directions of both spanning
    vectors and testing vectors.

    Parameters
    ----------
    span_t : ndarray
        Theta scanning direction angles (degrees)
    span_p : ndarray
        Phi scanning direction angles (degrees)
    test_t : ndarray, optional
        Theta testing direction angles (degrees)
    test_p : ndarray, optional
        Phi testing direction angles (degrees)
    """
    from ..utils import deg2rad
    
    fig_props = get_figure_properties()
    color = get_color_map()
    
    x = np.sin(deg2rad(span_t)) * np.cos(deg2rad(span_p))
    y = np.sin(deg2rad(span_t)) * np.sin(deg2rad(span_p))
    z = np.cos(deg2rad(span_t))
    o = np.zeros_like(x)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.quiver(o, o, o, x, y, z, length=1, color=color, arrow_length_ratio=0.1)
    ax.plot(x, y, z, '.k', markersize=25)
    
    if test_t is not None and test_p is not None:
        xt = np.sin(deg2rad(test_t)) * np.cos(deg2rad(test_p))
        yt = np.sin(deg2rad(test_t)) * np.sin(deg2rad(test_p))
        zt = np.cos(deg2rad(test_t))
        ot = np.zeros_like(xt)
        
        ax.quiver(ot, ot, ot, xt, yt, zt, length=1, color='r', arrow_length_ratio=0.1)
        ax.plot(xt, yt, zt, 'sr', markersize=10)
        ax.legend(['Scan angle space selection', 'Tested scan angles'])
    else:
        ax.legend(['Scan angle space selection'])
    
    ax.set_xlabel('x', fontsize=fig_props['fs'])
    ax.set_ylabel('y', fontsize=fig_props['fs'])
    ax.set_zlabel('z', fontsize=fig_props['fs'])
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_box_aspect([1, 1, 1])
    plt.show()


def sf_plot_array_geom(array_pos):
    """
    Plots the array point sources.

    Parameters
    ----------
    array_pos : ndarray (3, N_a)
        Point sources positions in Cartesian coordinates
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    ax.plot(array_pos[0, :], array_pos[1, :], '.k', markersize=20)
    ax.set_xlabel('x [λ]')
    ax.set_ylabel('y [λ]')
    ax.set_title('Array Geometry')
    ax.set_aspect('equal')
    plt.show()


def print_pdf(path, filename, orient_type='portrait'):
    """
    Prints to PDF with Matplotlib backend.

    Parameters
    ----------
    path : str
        Path to directory
    filename : str
        Filename (without extension)
    orient_type : str, optional
        'landscape' or 'portrait'
    """
    fig = plt.gcf()
    full_path = path + filename + '.pdf'
    fig.savefig(full_path, orientation=orient_type, format='pdf')
    print(f'Saved to {full_path}')


def print_eps(path, filename, orient_type='portrait'):
    """
    Prints to EPS with Matplotlib backend.

    Parameters
    ----------
    path : str
        Path to directory
    filename : str
        Filename (without extension)
    orient_type : str, optional
        'landscape' or 'portrait'
    """
    fig = plt.gcf()
    full_path = path + filename + '.eps'
    fig.savefig(full_path, orientation=orient_type, format='eps')
    print(f'Saved to {full_path}')
