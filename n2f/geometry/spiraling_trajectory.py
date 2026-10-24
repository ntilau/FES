"""Get spiraling helicoidal trajectory function."""

import numpy as np


def get_spiraling_helicoidal_trajectory(n_scan, n_test, plot=False, fact=1):
    """
    Spherical spiraling trajectory for septentrional hemisphere scanning.

    Parameters
    ----------
    n_scan : int
        Number of scanning angles
    n_test : int
        Number of test angles in the opposite trajectory
    plot : bool, optional
        If True, plots the trajectory
    fact : float, optional
        Factor for theta range. If fact==1, theta is in the range of [0°,90°],
        fact==1/2 -> [0°,45°]

    Returns
    -------
    traj_theta : ndarray
        Scan angles theta coordinates (in degrees)
    traj_phi : ndarray
        Scan angles phi coordinates (in degrees)
    opp_traj_theta : ndarray
        Test scan angles theta coordinates (in degrees)
    opp_traj_phi : ndarray
        Test scan angles phi coordinates (in degrees)
    """
    # Parametric elliptic trajectory for main scan path
    t_ell = np.linspace(0, 0.5, n_scan)
    phi_ell = t_ell * 16 * np.pi
    theta_ell = t_ell * np.pi * fact
    
    x_ell = np.cos(phi_ell) * np.sin(theta_ell)
    y_ell = np.sin(phi_ell) * np.sin(theta_ell)
    z_ell = np.cos(theta_ell)
    
    # Opposite trajectory for testing
    t_ell_inv = np.linspace(0, 0.5, n_test)
    phi_ell_inv = t_ell_inv * 16 * np.pi + np.pi
    theta_ell_inv = t_ell_inv * np.pi
    
    x_ell_inv = np.cos(phi_ell_inv) * np.sin(theta_ell_inv)
    y_ell_inv = np.sin(phi_ell_inv) * np.sin(theta_ell_inv)
    z_ell_inv = np.cos(theta_ell_inv)
    
    if plot:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        ax.plot(x_ell, y_ell, z_ell, '-', linewidth=2, color=[0.1216, 0.2863, 0.4902])
        
        if n_test > 0:
            ax.plot(x_ell_inv, y_ell_inv, z_ell_inv, '.r', linewidth=2)
            ax.legend(['Selection trajectory', 'Tested trajectory'])
        else:
            ax.legend(['Selection trajectory'])
        
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        ax.set_zlabel('z', fontsize=12)
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([0, 1])
        ax.set_box_aspect([1, 1, 1])
        plt.show()
    
    # Convert to degrees
    from ..utils.rad2deg import rad2deg
    
    traj_theta = rad2deg(theta_ell)
    traj_phi = np.mod(rad2deg(phi_ell), 360)
    opp_traj_theta = rad2deg(theta_ell_inv)
    opp_traj_phi = np.mod(rad2deg(phi_ell_inv), 360)
    
    return traj_theta, traj_phi, opp_traj_theta, opp_traj_phi
