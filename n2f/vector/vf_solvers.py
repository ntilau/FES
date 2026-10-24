"""Vector near-field and far-field computation functions."""

import numpy as np
from ..transforms import cross_operator, cartesian2spherical
from ..utils import deg2rad


def vf_compute_gain(z0, et_ff, ep_ff, P):
    """
    Returns the gain (P=Pacc) and directivity (P=Prad) in theta and phi polarizations.

    Parameters
    ----------
    z0 : float
        Free-space wave impedance [ohm]
    et_ff : ndarray
        Far electric field detector in theta polarization
    ep_ff : ndarray
        Far electric field detector in phi polarization
    P : float
        Power for comparison

    Returns
    -------
    gain_t : ndarray
        Gain related to theta polarization
    gain_p : ndarray
        Gain related to phi polarization
    """
    gain_t = 2 * np.pi / P * np.abs(et_ff)**2 / z0
    gain_p = 2 * np.pi / P * np.abs(ep_ff)**2 / z0
    
    return gain_t, gain_p


def vf_nf_solver(k0, z0, array_pos, surf_pos, n, dS, J, M):
    """
    Computes the near electric and magnetic fields produced by arrays of
    electric and magnetic currents over a bounding surface.

    Parameters
    ----------
    k0 : float
        Free-space wavenumber [rad/m]
    z0 : float
        Free-space impedance [ohm]
    array_pos : ndarray (3, N_a)
        Positions of the radiating current sources [m]
    surf_pos : ndarray (3, N_pts)
        Positions of the surface sampling points [m]
    n : ndarray (3, N_pts)
        Outward unit normals at the surface points
    dS : ndarray (N_pts,)
        Surface patch areas [m^2]
    J : ndarray (3, N_a)
        Electric current densities at each source
    M : ndarray (3, N_a)
        Magnetic current densities at each source

    Returns
    -------
    E : ndarray (3, N_pts)
        Electric field on the surface [V/m]
    H : ndarray (3, N_pts)
        Magnetic field on the surface [A/m]
    S : ndarray (N_pts,)
        Poynting vector normal component on the surface
    Pr : float
        Total radiated power [W]
    """
    E = np.zeros(surf_pos.shape)
    H = np.zeros(surf_pos.shape)
    
    for i in range(array_pos.shape[1]):
        # Vector from the i-th source to each surface sampling point
        Rx = surf_pos[0, :] - array_pos[0, i]
        Ry = surf_pos[1, :] - array_pos[1, i]
        Rz = surf_pos[2, :]
        R = np.sqrt(Rx**2 + Ry**2 + Rz**2)
        RxV = Rx / R
        RyV = Ry / R
        RzV = Rz / R
        
        # Source current projections
        JdotRV = J[0, i] * RxV + J[1, i] * RyV + J[2, i] * RzV
        MdotRV = M[0, i] * RxV + M[1, i] * RyV + M[2, i] * RzV
        
        # Cross-products used in the Kottler field expressions
        JcrossRVx, JcrossRVy, JcrossRVz = cross_operator(J[0, i], J[1, i], J[2, i], RxV, RyV, RzV)
        McrossRVx, McrossRVy, McrossRVz = cross_operator(M[0, i], M[1, i], M[2, i], RxV, RyV, RzV)
        
        # Electric field contributions
        kr_inv = 1 / (k0 * R)
        kr_inv2 = kr_inv**2
        kr_inv3 = kr_inv**3
        phase = np.exp(-1j * k0 * R)
        
        E[0, :] += ((z0 * k0**2 / (4 * np.pi) * 
                    (J[0, i] * (-1j * kr_inv - kr_inv2 + 1j * kr_inv3) +
                     JdotRV * RxV * (1j * kr_inv + 3 * kr_inv2 - 3j * kr_inv3)) - 
                    k0**2 / (4 * np.pi) * McrossRVx * (1j * kr_inv + kr_inv2)) * phase)
        
        E[1, :] += ((z0 * k0**2 / (4 * np.pi) * 
                    (J[1, i] * (-1j * kr_inv - kr_inv2 + 1j * kr_inv3) +
                     JdotRV * RyV * (1j * kr_inv + 3 * kr_inv2 - 3j * kr_inv3)) - 
                    k0**2 / (4 * np.pi) * McrossRVy * (1j * kr_inv + kr_inv2)) * phase)
        
        E[2, :] += ((z0 * k0**2 / (4 * np.pi) * 
                    (J[2, i] * (-1j * kr_inv - kr_inv2 + 1j * kr_inv3) +
                     JdotRV * RzV * (1j * kr_inv + 3 * kr_inv2 - 3j * kr_inv3)) - 
                    k0**2 / (4 * np.pi) * McrossRVz * (1j * kr_inv + kr_inv2)) * phase)
        
        # Magnetic field contributions
        H[0, :] += ((k0**2 / (4 * np.pi * z0) * 
                    (M[0, i] * (-1j * kr_inv - kr_inv2 + 1j * kr_inv3) +
                     MdotRV * RxV * (1j * kr_inv + 3 * kr_inv2 - 3j * kr_inv3)) + 
                    k0**2 / (4 * np.pi) * JcrossRVx * (1j * kr_inv + kr_inv2)) * phase)
        
        H[1, :] += ((k0**2 / (4 * np.pi * z0) * 
                    (M[1, i] * (-1j * kr_inv - kr_inv2 + 1j * kr_inv3) +
                     MdotRV * RyV * (1j * kr_inv + 3 * kr_inv2 - 3j * kr_inv3)) + 
                    k0**2 / (4 * np.pi) * JcrossRVy * (1j * kr_inv + kr_inv2)) * phase)
        
        H[2, :] += ((k0**2 / (4 * np.pi * z0) * 
                    (M[2, i] * (-1j * kr_inv - kr_inv2 + 1j * kr_inv3) +
                     MdotRV * RzV * (1j * kr_inv + 3 * kr_inv2 - 3j * kr_inv3)) + 
                    k0**2 / (4 * np.pi) * McrossRVz * (1j * kr_inv + kr_inv2)) * phase)
    
    # Compute the normal component of the complex Poynting vector and total power
    Sp = np.cross(E.T, np.conj(H).T).T
    S = np.sum(Sp * n, axis=0)
    Pr = 0.5 * np.real(np.sum(S * dS))
    
    return E, H, S, Pr


def vf_nf2ff_solver(k0, z0, surf_pos, n, dS, E, H, theta_ff, phi_ff):
    """
    Computes far-field electric field components from near-field surface
    E and H fields using Huygens' principle for vector fields.

    Parameters
    ----------
    k0 : float
        Free-space wavenumber [rad/m]
    z0 : float
        Free-space impedance [ohm]
    surf_pos : ndarray (3, N_pts)
        Cartesian coordinates of the surface points [m]
    n : ndarray (3, N_pts)
        Outward unit normals at each surface point
    dS : ndarray (N_pts,)
        Surface patch areas [m^2]
    E : ndarray (3, N_pts)
        Electric near-field values [V/m]
    H : ndarray (3, N_pts)
        Magnetic near-field values [A/m]
    theta_ff : ndarray (N_phi, N_theta)
        Elevation angles (meshgrid) [rad]
    phi_ff : ndarray (N_phi, N_theta)
        Azimuth angles (meshgrid) [rad]

    Returns
    -------
    et_ff : ndarray (N_phi, N_theta)
        Theta-component of the far electric field [V/m]
    ep_ff : ndarray (N_phi, N_theta)
        Phi-component of the far electric field [V/m]
    """
    # Equivalent surface currents for Huygens sources
    Js = cross_operator(n[0, :], n[1, :], n[2, :], H[0, :], H[1, :], H[2, :])[0:3]
    Ms = cross_operator(E[0, :], E[1, :], E[2, :], n[0, :], n[1, :], n[2, :])[0:3]
    
    # Initialize far-field Cartesian components
    ex_ff = np.zeros(theta_ff.shape, dtype=complex)
    ey_ff = np.zeros(theta_ff.shape, dtype=complex)
    ez_ff = np.zeros(theta_ff.shape, dtype=complex)
    
    for i in range(theta_ff.shape[1]):
        for j in range(phi_ff.shape[0]):
            # Observation unit vector in the far-field direction
            rff_vx = np.sin(theta_ff[j, i]) * np.cos(phi_ff[j, i])
            rff_vy = np.sin(theta_ff[j, i]) * np.sin(phi_ff[j, i])
            rff_vz = np.cos(theta_ff[j, i])
            
            # Phase factor for each surface point
            green = np.exp(1j * k0 * (rff_vx * surf_pos[0, :] +
                                      rff_vy * surf_pos[1, :] +
                                      rff_vz * surf_pos[2, :]))
            
            # Projection of equivalent electric current
            js_dot_rff_v = Js[0, :] * rff_vx + Js[1, :] * rff_vy + Js[2, :] * rff_vz
            
            mcross_rff_x, mcross_rff_y, mcross_rff_z = cross_operator(
                Ms[0, :], Ms[1, :], Ms[2, :], rff_vx, rff_vy, rff_vz)
            
            # Surface integral for far-field Cartesian components
            ex_ff[j, i] = -1j * k0 / (4 * np.pi) * np.sum(
                (z0 * (Js[0, :] - js_dot_rff_v * rff_vx) + mcross_rff_x) * green * dS)
            ey_ff[j, i] = -1j * k0 / (4 * np.pi) * np.sum(
                (z0 * (Js[1, :] - js_dot_rff_v * rff_vy) + mcross_rff_y) * green * dS)
            ez_ff[j, i] = -1j * k0 / (4 * np.pi) * np.sum(
                (z0 * (Js[2, :] - js_dot_rff_v * rff_vz) + mcross_rff_z) * green * dS)
    
    # Convert Cartesian far fields to spherical components
    er_ff, et_ff, ep_ff = cartesian2spherical(ex_ff, ey_ff, ez_ff, theta_ff, phi_ff)
    
    return et_ff, ep_ff
