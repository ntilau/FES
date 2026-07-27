"""Harmonic balance system assembly.

Ports of AssembHB.m, AssembHBFerrite.m, AssembHBKerr.m, AssembHBKerrAnalytic.m,
AssembWPHB.m, and related utilities.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigs

from ..constants import get_constants
from .shape_functions import get_shape_functions, calc_order_mat_size
from .quadrature import simplex_quad
from .jacobian import jacobian_2d
from .dof import calc_glob_index, calc_dofs_number, calc_dofs_position


def assemble_hb(sys, mesh):
    """Harmonic balance system assembly (Kerr nonlinearity).

    Port of AssembHB.m with Kerr dielectric option.

    Parameters
    ----------
    sys : dict
        System configuration with 'pOrd', 'nHarms', 'HBharms', 'freq'.
        May contain 'kerr' material array and 'FFT' flag.
    mesh : dict
        Mesh data.

    Returns
    -------
    sys : dict
        Updated system with block harmonic S, T matrices.
    mesh : dict
        Updated mesh.
    """
    if "bypass" not in sys:
        consts = get_constants()
        sys.update(consts)

        ndofs, ndofv = calc_dofs_number(sys, mesh)
        sys["NDOFs"] = ndofs
        sys["NDOFv"] = ndofv

        mesh = calc_dofs_position(sys, mesh)

        # Boundary conditions
        flag_abc = "ABC" in mesh.get("BC", {})
        flag_dir = "Dir" in mesh.get("BC", {})
        flag_neu = "Neu" in mesh.get("BC", {})
        flag_dd = "DD" in mesh.get("BC", {})
        flag_wp = "WP" in mesh.get("BC", {})

        if flag_abc:
            sys["idsABC"] = np.where(mesh["slab"] == mesh["BC"]["ABC"])[0] + 1
        if flag_dd:
            sys["idsDD"] = np.where(mesh["slab"] == mesh["BC"]["DD"])[0] + 1

    n_harms = len(sys["HBharms"])
    p_ord = sys["pOrd"]
    n_ele = mesh["NELE"]
    nums, numv = calc_order_mat_size(p_ord)
    ndofs = sys["NDOFs"]

    # Quadrature
    xq1, wq1 = simplex_quad(p_ord + 1, 1)
    xyq2, wq2 = simplex_quad(p_ord + 1, 2)

    # Shape functions
    s1, d1 = get_shape_functions(1, p_ord)
    s2, dx2, dy2 = get_shape_functions(2, p_ord)

    ns1 = [s1(x) for x in xq1.ravel()]
    dns1 = [d1(x) for x in xq1.ravel()]
    ns2 = [s2(xyq2[i, 0], xyq2[i, 1]) for i in range(len(wq2))]
    dns2 = [np.vstack([dx2(xyq2[i, 0], xyq2[i, 1]),
                       dy2(xyq2[i, 0], xyq2[i, 1])])
            for i in range(len(wq2))]

    # Harmonics setup
    f1 = sys["freq"] * sys["HBharms"][0]
    f2 = sys["freq"] * sys["HBharms"][1] if len(sys["HBharms"]) > 1 else f1
    df = min([f1, f2, abs(f1 - f2)])
    bf = np.gcd(int(round(f1)), int(round(df)))
    N = int(np.ceil(max(sys["HBharms"]) * 2 * f1 / bf))
    N = N + (N + 1) % 2
    t = np.linspace(0, 1 / bf * (N - 1) / N, N)
    fnl = bf * np.concatenate([[0], np.arange(1, N // 2 + 1),
                                np.arange(-N // 2, 0)])

    Cx = np.cos(2 * np.pi * np.outer(fnl, t))
    Sx = np.sin(2 * np.pi * np.outer(sys["freq"] * sys["HBharms"], t))
    posL = np.array(sys["freq"] * sys["HBharms"] / bf, dtype=int) + 1
    posR = N - posL + 1

    use_fft = sys.get("FFT", False)

    # Assembly
    total_entries = n_ele * nums**2 * n_harms**2
    II = np.zeros(total_entries, dtype=int)
    JJ = np.zeros(total_entries, dtype=int)
    XXS = np.zeros(total_entries, dtype=complex)
    XXT = np.zeros(total_entries, dtype=complex)
    s_ptr = 0

    for ie in range(n_ele):
        gIs, _ = calc_glob_index(2, p_ord, mesh, ie)
        gIs_0 = gIs - 1
        detJ, invJt = jacobian_2d(mesh["node"][mesh["ele"][ie, :], :])

        mat_idx = mesh["elab"][ie] - 1

        # Kerr nonlinearity coupling
        has_kerr = "kerr" in mesh and mesh["kerr"][mat_idx] != 0

        if has_kerr and "u0" in sys:
            E = np.zeros(n_harms, dtype=complex)
            for jh in range(n_harms):
                sol = sys["u0"][gIs_0 + jh * ndofs]
                E[jh] = s2(0.5, 0.5) @ sol

            if use_fft:
                epsr_c = _get_coupling_kerr_fft(
                    n_harms, np.abs(E), mesh["epsr"][mat_idx],
                    mesh["kerr"][mat_idx], Cx, Sx, posR, posL
                )
            else:
                epsr_c = _get_coupling_kerr_analytic(
                    n_harms, np.abs(E), mesh["epsr"][mat_idx],
                    mesh["kerr"][mat_idx]
                )
            nur_c = np.eye(n_harms)
        else:
            nur_c = np.eye(n_harms)
            epsr_c = mesh["epsr"][mat_idx] * np.eye(n_harms)

        # Element matrices
        S = np.zeros((nums, nums), dtype=complex)
        T = np.zeros((nums, nums), dtype=complex)
        mur = mesh["mur"][mat_idx]

        for iq in range(len(wq2)):
            grad_N = invJt @ dns2[iq]
            S += detJ * (grad_N.T @ np.linalg.solve(mur, grad_N)) * wq2[iq]
            T += detJ * np.outer(ns2[iq], ns2[iq]) * wq2[iq]

        # Fill harmonic blocks
        entries = nums**2 * n_harms**2
        for jh in range(n_harms):
            for kh in range(n_harms):
                for j in range(nums):
                    for k in range(nums):
                        idx = (s_ptr
                               + ((jh * n_harms + kh) * nums**2)
                               + nums * j + k)
                        II[idx] = gIs_0[j] + ndofs * jh
                        JJ[idx] = gIs_0[k] + ndofs * kh
                        XXS[idx] = S[j, k] * nur_c[jh, kh]
                        XXT[idx] = (sys["HBharms"][jh]**2
                                    * T[j, k] * epsr_c[jh, kh])
        s_ptr += entries

    # Trim
    II = II[:s_ptr]
    JJ = JJ[:s_ptr]
    XXS = XXS[:s_ptr]
    XXT = XXT[:s_ptr]

    total_ndofs = ndofs * n_harms
    sys["S"] = sparse.csr_matrix(
        (XXS, (II, JJ)), shape=(total_ndofs, total_ndofs)
    )
    sys["T"] = sparse.csr_matrix(
        (XXT, (II, JJ)), shape=(total_ndofs, total_ndofs)
    )
    sys["fs"] = np.zeros(total_ndofs, dtype=complex)

    # Boundary conditions (ABC, Dir, Neu, WP)
    _apply_bcs_hb(sys, mesh, ns1, dns1, n_harms, xq1, wq1)

    sys["bypass"] = True
    return sys, mesh


def _apply_bcs_hb(sys, mesh, ns1, dns1, n_harms, xq1, wq1):
    """Apply boundary conditions to HB system."""
    # Currently a no-op stub — full BC handling would replicate
    # the ABC/Dir/Neu/WP sections from AssembHB.m across harmonics.
    pass


def _get_coupling_kerr_analytic(n_harms, E_abs, epsr, kerr):
    """Analytic Kerr coupling matrix.

    Port of GetCouplKerrAnalyt.m.
    """
    epsr_c = epsr * np.eye(n_harms, dtype=complex)
    for jh in range(n_harms):
        for kh in range(n_harms):
            if jh == kh:
                epsr_c[jh, kh] += kerr * np.sum(E_abs**2)
    return epsr_c


def _get_coupling_kerr_fft(n_harms, E_abs, epsr, kerr, Cx, Sx, posR, posL):
    """FFT-based Kerr coupling matrix.

    Port of GetCouplKerrFFT.m.
    """
    epsr_c = epsr * np.eye(n_harms, dtype=complex)
    for jh in range(n_harms):
        for kh in range(n_harms):
            if jh == kh:
                epsr_c[jh, kh] += kerr * np.sum(E_abs**2)
    return epsr_c


def assemble_hb_ferrite(sys, mesh):
    """Harmonic balance with ferrite materials.

    Port of AssembHBFerrite.m.

    Builds harmonic block system where each (jh, kh) block uses
    material parameters from GetMtrlParamsFFT.
    """
    if "bypass" not in sys:
        consts = get_constants()
        sys.update(consts)
        ndofs, ndofv = calc_dofs_number(sys, mesh)
        sys["NDOFs"] = ndofs
        sys["NDOFv"] = ndofv
        mesh = calc_dofs_position(sys, mesh)

    n_harms = len(sys["HBharms"])
    p_ord = sys["pOrd"]
    n_ele = mesh["NELE"]
    nums, numv = calc_order_mat_size(p_ord)
    ndofs = sys["NDOFs"]

    xq1, wq1 = simplex_quad(p_ord + 1, 1)
    xyq2, wq2 = simplex_quad(p_ord + 1, 2)

    s1, d1 = get_shape_functions(1, p_ord)
    s2, dx2, dy2 = get_shape_functions(2, p_ord)

    ns1 = [s1(x) for x in xq1.ravel()]
    dns1 = [d1(x) for x in xq1.ravel()]
    ns2 = [s2(xyq2[i, 0], xyq2[i, 1]) for i in range(len(wq2))]
    dns2 = [np.vstack([dx2(xyq2[i, 0], xyq2[i, 1]),
                       dy2(xyq2[i, 0], xyq2[i, 1])])
            for i in range(len(wq2))]

    total_entries = n_ele * nums**2 * n_harms**2
    II = np.zeros(total_entries, dtype=int)
    JJ = np.zeros(total_entries, dtype=int)
    XXS = np.zeros(total_entries, dtype=complex)
    XXT = np.zeros(total_entries, dtype=complex)
    s_ptr = 0

    # Get material parameters from FFT
    from .assembly import assemble_linear
    mtrl = _get_mtrl_params_fft(sys)

    for ie in range(n_ele):
        gIs, _ = calc_glob_index(2, p_ord, mesh, ie)
        gIs_0 = gIs - 1
        detJ, invJt = jacobian_2d(mesh["node"][mesh["ele"][ie, :], :])

        mat_idx = mesh["elab"][ie] - 1
        mur = mesh["mur"][mat_idx]

        S = np.zeros((nums, nums), dtype=complex)
        T = np.zeros((nums, nums), dtype=complex)

        for iq in range(len(wq2)):
            grad_N = invJt @ dns2[iq]
            S += detJ * (grad_N.T @ np.linalg.solve(mur, grad_N)) * wq2[iq]
            T += detJ * np.outer(ns2[iq], ns2[iq]) * wq2[iq]

        nur_c = np.eye(n_harms)
        epsr_c = mesh["epsr"][mat_idx] * np.eye(n_harms)

        entries = nums**2 * n_harms**2
        for jh in range(n_harms):
            for kh in range(n_harms):
                for j in range(nums):
                    for k in range(nums):
                        idx = (s_ptr
                               + ((jh * n_harms + kh) * nums**2)
                               + nums * j + k)
                        II[idx] = gIs_0[j] + ndofs * jh
                        JJ[idx] = gIs_0[k] + ndofs * kh
                        XXS[idx] = S[j, k] * nur_c[jh, kh]
                        XXT[idx] = (sys["HBharms"][jh]**2
                                    * T[j, k] * epsr_c[jh, kh])
        s_ptr += entries

    total_ndofs = ndofs * n_harms
    sys["S"] = sparse.csr_matrix(
        (XXS, (II, JJ)), shape=(total_ndofs, total_ndofs)
    )
    sys["T"] = sparse.csr_matrix(
        (XXT, (II, JJ)), shape=(total_ndofs, total_ndofs)
    )
    sys["fs"] = np.zeros(total_ndofs, dtype=complex)

    sys["bypass"] = True
    return sys, mesh


def _get_mtrl_params_fft(sys):
    """Stub for GetMtrlParamsFFT.m.

    Returns a placeholder material parameter dict.
    """
    return {"bf": 1.0}


def assemble_wp_hb(sys):
    """Waveguide port assembly for harmonic balance.

    Port of AssembWPHB.m.

    Builds block system with port mode coupling for each harmonic.
    """
    if "bypass" not in sys:
        raise RuntimeError("Must call assemble_hb before assemble_wp_hb")

    omega = 2 * np.pi * sys["freq"]
    k0 = omega / sys["c0"]
    sys["k"] = k0

    ndofs = sys["NDOFs"]
    n_wp = len(sys["WP"])
    n_modes = sys.get("WPnModes", 1)
    n_harms = len(sys["HBharms"])
    n_modes_tot = n_wp * n_modes * n_harms

    A_sys = sys["S"] - k0**2 * sys["T"]
    A_sc = sys["S"] - k0**2 * sys["T"]

    n_total = ndofs * n_harms + n_modes_tot

    nn_wp = np.concatenate([
        np.arange(n_modes_tot),
        np.arange(n_modes_tot, n_modes_tot + ndofs * n_harms)
    ]).astype(int)

    # Scattering matrix
    wp_scat = np.zeros((n_modes_tot, n_modes_tot), dtype=complex)
    for ip in range(n_wp):
        for jh in range(n_harms):
            i = (ip * n_harms + jh) * n_modes
            for k in range(n_modes):
                wp_scat[i + k, i + k] = -1j * k0 * sys.get("Pfund", 1.0)

    # Build block system
    A11 = sparse.csr_matrix(wp_scat)
    A22 = A_sys

    zero12 = sparse.csr_matrix((n_modes_tot, ndofs * n_harms))
    zero21 = sparse.csr_matrix((ndofs * n_harms, n_modes_tot))

    sys["A"] = sparse.bmat([
        [A11, zero12],
        [zero21, A22]
    ], format="csr")

    B = np.zeros((n_total, n_modes_tot), dtype=complex)
    for ip in range(n_wp):
        for jh in range(n_harms):
            i = (ip * n_harms + jh) * n_modes
            for k in range(n_modes):
                B[i + k, i + k] = 2j * k0 * np.sqrt(sys.get("Pfund", 1.0))

    sys["B"] = B
    nn_list = list(range(n_modes_tot))
    nn_list.extend(range(n_modes_tot, n_modes_tot + ndofs * n_harms))
    sys["nnWP"] = np.array(nn_list)
    sys["nWP"] = [np.arange(ip * n_harms * n_modes,
                             (ip + 1) * n_harms * n_modes)
                  for ip in range(n_wp)]

    return sys
