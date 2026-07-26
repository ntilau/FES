"""Finite element system matrix assembly.

Port of AssembLin.m, AssembWP.m, AssembDD.m, AssembSolveDD.m,
AssembSolveFull.m, and related utilities.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigs

from ..constants import get_constants
from .shape_functions import (
    get_shape_functions,
    calc_curl_shape_functions,
    calc_order_mat_size,
)
from .quadrature import simplex_quad
from .jacobian import jacobian_2d
from .dof import calc_glob_index, calc_dofs_number, calc_dofs_position


def assemble_linear(sys, mesh):
    """Assemble the linear finite element system.

    Port of AssembLin.m. Builds:
    - S (stiffness) and T (mass) scalar matrices
    - St (curl-curl) and Tt (mass) vector matrices
    - G (gradient coupling) matrix
    - Boundary condition matrices: ABC, Dir, Neu, DD, WP
    - RHS vector fs

    Parameters
    ----------
    sys : dict
        System configuration. Must contain at minimum 'pOrd'.
        May contain 'k' (wavenumber), 'kEinc' (incident direction),
        'Hcurl' flag for vector elements, boundary conditions.
    mesh : dict
        Mesh data from read_poly() or build_regular_square().

    Returns
    -------
    sys : dict
        Updated system with assembled matrices.
    mesh : dict
        Updated mesh with DOF positions.
    """
    bypass = sys.get("bypass", False)
    flag_abc = False
    flag_dir = False
    flag_neu = False
    flag_dd = False
    flag_wp = False

    if not bypass:
        # Physical constants
        consts = get_constants()
        sys.update(consts)

        # Count DOFs
        ndofs, ndofv = calc_dofs_number(sys, mesh)
        sys["NDOFs"] = ndofs
        sys["NDOFv"] = ndofv

        # Position DOFs
        mesh = calc_dofs_position(sys, mesh)

        # Detect boundary conditions
        if "BC" in mesh:
            bc = mesh["BC"]
            if "ABC" in bc:
                ids_abc = np.where(mesh["slab"] == bc["ABC"])[0] + 1
                sys["idsABC"] = ids_abc
                flag_abc = True
            if "Dir" in bc:
                ids_dir = []
                for ibc, val in enumerate(np.atleast_1d(bc["Dir"])):
                    ids_dir.append(np.where(mesh["slab"] == val)[0] + 1)
                flag_dir = True
            if "Neu" in bc:
                ids_neu = np.where(mesh["slab"] == bc["Neu"])[0] + 1
                flag_neu = True
            if "DD" in bc:
                ids_dd = np.where(mesh["slab"] == bc["DD"])[0] + 1
                sys["idsDD"] = ids_dd
                flag_dd = True
            if "WP" in bc:
                ids_wp = []
                for ibc, val in enumerate(np.atleast_1d(bc["WP"])):
                    ids_wp.append(np.where(mesh["slab"] == val)[0] + 1)
                flag_wp = True

        # Material properties
        n_mtrl = len(np.unique(mesh["elab"]))
        if "epsr" not in mesh:
            mesh["epsr"] = np.ones(n_mtrl)
        if "mur" not in mesh:
            mesh["mur"] = [np.eye(2) for _ in range(n_mtrl)]

    # Element matrix sizes
    nums, numv = calc_order_mat_size(sys["pOrd"])

    # Quadrature
    xq1, wq1 = simplex_quad(sys["pOrd"] + 1, 1)
    xyq2, wq2 = simplex_quad(sys["pOrd"] + 1, 2)

    # Shape functions
    use_vector = "Hcurl" in sys
    s1, d1 = get_shape_functions(1, sys["pOrd"])
    s2, dx2, dy2 = get_shape_functions(2, sys["pOrd"])

    # Pre-compute scalar shape functions at quadrature points
    ns1 = [s1(x) for x in xq1.ravel()]
    dns1 = [d1(x) for x in xq1.ravel()]

    ns2 = [s2(xyq2[i, 0], xyq2[i, 1]) for i in range(len(wq2))]
    dns2 = []
    for i in range(len(wq2)):
        x, y = xyq2[i, 0], xyq2[i, 1]
        dns2.append(np.vstack([dx2(x, y), dy2(x, y)]))

    # Vector shape functions
    if use_vector:
        s2_1, dx2_1, dy2_1 = get_shape_functions(2, 1)
        ns2_1 = [s2_1(xyq2[i, 0], xyq2[i, 1]) for i in range(len(wq2))]
        dns2_1 = []
        for i in range(len(wq2)):
            x, y = xyq2[i, 0], xyq2[i, 1]
            dns2_1.append(np.vstack([dx2_1(x, y), dy2_1(x, y)]))

    # Pre-allocate sparse triplets
    n_ele = mesh["NELE"]
    p_ord = sys["pOrd"]

    iis = np.zeros(n_ele * nums**2, dtype=int)
    jjs = np.zeros(n_ele * nums**2, dtype=int)
    xxs = np.zeros(n_ele * nums**2, dtype=complex)
    xxt = np.zeros(n_ele * nums**2, dtype=complex)
    is_ptr = 0

    if use_vector:
        iiv = np.zeros(n_ele * numv**2, dtype=int)
        jjv = np.zeros(n_ele * numv**2, dtype=int)
        xxsv = np.zeros(n_ele * numv**2, dtype=complex)
        xxtv = np.zeros(n_ele * numv**2, dtype=complex)
        xxtv2 = np.zeros(n_ele * numv**2, dtype=complex)
        iivs = np.zeros(n_ele * numv * nums, dtype=int)
        jjvs = np.zeros(n_ele * numv * nums, dtype=int)
        xxgvs = np.zeros(n_ele * numv * nums, dtype=complex)
        iv_ptr = 0
        ivs_ptr = 0

    # Element loop
    for ie in range(n_ele):
        gIs, gIv = calc_glob_index(2, p_ord, mesh, ie)
        gIs_0 = gIs - 1  # to 0-based
        gIv_0 = gIv - 1

        tri_verts = mesh["node"][mesh["ele"][ie, :], :]
        detJ, invJt = jacobian_2d(tri_verts)

        mat_idx = int(mesh["elab"][ie] - 1)  # 0-based material index

        # Scalar element matrices
        S = np.zeros((nums, nums), dtype=complex)
        T = np.zeros((nums, nums), dtype=complex)

        mur = mesh["mur"][mat_idx]
        epsr = mesh["epsr"][mat_idx]

        for iq in range(len(wq2)):
            grad_N = invJt @ dns2[iq]
            S = S + detJ * (grad_N.T @ np.linalg.solve(mur, grad_N)) * wq2[iq]
            T = T + detJ * (np.outer(ns2[iq], ns2[iq])) * wq2[iq] * epsr

        # Store scalar element matrices
        for j in range(nums):
            for k in range(nums):
                idx = is_ptr + nums * j + k
                iis[idx] = gIs_0[j]
                jjs[idx] = gIs_0[k]
                xxs[idx] = S[j, k]
                xxt[idx] = T[j, k]
        is_ptr += nums**2

        # Vector element matrices
        if use_vector:
            Sv = np.zeros((numv, numv), dtype=complex)
            Tv = np.zeros((numv, numv), dtype=complex)
            Gvs = np.zeros((numv, nums), dtype=complex)

            for iq in range(len(wq2)):
                x = ns2_1[iq]
                dx = invJt @ dns2_1[iq]

                if p_ord == 1:
                    Nv2 = np.zeros((3, 2))
                    Nv2[0, :] = x[1] * dx[:, 2] - x[2] * dx[:, 1]
                    Nv2[1, :] = x[0] * dx[:, 2] - x[2] * dx[:, 0]
                    Nv2[2, :] = x[0] * dx[:, 1] - x[1] * dx[:, 0]
                    dNv2 = np.array([2.0, -2.0, 2.0])
                elif p_ord == 2:
                    Nv2 = np.zeros((8, 2))
                    Nv2[0, :] = x[1] * dx[:, 2] - x[2] * dx[:, 1]
                    Nv2[1, :] = x[0] * dx[:, 2] - x[2] * dx[:, 0]
                    Nv2[2, :] = x[0] * dx[:, 1] - x[1] * dx[:, 0]
                    Nv2[3, :] = 4 * (x[1] * dx[:, 2] + x[2] * dx[:, 1])
                    Nv2[4, :] = 4 * (x[0] * dx[:, 2] + x[2] * dx[:, 0])
                    Nv2[5, :] = 4 * (x[0] * dx[:, 1] + x[1] * dx[:, 0])
                    Nv2[6, :] = x[0] * x[1] * dx[:, 2] - x[0] * x[2] * dx[:, 1]
                    Nv2[7, :] = x[0] * x[1] * dx[:, 2] - x[1] * x[2] * dx[:, 0]
                    dNv2 = np.array([2.0, -2.0, 2.0, 0.0, 0.0, 0.0,
                                     2 * x[0] - x[1] - x[2],
                                     x[0] + x[2] - 2 * x[1]])
                elif p_ord == 3:
                    Nv2 = np.zeros((15, 2))
                    Nv2[0, :] = x[1] * dx[:, 2] - x[2] * dx[:, 1]
                    Nv2[1, :] = x[0] * dx[:, 2] - x[2] * dx[:, 0]
                    Nv2[2, :] = x[0] * dx[:, 1] - x[1] * dx[:, 0]
                    Nv2[3, :] = 4 * (x[1] * dx[:, 2] + x[2] * dx[:, 1])
                    Nv2[4, :] = 4 * (x[0] * dx[:, 2] + x[2] * dx[:, 0])
                    Nv2[5, :] = 4 * (x[0] * dx[:, 1] + x[1] * dx[:, 0])
                    Nv2[6, :] = x[0] * x[1] * dx[:, 2] - x[0] * x[2] * dx[:, 1]
                    Nv2[7, :] = x[0] * x[1] * dx[:, 2] - x[1] * x[2] * dx[:, 0]
                    Nv2[8, :] = (x[1] * (x[1] - 2 * x[2]) * dx[:, 2]
                                 + x[2] * (-x[2] + 2 * x[1]) * dx[:, 1])
                    Nv2[9, :] = (x[0] * (x[0] - 2 * x[2]) * dx[:, 2]
                                 + x[2] * (-x[2] + 2 * x[0]) * dx[:, 0])
                    Nv2[10, :] = (x[0] * (x[0] - 2 * x[1]) * dx[:, 1]
                                  + x[1] * (-x[1] + 2 * x[0]) * dx[:, 0])
                    Nv2[11, :] = (x[1] * x[0] * dx[:, 2]
                                  + x[0] * x[2] * dx[:, 1]
                                  + x[1] * x[2] * dx[:, 0])
                    Nv2[12, :] = (-x[1] * x[0] * (x[1] - 2 * x[2]) * dx[:, 2]
                                  - x[0] * x[2] * (-x[2] + 2 * x[1]) * dx[:, 1]
                                  + 3 * x[1] * x[2] * (x[1] - x[2]) * dx[:, 0])
                    Nv2[13, :] = (-x[0] * x[1] * (x[0] - 2 * x[2]) * dx[:, 2]
                                  + 3 * x[0] * x[2] * (x[0] - x[2]) * dx[:, 1]
                                  - x[1] * x[2] * (-x[2] + 2 * x[0]) * dx[:, 0])
                    Nv2[14, :] = (3 * x[0] * x[1] * (x[0] - x[1]) * dx[:, 2]
                                  - x[0] * x[2] * (x[0] - 2 * x[1]) * dx[:, 1]
                                  - x[1] * x[2] * (-x[1] + 2 * x[0]) * dx[:, 0])
                    dNv2 = np.array([
                        2.0, -2.0, 2.0,
                        0.0, 0.0, 0.0,
                        2 * x[0] - x[1] - x[2],
                        x[0] + x[2] - 2 * x[1],
                        0.0, 0.0, 0.0, 0.0,
                        -16 * x[1] * x[2] + 4 * x[2]**2 + 4 * x[1]**2,
                        16 * x[0] * x[2] - 4 * x[2]**2 - 4 * x[0]**2,
                        -16 * x[1] * x[0] + 4 * x[1]**2 + 4 * x[0]**2
                    ]) * (dx[0, 1] * dx[1, 2] - dx[1, 1] * dx[0, 2])
                else:
                    raise ValueError(f"Vector shape functions not implemented for order {p_ord}")

                Sv = Sv + detJ * np.outer(dNv2, dNv2) * wq2[iq]
                Tv = Tv + detJ * (Nv2 @ Nv2.T) * wq2[iq]
                Gvs = Gvs + detJ * (Nv2 @ grad_N) * wq2[iq]

            # Store vector element matrices
            for j in range(numv):
                for k in range(numv):
                    idx = iv_ptr + numv * j + k
                    iiv[idx] = gIv_0[j]
                    jjv[idx] = gIv_0[k]
                    xxsv[idx] = Sv[j, k]
                    xxtv[idx] = Tv[j, k] * epsr
                    xxtv2[idx] = Tv[j, k]
            iv_ptr += numv**2

            # Gradient coupling matrix
            for j in range(numv):
                for k in range(nums):
                    idx = ivs_ptr + nums * j + k
                    iivs[idx] = gIv_0[j]
                    jjvs[idx] = gIs_0[k]
                    xxgvs[idx] = Gvs[j, k]
            ivs_ptr += numv * nums

    # Build global sparse matrices
    sys["S"] = sparse.csr_matrix(
        (xxs, (iis, jjs)), shape=(ndofs, ndofs)
    )
    sys["T"] = sparse.csr_matrix(
        (xxt, (iis, jjs)), shape=(ndofs, ndofs)
    )
    sys["fs"] = np.zeros(ndofs, dtype=complex)

    if use_vector:
        sys["St"] = sparse.csr_matrix(
            (xxsv, (iiv, jjv)), shape=(ndofv, ndofv)
        )
        sys["Tt"] = sparse.csr_matrix(
            (xxtv, (iiv, jjv)), shape=(ndofv, ndofv)
        )
        sys["Tt2"] = sparse.csr_matrix(
            (xxtv2, (iiv, jjv)), shape=(ndofv, ndofv)
        )
        sys["G"] = sparse.csr_matrix(
            (xxgvs, (iivs, jjvs)), shape=(ndofv, ndofs)
        )

    # ================================================================
    # Boundary conditions
    # ================================================================

    # Absorbing boundary condition
    if flag_abc:
        ids_abc = sys.get("idsABC", [])
        if len(ids_abc) > 0:
            k = sys.get("k", 1.0)
            k_einc = sys.get("kEinc", np.array([0.0, 0.0]))
            ref_node = mesh.get("refNode", np.zeros((ndofs, 3)))
            sys["fsEinc"] = np.exp(
                -1j * k * (ref_node[:, :2] @ k_einc)
            )

            NSABC = len(ids_abc)
            max_entries = NSABC * (p_ord + 1)**2
            ii_abc = np.zeros(max_entries, dtype=int)
            jj_abc = np.zeros(max_entries, dtype=int)
            xx_abc = np.zeros(max_entries)
            is_abc = 0

            for ie in range(n_ele):
                on_abc = np.array([
                    np.sum(np.abs(mesh["spig"][ie, 0]) == ids_abc),
                    np.sum(np.abs(mesh["spig"][ie, 1]) == ids_abc),
                    np.sum(np.abs(mesh["spig"][ie, 2]) == ids_abc),
                ])
                if np.sum(on_abc) > 0:
                    id_on_abc = np.where(on_abc > 0)[0]
                    for i in id_on_abc:
                        spig_id = mesh["spig"][ie, i]
                        node_id = mesh["spig2"][abs(spig_id) - 1, :]
                        int_node = mesh["ele"][ie, :]
                        int_node = int_node[
                            (int_node != node_id[0]) & (int_node != node_id[1])
                        ][0]

                        gIs = calc_glob_index(1, p_ord, mesh, ie, i + 1)[0]
                        gIs_0 = gIs - 1

                        l_vec = np.diff(mesh["node"][node_id, :], axis=0).ravel()
                        l = np.linalg.norm(l_vec)

                        n_vec = np.cross(
                            np.cross(
                                np.append(l_vec, 0),
                                np.append(np.diff(
                                    mesh["node"][[node_id[0], int_node], :],
                                    axis=0
                                ).ravel(), 0)
                            ),
                            np.append(l_vec, 0)
                        )[:2]
                        n_vec = n_vec / np.linalg.norm(n_vec)

                        TrBC = np.zeros((p_ord + 1, p_ord + 1))
                        for iq in range(len(wq1)):
                            TrBC += l * np.outer(ns1[iq], ns1[iq]) * wq1[iq]

                        rho = np.column_stack([
                            mesh["node"][node_id[0], :]
                            + np.outer(xq1.ravel(), l_vec),
                            np.zeros(len(wq1))
                        ])
                        v = np.array([0.0, 0.0, 1.0])
                        k_einc_3d = np.append(k_einc, 0)
                        n_3d = np.append(n_vec, 0)

                        Inc = (np.dot(v, v - np.cross(n_3d, np.cross(k_einc_3d, v)))
                               * np.exp(-1j * k * np.dot(
                                   rho[:, :2], k_einc
                               )))

                        frBC = np.zeros(p_ord + 1, dtype=complex)
                        for iq in range(len(wq1)):
                            frBC += l * ns1[iq] * Inc[iq] * wq1[iq]

                        entries = (p_ord + 1)**2
                        for j in range(p_ord + 1):
                            for kk in range(p_ord + 1):
                                idx = is_abc + (p_ord + 1) * j + kk
                                ii_abc[idx] = gIs_0[j]
                                jj_abc[idx] = gIs_0[kk]
                                xx_abc[idx] = TrBC[j, kk]
                        is_abc += entries

                        sys["fs"][gIs_0] += frBC

            ii_abc = ii_abc[:is_abc]
            jj_abc = jj_abc[:is_abc]
            xx_abc = xx_abc[:is_abc]
            sys["ABC"] = sparse.csr_matrix(
                (xx_abc, (ii_abc, jj_abc)), shape=(ndofs, ndofs)
            )
            sys["DirABC"] = np.unique(ii_abc)

    # Domain decomposition boundary condition
    if flag_dd:
        ids_dd = sys.get("idsDD", [])
        if len(ids_dd) > 0:
            NSDD = len(ids_dd)
            max_entries = NSDD * (p_ord + 1)**2
            ii_dd = np.zeros(max_entries, dtype=int)
            jj_dd = np.zeros(max_entries, dtype=int)
            xx_dd = np.zeros(max_entries)
            is_dd = 0
            k = sys.get("k", 1.0)
            k_einc = sys.get("kEinc", np.array([0.0, 0.0]))

            for ie in range(n_ele):
                on_dd = np.array([
                    np.sum(np.abs(mesh["spig"][ie, 0]) == ids_dd),
                    np.sum(np.abs(mesh["spig"][ie, 1]) == ids_dd),
                    np.sum(np.abs(mesh["spig"][ie, 2]) == ids_dd),
                ])
                if np.sum(on_dd) > 0:
                    id_on_dd = np.where(on_dd > 0)[0]
                    for i in id_on_dd:
                        spig_id = mesh["spig"][ie, i]
                        node_id = mesh["spig2"][abs(spig_id) - 1, :]
                        int_node = mesh["ele"][ie, :]
                        int_node = int_node[
                            (int_node != node_id[0]) & (int_node != node_id[1])
                        ][0]

                        gIs = calc_glob_index(1, p_ord, mesh, ie, i + 1)[0]
                        gIs_0 = gIs - 1

                        l_vec = np.diff(mesh["node"][node_id, :], axis=0).ravel()
                        l = np.linalg.norm(l_vec)
                        n_vec = np.cross(
                            np.append(l_vec, 0),
                            np.cross(
                                np.append(l_vec, 0),
                                np.append(np.diff(
                                    mesh["node"][[node_id[0], int_node], :],
                                    axis=0
                                ).ravel(), 0)
                            )
                        )[:2]
                        n_vec = n_vec / np.linalg.norm(n_vec)

                        TrBC = np.zeros((p_ord + 1, p_ord + 1))
                        for iq in range(len(wq1)):
                            TrBC += l * np.outer(ns1[iq], ns1[iq]) * wq1[iq]

                        rho = np.column_stack([
                            mesh["node"][node_id[0], :]
                            + np.outer(xq1.ravel(), l_vec),
                            np.zeros(len(wq1))
                        ])
                        dEinc = (-1j * k * np.dot(k_einc, n_vec)
                                 * np.exp(-1j * k * np.dot(
                                     rho[:, :2], k_einc
                                 )))

                        frBC = np.zeros(p_ord + 1, dtype=complex)
                        for iq in range(len(wq1)):
                            frBC += l * ns1[iq] * dEinc[iq] * wq1[iq]

                        entries = (p_ord + 1)**2
                        for j in range(p_ord + 1):
                            for kk in range(p_ord + 1):
                                idx = is_dd + (p_ord + 1) * j + kk
                                ii_dd[idx] = gIs_0[j]
                                jj_dd[idx] = gIs_0[kk]
                                xx_dd[idx] = TrBC[j, kk]
                        is_dd += entries

                        sys["fs"][gIs_0] += frBC

            ii_dd = ii_dd[:is_dd]
            jj_dd = jj_dd[:is_dd]
            xx_dd = xx_dd[:is_dd]
            sys["DD"] = sparse.csr_matrix(
                (xx_dd, (ii_dd, jj_dd)), shape=(ndofs, ndofs)
            )
            sys["DirDD"] = np.unique(ii_dd)

    # Dirichlet boundary condition
    if flag_dir:
        ids_dir_list = []
        for ibc, val in enumerate(np.atleast_1d(mesh["BC"]["Dir"])):
            ids_dir_list.append(np.where(mesh["slab"] == val)[0] + 1)

        for ibc, ids_dir in enumerate(ids_dir_list):
            if len(ids_dir) == 0:
                continue
            # Use list for dynamic growth (corner elements touch 2+ Dirichlet
            # edges, so the number of element-edge visits exceeds NSDir)
            ii_dir = []

            for ie in range(n_ele):
                on_dir = np.array([
                    np.sum(np.abs(mesh["spig"][ie, 0]) == ids_dir),
                    np.sum(np.abs(mesh["spig"][ie, 1]) == ids_dir),
                    np.sum(np.abs(mesh["spig"][ie, 2]) == ids_dir),
                ])
                if np.sum(on_dir) > 0:
                    id_on_dir = np.where(on_dir > 0)[0]
                    for i in id_on_dir:
                        gIs = calc_glob_index(1, p_ord, mesh, ie, i + 1)[0]
                        gIs_0 = gIs - 1
                        spig_id = abs(mesh["spig"][ie, i])
                        node_id = mesh["spig2"][spig_id - 1, :]
                        int_node = mesh["ele"][ie, :]
                        int_node = int_node[
                            (int_node != node_id[0]) & (int_node != node_id[1])
                        ][0]

                        l_vec = np.diff(mesh["node"][node_id, :], axis=0).ravel()
                        l = np.linalg.norm(l_vec)

                        frBC = np.zeros(p_ord + 1, dtype=complex)
                        for iq in range(len(wq1)):
                            frBC += l * ns1[iq] * wq1[iq]

                        for j in range(p_ord + 1):
                            ii_dir.append(gIs_0[j])
                        sys["fs"][gIs_0] += frBC

            sys[f"Dir_{ibc}"] = np.unique(np.array(ii_dir, dtype=int))

    # Neumann boundary condition
    if flag_neu:
        ids_neu = np.where(mesh["slab"] == mesh["BC"]["Neu"])[0] + 1
        NSNeu = len(ids_neu)
        ii_neu = np.zeros(NSNeu * (p_ord + 1)**2, dtype=int)
        is_neu = 0

        for ie in range(n_ele):
            on_neu = np.array([
                np.sum(np.abs(mesh["spig"][ie, 0]) == ids_neu),
                np.sum(np.abs(mesh["spig"][ie, 1]) == ids_neu),
                np.sum(np.abs(mesh["spig"][ie, 2]) == ids_neu),
            ])
            if np.sum(on_neu) > 0:
                id_on_neu = np.where(on_neu > 0)[0]
                for i in id_on_neu:
                    spig_id = abs(mesh["spig"][ie, i])
                    node_id = mesh["spig2"][spig_id - 1, :]
                    gIs = calc_glob_index(1, p_ord, mesh, ie, i + 1)[0]
                    gIs_0 = gIs - 1
                    l_vec = np.diff(mesh["node"][node_id, :], axis=0).ravel()
                    l = np.linalg.norm(l_vec)

                    frBC = np.zeros(p_ord + 1)
                    for iq in range(len(wq1)):
                        frBC += l * ns1[iq] * wq1[iq]

                    for j in range(p_ord + 1):
                        for kk in range(p_ord + 1):
                            idx = is_neu + (p_ord + 1) * j + kk
                            ii_neu[idx] = gIs_0[j]
                    is_neu += (p_ord + 1)**2
                    sys["fs"][gIs_0] += frBC

        sys["Neu"] = np.unique(ii_neu[:is_neu])

    # Waveguide port boundary condition
    if flag_wp:
        ids_wp_list = []
        for val in np.atleast_1d(mesh["BC"]["WP"]):
            ids_wp_list.append(np.where(mesh["slab"] == val)[0] + 1)

        sys["WP"] = []
        sys["WPvec"] = []
        sys["WPfc"] = []

        for ibc, ids_wp in enumerate(ids_wp_list):
            if len(ids_wp) == 0:
                continue
            NSWp = len(ids_wp)
            max_entries = NSWp * (p_ord + 1)**2
            ii_wp_s = np.zeros(max_entries, dtype=int)
            jj_wp_s = np.zeros(max_entries, dtype=int)
            ss_wp_s = np.zeros(max_entries)
            tt_wp_s = np.zeros(max_entries)
            is_wp = 0

            for ie in range(n_ele):
                on_wp = np.array([
                    np.sum(np.abs(mesh["spig"][ie, 0]) == ids_wp),
                    np.sum(np.abs(mesh["spig"][ie, 1]) == ids_wp),
                    np.sum(np.abs(mesh["spig"][ie, 2]) == ids_wp),
                ])
                if np.sum(on_wp) > 0:
                    id_on_wp = np.where(on_wp > 0)[0]
                    for i in id_on_wp:
                        gIs = calc_glob_index(1, p_ord, mesh, ie, i + 1)[0]
                        gIs_0 = gIs - 1
                        spig_id = abs(mesh["spig"][ie, i])
                        node_id = mesh["spig2"][spig_id - 1, :]
                        int_node = mesh["ele"][ie, :]
                        int_node = int_node[
                            (int_node != node_id[0]) & (int_node != node_id[1])
                        ][0]

                        l_vec = np.diff(mesh["node"][node_id, :], axis=0).ravel()
                        l = np.linalg.norm(l_vec)
                        detJ = l

                        Stt = np.zeros((p_ord + 1, p_ord + 1))
                        Ttt = np.zeros((p_ord + 1, p_ord + 1))
                        for iq in range(len(wq1)):
                            Stt += (l * np.outer(dns1[iq] / detJ, dns1[iq] / detJ)
                                    * wq1[iq])
                            Ttt += l * np.outer(ns1[iq], ns1[iq]) * wq1[iq]

                        entries = (p_ord + 1)**2
                        for j in range(p_ord + 1):
                            for kk in range(p_ord + 1):
                                idx = is_wp + (p_ord + 1) * j + kk
                                ii_wp_s[idx] = gIs_0[j]
                                jj_wp_s[idx] = gIs_0[kk]
                                ss_wp_s[idx] = Stt[j, kk]
                                tt_wp_s[idx] = Ttt[j, kk]
                        is_wp += entries

            ii_wp_s = ii_wp_s[:is_wp]
            jj_wp_s = jj_wp_s[:is_wp]
            ss_wp_s = ss_wp_s[:is_wp]
            tt_wp_s = tt_wp_s[:is_wp]

            SspWP = sparse.csr_matrix(
                (ss_wp_s, (ii_wp_s, jj_wp_s)), shape=(ndofs, ndofs)
            )
            TspWP = sparse.csr_matrix(
                (tt_wp_s, (ii_wp_s, jj_wp_s)), shape=(ndofs, ndofs)
            )

            wp_dofs = np.unique(ii_wp_s)

            # Remove Dirichlet DOFs from waveguide port
            for key in list(sys.keys()):
                if key.startswith("Dir_") and isinstance(sys[key], np.ndarray):
                    mask = np.isin(wp_dofs, sys[key])
                    wp_dofs = wp_dofs[~mask]

            SspWP = SspWP[wp_dofs, :][:, wp_dofs]
            TspWP = TspWP[wp_dofs, :][:, wp_dofs]

            wp_n_modes = sys.get("WPnModes", 1)
            try:
                n_wp_modes = min(wp_n_modes, max(1, SspWP.shape[0] - 2))
                if n_wp_modes < 1:
                    raise ValueError("Too few WP DOFs")
                eval_vec, evect = eigs(
                    SspWP, k=n_wp_modes, M=TspWP, which="SM"
                )
                idx = np.argsort(eval_vec.real)
                eval_vec = eval_vec[idx]
                evect = evect[:, idx]

                # Normalize
                norm_factor = np.sqrt(np.diag(evect.T @ TspWP @ evect).real)
                norm_factor[norm_factor < 1e-15] = 1.0
                evect = evect / norm_factor

                # Filter out non-positive eigenvalues
                valid = eval_vec.real > 1e-10
                if np.sum(valid) > 0:
                    eval_vec = eval_vec[valid]
                    evect = evect[:, valid]
                    wp_fc = np.sqrt(eval_vec.real) * sys["c0"] / (2 * np.pi)
                else:
                    raise ValueError("No valid positive eigenvalues")

                # Pad to wp_n_modes if fewer modes were found
                if len(wp_fc) < wp_n_modes:
                    pts = mesh["refNode"][wp_dofs, :2]
                    port_width = max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]))
                    pad_fc = sys["c0"] / (2 * max(port_width, 1e-6))
                    wp_fc = np.pad(wp_fc, (0, wp_n_modes - len(wp_fc)),
                                   constant_values=pad_fc)
                    evect = np.pad(evect, ((0, 0), (0, wp_n_modes - evect.shape[1])),
                                   mode="constant", constant_values=0)
                    evect[:, -1] = 1.0

            except Exception:
                # Fallback: estimate cutoff from port width along longest edge dimension
                pts = mesh["refNode"][wp_dofs, :2]
                port_width = max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]))
                port_width = max(port_width, 1e-6)  # avoid zero division
                wp_fc = np.full(wp_n_modes, sys["c0"] / (2 * port_width))
                evect = np.ones((len(wp_dofs), wp_n_modes))
                t_norm = evect.T @ TspWP @ evect
                t_norm = np.abs(np.diag(t_norm).real)
                t_norm[t_norm < 1e-15] = 1.0
                evect = evect / np.sqrt(t_norm)

            sys["WP"].append(wp_dofs)
            sys["WPvec"].append(evect)
            sys["WPfc"].append(wp_fc)

    sys["bypass"] = True
    return sys, mesh


def assemble_waveguide_port(sys, freq):
    """Assemble the waveguide port system for a given frequency.

    Port of AssembWP.m. Builds the reduced system matrix A and RHS B
    for the waveguide port scattering problem.

    Parameters
    ----------
    sys : dict
        Previously assembled system (from assemble_linear).
    freq : float
        Frequency in Hz.

    Returns
    -------
    sys : dict
        Updated system with A, B, nnWP, WPgvec keys.
    """
    if "bypass" not in sys:
        raise RuntimeError("Must call assemble_linear before assemble_waveguide_port")

    k0 = 2 * np.pi * freq / sys["c0"]
    sys["k"] = k0

    ndofs = sys["NDOFs"]
    n_wp = len(sys["WP"])
    wp_n_modes = sys.get("WPnModes", 1)
    n_rhs = n_wp * wp_n_modes

    # Scalar Helmholtz operator (matches MATLAB AssembWP.m)
    A = sys["S"] - k0**2 * sys["T"]

    # Interior DOFs (non-Dirichlet, non-WP)
    if "nnWP" not in sys:
        nn_wp = np.arange(ndofs)
        rem_id = []
        for key in list(sys.keys()):
            if key.startswith("Dir_") and isinstance(sys[key], np.ndarray):
                rem_id.extend(sys[key].tolist())
        for ip in range(n_wp):
            rem_id.extend(sys["WP"][ip])
        nn_wp = np.setdiff1d(nn_wp, rem_id)
        sys["nnWP"] = nn_wp
    else:
        nn_wp = sys["nnWP"]

    # Per-port quantities
    PP = np.zeros((n_rhs, n_rhs), dtype=complex)
    IP = np.zeros((len(nn_wp), n_rhs), dtype=complex)

    sys["WPgvec"] = [None] * n_wp

    for ip in range(n_wp):
        wp_dofs = sys["WP"][ip]
        wp_vec = sys["WPvec"][ip]
        n_modes = wp_vec.shape[1]

        if "Einc" in sys:
            wp_pow = np.diag(sys["Einc"]**2 * sys.get("Height", 1.0)
                             * np.emath.sqrt(1 - (sys["WPfc"][ip] / freq)**2) / sys["z0"])
        else:
            wp_pow = sys.get("WPpow", 1.0) * np.eye(n_modes) * 2 / sys.get("Height", 1.0)

        gamma = np.diag(1j * 2 * np.pi * freq / sys["c0"]
                        * np.emath.sqrt(1 - (np.array(sys["WPfc"][ip]) / freq)**2))
        gamma = np.abs(gamma.real) + 1j * gamma.imag

        A_wp = A[wp_dofs, :][:, wp_dofs]
        wp_gvec = wp_vec @ np.sqrt(1j * k0 * sys["z0"] * np.linalg.solve(gamma, wp_pow))
        sys["WPgvec"][ip] = wp_gvec

        idx = slice(ip * wp_n_modes, (ip + 1) * wp_n_modes)
        PP[idx, idx] = (wp_gvec.T @ A_wp @ wp_gvec
                        + 1j * k0 * sys["z0"] * wp_pow @ np.eye(n_modes))
        IP[:, idx] = A[nn_wp, :][:, wp_dofs] @ wp_gvec

    # Build block system
    PI = IP.T
    II = A[nn_wp, :][:, nn_wp]

    sys["A"] = sparse.bmat([
        [sparse.csc_matrix(PP), sparse.csc_matrix(PI)],
        [sparse.csc_matrix(IP), sparse.csc_matrix(II)]
    ], format="csc")

    # RHS
    B = np.zeros((n_rhs + len(nn_wp), n_rhs), dtype=complex)
    for ip in range(n_wp):
        if "Einc" in sys:
            wp_pow = np.diag(sys["Einc"]**2 * sys.get("Height", 1.0)
                             * np.emath.sqrt(1 - (sys["WPfc"][ip] / freq)**2) / sys["z0"])
        else:
            wp_pow = sys.get("WPpow", 1.0) * np.eye(wp_n_modes) * 2 / sys.get("Height", 1.0)
        idx = slice(ip * wp_n_modes, (ip + 1) * wp_n_modes)
        B[idx, idx] = 1j * k0 * sys["z0"] * 2 * wp_pow @ np.eye(wp_n_modes)

    sys["B"] = sparse.csc_matrix(B)

    # nWP for field reconstruction
    sys["nWP"] = [np.arange(ip * wp_n_modes, (ip + 1) * wp_n_modes)
                  for ip in range(n_wp)]

    return sys


def assemble_domain_decomposition(sys, mesh):
    """Assemble the domain decomposition system.

    Port of AssembDD.m / AssembSolveDD.m.

    Parameters
    ----------
    sys : dict
        System configuration.
    mesh : dict
        Mesh data.

    Returns
    -------
    sys : dict
        Updated system with DD matrices.
    mesh : dict
        Updated mesh.
    """
    from .boundary import get_bnd_map

    if "bypass" not in sys:
        sys, mesh = assemble_linear(sys, mesh)
        mesh, sys = get_bnd_map(sys, mesh)
        sys["Slin"] = sys["S"].copy()
        sys["Tlin"] = sys["T"].copy()
        sys["firstDD"] = True

    # Extract subdomain matrices
    bnd = sys["BndDoF"] - 1  # 0-based
    n_bnd = len(bnd)

    reg_ele = sys["RegEle"]
    reg_dof = sys["RegDoF"]
    reg_dof_map = sys["RegDoFmap"]
    ndofs = sys["NDOFs"]

    # Build Schur complement system
    # Interior blocks per region
    AII_list = []
    AIF_list = []
    AFI_list = []

    for ir in range(len(reg_ele)):
        int_dofs = reg_dof[ir] - 1  # 0-based
        # Interior matrix
        AII_list.append(sys["S"][int_dofs, :][:, int_dofs])
        # Interior-to-boundary coupling
        if len(int_dofs) > 0 and n_bnd > 0:
            AIF_list.append(sys["S"][int_dofs, :][:, bnd])
            AFI_list.append(sys["S"][bnd, :][:, int_dofs])
        else:
            AIF_list.append(None)
            AFI_list.append(None)

    AFF = sys["S"][bnd, :][:, bnd]

    sys["AII"] = AII_list
    sys["AIF"] = AIF_list
    sys["AFI"] = AFI_list
    sys["AFF"] = AFF
    sys["BndDoF_red"] = bnd

    return sys, mesh


def solve_dd_schur(sys):
    """Solve the domain decomposition system using Schur complement.

    Port of SolvDDschur.m.

    Parameters
    ----------
    sys : dict
        System with assembled DD matrices.

    Returns
    -------
    uF : ndarray
        Solution on the interface DOFs.
    """
    SF = sparse.csr_matrix(sys["AFF"].shape)
    for ir in range(len(sys["RegEle"])):
        AII = sys["AII"][ir]
        AIF = sys["AIF"][ir]
        AFI = sys["AFI"][ir]
        if AII is not None and AIF is not None:
            SF = SF - AFI @ sparse.linalg.spsolve(AII, AIF)

    SF = SF + sys["AFF"]
    gF = sys.get("gF", np.zeros(SF.shape[0]))
    uF = sparse.linalg.spsolve(SF, gF)
    return uF


def solve_direct(sys):
    """Direct solve of the assembled system.

    Port of the solve in ProjectElectrostatics.m.

    Parameters
    ----------
    sys : dict
        System with 'S' matrix and 'fs' RHS.

    Returns
    -------
    u : ndarray
        Solution vector.
    """
    A = sys.get("A", sys["S"])
    b = sys.get("b", sys["fs"])

    if "Dir" in sys:
        for ibc, dir_dofs in enumerate(sys["Dir"]):
            V = sys.get("V", [None] * (ibc + 1))[ibc]
            if V is not None:
                b = b - A[:, dir_dofs].dot(np.ones(len(dir_dofs)) * V)
            A = A.copy()
            A[dir_dofs, :] = 0
            A[:, dir_dofs] = 0
            A[dir_dofs, dir_dofs] = np.eye(len(dir_dofs))
            if V is not None:
                b[dir_dofs] = V

    u = sparse.linalg.spsolve(A, b)
    return u
