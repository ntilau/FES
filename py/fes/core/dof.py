"""Degree of freedom numbering and positioning utilities.

Ports of CalcDoFsNumber.m, CalcDoFsPositions.m, CalcGlobIndex.m,
and CalcOrderMatSize.m.
"""

import numpy as np


def calc_order_mat_size(order):
    """Number of scalar and vector DOFs per element for given polynomial order.

    Parameters
    ----------
    order : int
        Polynomial order (1-4).

    Returns
    -------
    nums : int
        Scalar DOFs per element.
    numv : int
        Vector DOFs per element.
    """
    nums = (order + 1) * (order + 2) // 2
    numv = order * (order + 2)
    return nums, numv


def calc_dofs_number(sys, mesh):
    """Count total degrees of freedom in the system.

    Port of CalcDoFsNumber.m.

    Parameters
    ----------
    sys : dict
        System configuration (must contain 'pOrd').
    mesh : dict
        Mesh data (must contain NNODE, NSPIG, NELE).

    Returns
    -------
    ndofs : int
        Number of scalar DOFs.
    ndofv : int
        Number of vector DOFs.
    """
    p = sys["pOrd"]
    ndofs = (mesh["NNODE"] + mesh["NSPIG"] * (p - 1)
             + mesh["NELE"] * (0 < (p - 2)) * (p - 2) * (p - 1) * 0.5)
    ndofs = int(ndofs)

    ndofv = int(p * mesh["NSPIG"] * (1 + (p > 2))
                + 2 * mesh["NELE"] * (p > 1) * (p - 1))
    return ndofs, ndofv


def calc_glob_index(dim, p_ord, mesh, ie, is_edge=None):
    """Compute global index arrays for element ie.

    Port of CalcGlobIndex.m.

    Parameters
    ----------
    dim : int
        Dimension (1 for edge, 2 for element interior).
    p_ord : int
        Polynomial order.
    mesh : dict
        Mesh data.
    ie : int
        Element index (0-based).
    is_edge : int, optional
        Local edge index (1-based). Only used when dim=1.

    Returns
    -------
    gIs : ndarray
        Global scalar DOF indices.
    gIv : ndarray
        Global vector DOF indices (only for dim=2).
    """
    NNODE = mesh["NNODE"]
    NELE = mesh["NELE"]
    NSPIG = mesh["NSPIG"]

    if dim == 1:
        # Edge-based indexing
        gIv = abs(mesh["spig"][ie, is_edge - 1])
        n_idx = mesh["spig2"][gIv - 1, :] + 1  # spig2 is 0-based → 1-based
        if p_ord == 1:
            gIs = n_idx.copy()
        elif p_ord == 2:
            gIs = np.concatenate([n_idx, [NNODE + gIv]])
        elif p_ord == 3:
            gIs = np.concatenate([n_idx, [NNODE + gIv, NNODE + NSPIG + gIv]])
        elif p_ord == 4:
            gIs = np.concatenate([
                n_idx,
                [NNODE + gIv],
                [NNODE + NSPIG + gIv],
                [NNODE + 2 * NSPIG + NELE + gIv]
            ])
        else:
            raise ValueError(f"Invalid order {p_ord}")
        return gIs, None

    elif dim == 2:
        # Element-based indexing
        # ele is 0-based in mesh; add 1 for 1-based indexing (matching MATLAB)
        ele = mesh["ele"][ie, :] + 1
        spig = np.abs(mesh["spig"][ie, :])

        if p_ord == 1:
            gIs = ele.copy()
            gIv = spig.copy()
        elif p_ord == 2:
            gIs = np.concatenate([ele, NNODE + spig])
            gIv = np.concatenate([
                spig,
                mesh["NSPIG"] + spig,
                [2 * mesh["NSPIG"] + ie + 1],
                [2 * mesh["NSPIG"] + mesh["NELE"] + ie + 1]
            ])
        elif p_ord == 3:
            gIs = np.concatenate([
                ele, NNODE + spig,
                NNODE + NSPIG + spig,
                [NNODE + 2 * NSPIG + ie + 1]
            ])
            gIv = np.concatenate([
                spig,
                mesh["NSPIG"] + spig,
                [2 * mesh["NSPIG"] + ie + 1],
                [2 * mesh["NSPIG"] + mesh["NELE"] + ie + 1],
                2 * mesh["NSPIG"] + 2 * mesh["NELE"] + spig,
                [3 * mesh["NSPIG"] + 2 * mesh["NELE"] + ie + 1],
                [3 * mesh["NSPIG"] + 3 * mesh["NELE"] + ie + 1],
                [3 * mesh["NSPIG"] + 4 * mesh["NELE"] + ie + 1],
                [3 * mesh["NSPIG"] + 5 * mesh["NELE"] + ie + 1]
            ])
        elif p_ord == 4:
            gIs = np.concatenate([
                ele, NNODE + spig,
                NNODE + NSPIG + spig,
                [NNODE + 2 * NSPIG + ie + 1],
                NNODE + 2 * NSPIG + NELE + spig,
                [NNODE + 3 * NSPIG + NELE + ie + 1],
                [NNODE + 3 * NSPIG + 2 * NELE + ie + 1]
            ])
            gIv = np.concatenate([
                spig,
                mesh["NSPIG"] + spig,
                [2 * mesh["NSPIG"] + ie + 1],
                [2 * mesh["NSPIG"] + mesh["NELE"] + ie + 1],
                2 * mesh["NSPIG"] + 2 * mesh["NELE"] + spig,
                [3 * mesh["NSPIG"] + 2 * mesh["NELE"] + ie + 1],
                [3 * mesh["NSPIG"] + 3 * mesh["NELE"] + ie + 1],
                [3 * mesh["NSPIG"] + 4 * mesh["NELE"] + ie + 1],
                [3 * mesh["NSPIG"] + 5 * mesh["NELE"] + ie + 1]
            ])
        else:
            raise ValueError(f"Invalid order {p_ord}")

        return gIs.astype(int), gIv.astype(int)
    else:
        raise ValueError(f"Invalid dim {dim}")


def calc_dofs_position(sys, mesh):
    """Compute physical positions of DOFs and refined elements.

    Port of CalcDoFsPositions.m.

    Parameters
    ----------
    sys : dict
        System configuration (must contain 'pOrd', 'NDOFs').
    mesh : dict
        Mesh data.

    Returns
    -------
    mesh : dict
        Updated mesh with 'refNode' and 'refEle'.
    """
    p = sys["pOrd"]
    ndofs = sys["NDOFs"]
    nele = mesh["NELE"]

    ref_node = np.zeros((ndofs, 3))
    ref_ele = np.zeros((nele * p**2, 3), dtype=int)

    node = mesh["node"]
    ele = mesh["ele"]

    for ie in range(nele):
        gIs, _ = calc_glob_index(2, p, mesh, ie)
        n_idx = ele[ie, :]           # 0-based
        s_idx = np.abs(mesh["spig"][ie, :])  # 1-based

        gIs0 = (gIs.astype(int) - 1)  # convert 1-based → 0-based
        n_idx0 = n_idx                # already 0-based

        ref_node[gIs0[:3], :2] = node[n_idx0, :]

        spig2 = mesh["spig2"]          # already 0-based
        spig2_0 = spig2[s_idx - 1, :]  # edge node indices, 0-based

        if p == 1:
            ref_ele[ie, :] = gIs0[:3]

        elif p == 2:
            mid = 0.5 * node[spig2_0[:, 0], :] + 0.5 * node[spig2_0[:, 1], :]
            ref_node[gIs0[3:6], :2] = mid
            ref_ele[4 * ie + 0, :] = gIs0[[0, 4, 5]]
            ref_ele[4 * ie + 1, :] = gIs0[[3, 4, 5]]
            ref_ele[4 * ie + 2, :] = gIs0[[1, 3, 5]]
            ref_ele[4 * ie + 3, :] = gIs0[[2, 3, 4]]

        elif p == 3:
            ref_node[gIs0[3:6], :2] = (2/3 * node[spig2_0[:, 0], :]
                                       + 1/3 * node[spig2_0[:, 1], :])
            ref_node[gIs0[6:9], :2] = (1/3 * node[spig2_0[:, 0], :]
                                       + 2/3 * node[spig2_0[:, 1], :])
            ref_node[gIs0[9], :2] = (1/3 * node[n_idx0[0], :]
                                     + 1/3 * node[n_idx0[1], :]
                                     + 1/3 * node[n_idx0[2], :])
            ref_ele[9 * ie + 0, :] = gIs0[[0, 5, 4]]
            ref_ele[9 * ie + 1, :] = gIs0[[5, 4, 9]]
            ref_ele[9 * ie + 2, :] = gIs0[[5, 8, 9]]
            ref_ele[9 * ie + 3, :] = gIs0[[3, 8, 9]]
            ref_ele[9 * ie + 4, :] = gIs0[[1, 3, 8]]
            ref_ele[9 * ie + 5, :] = gIs0[[3, 6, 9]]
            ref_ele[9 * ie + 6, :] = gIs0[[7, 6, 9]]
            ref_ele[9 * ie + 7, :] = gIs0[[4, 7, 9]]
            ref_ele[9 * ie + 8, :] = gIs0[[2, 7, 6]]

        elif p == 4:
            ref_node[gIs0[3:6], :2] = (0.75 * node[spig2_0[:, 0], :]
                                       + 0.25 * node[spig2_0[:, 1], :])
            ref_node[gIs0[6:9], :2] = (0.5 * node[spig2_0[:, 0], :]
                                       + 0.5 * node[spig2_0[:, 1], :])
            ref_node[gIs0[10:13], :2] = (0.25 * node[spig2_0[:, 0], :]
                                         + 0.75 * node[spig2_0[:, 1], :])
            ref_node[gIs0[9], :2] = (0.5 * node[n_idx0[0], :]
                                     + 0.25 * node[n_idx0[1], :]
                                     + 0.25 * node[n_idx0[2], :])
            ref_node[gIs0[13], :2] = (0.25 * node[n_idx0[0], :]
                                      + 0.5 * node[n_idx0[1], :]
                                      + 0.25 * node[n_idx0[2], :])
            ref_node[gIs0[14], :2] = (0.25 * node[n_idx0[0], :]
                                      + 0.25 * node[n_idx0[1], :]
                                      + 0.5 * node[n_idx0[2], :])

            ref_ele[16 * ie + 0, :] = gIs0[[0, 4, 5]]
            ref_ele[16 * ie + 1, :] = gIs0[[4, 5, 9]]
            ref_ele[16 * ie + 2, :] = gIs0[[5, 8, 9]]
            ref_ele[16 * ie + 3, :] = gIs0[[8, 9, 13]]
            ref_ele[16 * ie + 4, :] = gIs0[[8, 12, 13]]
            ref_ele[16 * ie + 5, :] = gIs0[[3, 12, 13]]
            ref_ele[16 * ie + 6, :] = gIs0[[1, 3, 12]]
            ref_ele[16 * ie + 7, :] = gIs0[[3, 6, 13]]
            ref_ele[16 * ie + 8, :] = gIs0[[6, 13, 14]]
            ref_ele[16 * ie + 9, :] = gIs0[[9, 13, 14]]
            ref_ele[16 * ie + 10, :] = gIs0[[7, 9, 14]]
            ref_ele[16 * ie + 11, :] = gIs0[[7, 9, 4]]
            ref_ele[16 * ie + 12, :] = gIs0[[7, 11, 14]]
            ref_ele[16 * ie + 13, :] = gIs0[[10, 11, 14]]
            ref_ele[16 * ie + 14, :] = gIs0[[6, 10, 14]]
            ref_ele[16 * ie + 15, :] = gIs0[[2, 10, 11]]

    ref_ele = np.sort(ref_ele, axis=1)

    mesh["refNode"] = ref_node
    mesh["refEle"] = ref_ele
    return mesh
