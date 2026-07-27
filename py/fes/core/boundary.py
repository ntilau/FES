"""Boundary condition mapping for domain decomposition.

Port of GetBndMap.m.
"""

import numpy as np
from .dof import calc_glob_index, calc_dofs_number


def get_bnd_map(sys, mesh):
    """Compute boundary DOF mapping for domain decomposition.

    Port of GetBndMap.m.

    Parameters
    ----------
    sys : dict
        System configuration.
    mesh : dict
        Mesh data.

    Returns
    -------
    mesh : dict
        Updated mesh with region element/DOF lists.
    sys : dict
        Updated system with DDmap, BndDoF, RegEle, RegDoF, RegDoFmap.
    """
    ids_dd = np.where(mesh["slab"] == mesh["BC"]["DDschur"])[0] + 1

    unique_elab = np.unique(mesh["elab"])
    n_reg = len(unique_elab)

    reg_ele = [[] for _ in range(n_reg)]
    reg_dof = [[] for _ in range(n_reg)]
    reg_dof_map = [[] for _ in range(n_reg)]
    bnd_dof = []

    for ie in range(mesh["NELE"]):
        gIs, _ = calc_glob_index(2, sys["pOrd"], mesh, ie)
        tmp = mesh["elab"][ie] - 1  # 0-based
        reg_ele[tmp] = list(set(reg_ele[tmp] + [ie]))
        reg_dof[tmp] = list(set(reg_dof[tmp] + gIs.tolist()))

        on_dd = [
            np.sum(np.abs(mesh["spig"][ie, 0]) == ids_dd),
            np.sum(np.abs(mesh["spig"][ie, 1]) == ids_dd),
            np.sum(np.abs(mesh["spig"][ie, 2]) == ids_dd)
        ]

        if np.sum(on_dd) > 0:
            id_on_dd = [i for i, v in enumerate(on_dd) if v > 0]
            for i in id_on_dd:
                gIs, _ = calc_glob_index(1, sys["pOrd"], mesh, ie, i + 1)
                bnd_dof.extend(gIs.tolist())

    bnd_dof = np.unique(bnd_dof).astype(int)

    ndofs, _ = calc_dofs_number(sys, mesh)
    dd_map = np.zeros(ndofs, dtype=int)
    roof = len(bnd_dof)

    reg_order = list(range(1, n_reg + 1))
    for ir in reg_order:
        ir0 = ir - 1
        dofs = np.array(reg_dof[ir0], dtype=int)
        dofs = np.setdiff1d(dofs, bnd_dof)
        reg_dof[ir0] = dofs
        reg_dof_map[ir0] = roof + np.arange(1, len(dofs) + 1)
        dd_map[dofs - 1] = reg_dof_map[ir0]
        roof = roof + len(dofs)

    dd_map[bnd_dof - 1] = np.arange(1, len(bnd_dof) + 1)

    sys["DDmap"] = dd_map
    sys["BndDoF"] = bnd_dof
    sys["RegEle"] = reg_ele
    sys["RegDoF"] = reg_dof
    sys["RegDoFmap"] = reg_dof_map

    return mesh, sys
