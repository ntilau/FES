"""3D H(curl) curl-curl assembly with Kerr nonlinearity (harmonic balance).

Solves the nonlinear curl-curl problem where the permittivity of a
designated material depends on the electric-field intensity via the Kerr
model::

    epsr_eff = epsr_base + kerr * |E|^2

A fixed-point iteration with relaxation updates the Kerr-shifted
permittivity until convergence, re-assembling and re-solving the
:mod:`fes.core.assemblers.em_e_fd` system at each step.

With ``kerr == 0`` or no matching nonlinear material this reduces to a
straight :func:`assemble_em_e_fd` solve.  With ``n_harm > 1`` the same
Kerr-shifted permittivity is applied to all harmonics (diagonal harmonic
coupling, mirroring :mod:`fes.core.harmonic_balance`); the driver currently
solves at the fundamental frequency only.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ...constants import C0, PI
from ..dof import vector_dof_map
from ..element_matrix import _DNS_REF_3D
from ..shape3d import shape_3d
from .em_e_fd import (
    FDPort,
    FDSystem,
    assemble_em_e_fd,
    NullRightHandSideError,
)

__all__ = ["FDPort", "FDSystem", "assemble_em_e_fd_nl"]

#: Fixed-point iteration tolerance on the Kerr-shifted permittivity.
_KERR_TOL = 1e-6
#: Maximum fixed-point iterations.
_KERR_MAX_ITER = 50


def _find_nl_label(mesh, nl_mtrl_name: str) -> Optional[int]:
    """Return the tet label of the nonlinear material, or ``None``."""
    if not nl_mtrl_name:
        return None
    for label, solid in mesh.materials.items():
        if solid.name == nl_mtrl_name or solid.material == nl_mtrl_name:
            return int(label)
    return None


def _kerr_epsr(result, mesh, nl_label: int, base_epsr: float, kerr: float,
               order: int) -> float:
    """Compute ``base_epsr + kerr * mean(|E|^2)`` over nonlinear tets.

    The electric field is evaluated at the tetrahedron barycentre from the
    H(curl) solution vector using the reference edge basis functions.
    """
    tet_ids = np.flatnonzero(mesh.tet_labels == nl_label)
    if tet_ids.size == 0 or result.Sol is None:
        return base_epsr

    # Reference barycentre (x=y=z=1/4).
    centre = np.array([[0.25, 0.25, 0.25]])
    total_E_sq = 0.0
    count = 0

    for tet_id in tet_ids:
        geo = mesh.tet_geo(int(tet_id))
        J = _DNS_REF_3D @ geo
        inv_j = np.linalg.inv(J).T
        sh = shape_3d(order, centre, inv_j)
        Nv = sh.Nv[0]  # (numv, 3)

        v = vector_dof_map(mesh, order, int(tet_id))
        E = np.zeros(3, dtype=complex)
        sol_col = result.Sol[:, 0] if result.Sol.ndim == 2 else result.Sol
        for i in range(len(v)):
            E += sol_col[v[i]] * Nv[i]

        total_E_sq += float(np.sum(np.abs(E) ** 2))
        count += 1

    if count == 0:
        return base_epsr

    mean_E_sq = total_E_sq / count
    return base_epsr + kerr * mean_E_sq


def assemble_em_e_fd_nl(
    mesh,
    freq: float,
    order: int = 2,
    *,
    power: float = 1.0,
    solve: bool = True,
    e_inc: Optional[Sequence[float]] = None,
    k_inc: Optional[Sequence[float]] = None,
    n_harm: int = 1,
    nl_mtrl_name: str = "",
    kerr: float = 0.0,
    relax: float = 0.5,
) -> FDSystem:
    """Assemble and solve the 3D curl-curl system with Kerr nonlinearity.

    Fixed-point iteration: at each step the Kerr-shifted permittivity
    ``epsr_base + kerr * |E|^2`` is assigned to the nonlinear material,
    the :mod:`fes.core.assemblers.em_e_fd` system is assembled and solved,
    and the field is recomputed.  Relaxation damps the update::

        epsr_new = (1 - relax) * epsr_old + relax * (epsr_base + kerr * |E|^2)

    With ``kerr == 0`` or no matching nonlinear material this falls back to
    a plain :func:`assemble_em_e_fd` solve.

    Returns an :class:`FDSystem` with the same fields as
    :func:`assemble_em_e_fd`.
    """
    order = int(order)
    freq = float(freq)
    n_harm = max(int(n_harm), 1)

    # Locate the nonlinear material.
    nl_label = _find_nl_label(mesh, nl_mtrl_name)

    if nl_label is None or kerr == 0.0:
        return assemble_em_e_fd(
            mesh, freq, order, power=power, solve=solve,
            e_inc=e_inc, k_inc=k_inc,
        )

    solid = mesh.materials[nl_label]
    base_epsr = float(solid.epsr)
    epsr_current = base_epsr

    result = None
    for _ in range(_KERR_MAX_ITER):
        # Temporarily assign the Kerr-shifted permittivity.
        solid.epsr = epsr_current

        result = assemble_em_e_fd(
            mesh, freq, order, power=power, solve=solve,
            e_inc=e_inc, k_inc=k_inc,
        )

        if not solve:
            break

        # Compute the Kerr-shifted permittivity from the field energy.
        epsr_new = _kerr_epsr(result, mesh, nl_label, base_epsr, kerr, order)

        # Convergence check.
        if abs(epsr_new - epsr_current) < _KERR_TOL * max(1.0, abs(base_epsr)):
            epsr_current = epsr_new
            break

        # Damped update.
        epsr_current = (1.0 - relax) * epsr_current + relax * epsr_new

    # Restore the original material.
    solid.epsr = base_epsr

    if result is None:
        raise NullRightHandSideError("Null Right Hand Side")

    return result
