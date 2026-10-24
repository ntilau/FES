"""Far-field radiation export ported from ``cpp/src/radiation.cpp``.

The original ``rad`` class is constructed by ``post_processor::save_data``
after the field export when ``+rad`` is given.  It evaluates the far field of
the H(curl) solution on a ``n_theta x n_phi`` spherical grid and writes
``<name>_<freq>_rad.vtk``.

Reproduced verbatim, including the details that make the numbers what they
are:

* the far-field quadrature is ``quad(p_ord)`` -- one order *lower* than the
  assembly rule ``quad(p_ord + 1)``;
* ``fullSol = sum(sol, 1) * sqrt(2)`` sums the solution columns before the
  element gather;
* the reference triangle each face is evaluated on is chosen by the local
  index of the tetrahedron vertex across the face (``RefFace``): case 0 is the
  face ``x + y + z = 1`` and cases 1-3 are ``x = 0``, ``y = 0``, ``z = 0``;
* the face normal is ``RefNorm = -n`` (``n`` points away from the interior
  vertex), and ``arma::dot(a, b)`` does *not* conjugate (its complex kernel
  computes ``sum(a_i * b_i)``; ``arma::cdot`` is the conjugating variant), so
  it maps to ``numpy.dot``, not ``numpy.vdot``.  This matters in
  ``dot(Je, rDir)``: conjugating ``Je`` would leave the far field with a
  component along the observation direction;
* ``Prad`` accumulates ``0.5*Re(dot(cross(Jm, conj(Je)), RefNorm))``;
* the far field uses ``G = exp(i sqrt(mur epsr) kv rv.GlobPnt) *
  i sqrt(mur epsr) kv / (4 pi)`` and the ``z0 sqrt(mur/epsr)`` term around
  the equivalent currents.

``rad`` is a 3D post-processor, but the reference constructs it for every
formulation when ``+rad`` is given.  On a 2D mesh the face list is empty, so
``Prad`` stays zero and every derived quantity degenerates exactly as the
reference's does (``DirCoeff = inf``, ``Dir = nan``); the writer is shared.

The writer follows ``rad::Savefield`` line for line, including the
``(float) rMag * real(Dir)`` cast that rounds *only* ``rMag`` before the
double multiply, the cells block that never emits a newline, the
``SCALARS RCS_[dB]`` name under ``+einc`` (else ``Dir_[dB]``), and the
six-significant-digit formatting of the default ``std::ostream``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from ..constants import C0, MU0, PI, Z0
from ..core.dof import vector_dof_map
from ..core.element_matrix import (
    Material,
    face_axes,
    tetrahedron_jacobian,
    triangle_jacobian,
)
from ..core.quadrature import quadrature_2d
from ..core.shape3d import shape_3d
from .field import _f32, freq_tag

__all__ = ["RadiationPattern", "radiation_pattern", "write_radiation_vtk"]

#: ``RefGeo`` per ``RefFace`` case, exactly as ``radiation.cpp`` fills it.
#: Row 0 is the constant part, rows 1 and 2 multiply ``xi0`` and ``xi1``.
_REF_GEO = (
    np.array([[1.0, 0.0, 0.0], [-1.0, 1.0, 0.0], [-1.0, 0.0, 1.0]]),
    np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
    np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
)


@dataclass
class RadiationPattern:
    """``rad``'s numerical state after ``Savefield``'s pre-write block."""

    Prad: float
    DirCoeff: float
    delta: float
    n_theta: int
    n_phi: int
    Ef: np.ndarray
    Dir: np.ndarray
    einc: bool = False

    @property
    def magE(self) -> np.ndarray:
        """``arma::norm(Ef.row(i), 2)`` for every direction."""
        return np.linalg.norm(self.Ef, axis=1)

    @property
    def max_dir(self) -> float:
        """``10*log10(DirCoeff*max|E|^2)``, the ``maxDir`` stdout line."""
        with np.errstate(divide="ignore", invalid="ignore"):
            return float(
                10.0 * np.log10(self.DirCoeff * float(np.max(self.magE)) ** 2)
            )


def _linspace(start: float, end: float, num: int) -> np.ndarray:
    """``arma::linspace`` (note ``num == 1`` returns *end*, not *start*)."""
    num = int(num)
    if num == 1:
        return np.array([float(end)])
    if num < 2:
        return np.zeros(0)
    delta = (end - start) / float(num - 1)
    values = start + np.arange(num - 1, dtype=float) * delta
    return np.append(values, float(end))


def _direction_grid(n_theta: int, n_phi: int) -> np.ndarray:
    """Return the ``(n_theta*n_phi, 3)`` ``Dir`` matrix in oracle order."""
    thetas = _linspace(0.0, 180.0, n_theta)
    phis = _linspace(0.0, 360.0, n_phi)
    theta = thetas[:, None] / 180.0 * PI
    phi = phis[None, :] / 180.0 * PI
    sin_theta = np.sin(theta)
    direction = np.empty((thetas.size, phis.size, 3))
    direction[..., 0] = sin_theta * np.cos(phi)
    direction[..., 1] = sin_theta * np.sin(phi)
    direction[..., 2] = np.cos(theta)
    return direction.reshape(thetas.size * phis.size, 3)


def radiation_pattern(
    mesh,
    sol: np.ndarray,
    freq: float,
    *,
    order: int,
    n_theta: float,
    n_phi: float,
    e_inc: Optional[Sequence[float]] = None,
) -> RadiationPattern:
    """Evaluate the far field exactly like the ``rad`` constructor.

    *sol* is the full DOF solution ``(n_dof, n_columns)`` and *e_inc* the raw
    ``+einc`` electric-field vector (``opt->E``, not divided by ``sqrt(2)``).
    """
    sol = np.asarray(sol)
    full = np.sum(sol, axis=1) * np.sqrt(2.0)

    n_theta = int(n_theta)
    n_phi = int(n_phi)
    direction = _direction_grid(n_theta, n_phi)
    n_dirs = direction.shape[0]

    ef = np.zeros((n_dirs, 3), dtype=complex)
    prad = 0.0
    h_const = complex(0.0, -1.0 / (2.0 * PI * freq * MU0))
    kv = 2.0 * PI * freq / C0
    points, weights = quadrature_2d(order)
    # A 2D mesh has no 3D face topology; ``rad`` then finds no faces at all.
    faces_for_boundary = getattr(mesh, "faces_for_boundary", None)

    for bc in mesh.boundaries:
        if bc.type != "Radiation" or faces_for_boundary is None:
            continue
        for fid in faces_for_boundary(bc):
            fid = int(fid)
            adjacent = mesh.face_adj_tet[fid]
            tet_id = int(adjacent[0])
            material = Material.from_solid(mesh.material_for_tet(tet_id))
            epsr = complex(material.epsr, material.epsr2)
            mur = material.mur

            inv_j = tetrahedron_jacobian(mesh.tet_geo(tet_id))[1]
            t_sol = full[vector_dof_map(mesh, order, tet_id)]
            geo = mesh.fac_geo(fid)
            int_node, ref_face = mesh.int_node_ref(fid)
            ref_geo = _REF_GEO[ref_face]

            _, _, normal, c_geo2 = face_axes(geo, int_node)
            ref_norm = -normal
            det_j2 = triangle_jacobian(c_geo2)[0]

            v0 = geo[0]
            v1 = geo[1] - v0
            v2 = geo[2] - v0

            for iq in range(points.shape[0]):
                xi0, xi1 = points[iq]
                loc = ref_geo[0] + ref_geo[1] * xi0 + ref_geo[2] * xi1
                glob = v0 + v1 * xi0 + v2 * xi1

                shp = shape_3d(order, loc[None, :], inv_j)
                jm = np.cross(ref_norm, shp.Nv[0].T @ t_sol)
                je = np.cross(ref_norm, h_const * (shp.dNv[0].T @ t_sol))

                cross_jm_je = np.cross(jm, np.conj(je))
                # ``arma::dot`` does not conjugate; ``RefNorm`` is real, so
                # the real part is the same as with ``np.vdot``.
                prad += (
                    0.5
                    * float(np.real(np.dot(cross_jm_je, ref_norm)))
                    * weights[iq]
                    * det_j2
                )

                root = np.sqrt(mur * epsr)
                phase = np.exp(1j * root * kv * (direction @ glob))
                g = phase * (1j * root * kv / 4.0 / PI)
                # ``arma::dot(Je, rDir)`` is ``sum(Je_i * rDir_i)``: the
                # oracle's stub does not conjugate (only ``cdot`` does).
                dot_je = je @ direction.T
                cross_jm = np.cross(np.broadcast_to(jm, direction.shape), direction)
                correction = (
                    Z0
                    * np.sqrt(mur / epsr)
                    * (dot_je[:, None] * direction - je[None, :])
                    - cross_jm
                )
                ef -= (g[:, None] * correction) * (weights[iq] * det_j2)

    prad = abs(prad)
    mag_e = np.linalg.norm(ef, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        delta = 20.0 * np.log10(np.max(mag_e) - np.min(mag_e))
    einc = e_inc is not None
    if einc:
        pol = np.asarray(e_inc, dtype=float)
        dir_coeff = (2.0 * np.sqrt(PI) / np.linalg.norm(pol)) ** 2
        with np.errstate(divide="ignore", invalid="ignore"):
            delta = -10.0 * np.log10(dir_coeff * np.min(mag_e) ** 2)
    else:
        # IEEE division: a portless 2D mesh has no faces and ``Prad == 0``,
        # which the oracle turns into ``+inf`` rather than raising.
        denominator = prad * Z0
        dir_coeff = (
            2.0 * PI / denominator if denominator != 0.0 else float("inf")
        )

    return RadiationPattern(
        Prad=float(prad),
        DirCoeff=float(dir_coeff),
        delta=float(delta),
        n_theta=n_theta,
        n_phi=n_phi,
        Ef=ef,
        Dir=direction,
        einc=einc,
    )


def _six(value: float) -> str:
    """``std::ostream << double``: six significant digits."""
    return f"{float(value):.6g}"


def _render(pattern: RadiationPattern) -> str:
    """Return the ``rad::Savefield`` text for *pattern*."""
    out: List[str] = []
    add = out.append
    add("# vtk DataFile Version 2.0\n")
    add("radiation data\n")
    add("ASCII\n")
    add("DATASET UNSTRUCTURED_GRID\n")
    add(f"POINTS {pattern.Dir.shape[0]} float \n")

    with np.errstate(divide="ignore", invalid="ignore"):
        db = 10.0 * np.log10(pattern.DirCoeff * np.abs(pattern.magE) ** 2)
    for index in range(pattern.Dir.shape[0]):
        # ``(float) rMag * real(Dir(i,k))``: the cast binds to rMag only, so
        # the product stays a double and prints at six significant digits.
        r_mag = float(np.float32(float(db[index]) + pattern.delta))
        x, y, z = pattern.Dir[index]
        add(f"{_six(r_mag * float(x))} {_six(r_mag * float(y))} "
            f"{_six(r_mag * float(z))}\n")

    n_theta, n_phi = pattern.n_theta, pattern.n_phi
    cells = (n_theta - 1) * (n_phi - 1)
    add(f"CELLS {cells} {cells * 5}\n")
    # The oracle emits the whole block without newlines: each cell ends with a
    # trailing space, so ``CELL_TYPES`` continues on the same line.
    add(
        "".join(
            f"4 {i * n_phi + j} {i * n_phi + j + 1} "
            f"{(i + 1) * n_phi + j + 1} {(i + 1) * n_phi + j} "
            for i in range(n_theta - 1)
            for j in range(n_phi - 1)
        )
    )
    add(f"CELL_TYPES {cells}\n")
    out.extend("7\n" for _ in range(cells))

    add(f"POINT_DATA {pattern.Ef.shape[0]}\n")
    if pattern.einc:
        add("SCALARS RCS_[dB] float 1\n")
    else:
        add("SCALARS Dir_[dB] float 1\n")
    add("LOOKUP_TABLE jet\n")
    out.extend(f"{_six(value)}\n" for value in db)

    add("SCALARS E_far_norm_[V/m] float 1\n")
    add("LOOKUP_TABLE jet\n")
    for index in range(pattern.Ef.shape[0]):
        add(f"{_f32(np.linalg.norm(np.abs(pattern.Ef[index])))}\n")

    for name, values in (
        ("E_far_abs_[V/m]", np.abs(pattern.Ef)),
        ("E_far_real_[V/m]", np.real(pattern.Ef)),
        ("E_far_imag_[V/m]", np.imag(pattern.Ef)),
    ):
        add(f"VECTORS {name} float\n")
        for x, y, z in values:
            add(f"{_f32(x)} {_f32(y)} {_f32(z)}\n")
    return "".join(out)


def write_radiation_vtk(
    mesh,
    sol: np.ndarray,
    freq: float,
    field_name: str | Path,
    *,
    order: int,
    n_theta: float,
    n_phi: float,
    e_inc: Optional[Sequence[float]] = None,
) -> Path:
    """Write ``<field_name>_<freq>_rad.vtk`` exactly like ``rad::Savefield``."""
    pattern = radiation_pattern(
        mesh,
        sol,
        freq,
        order=order,
        n_theta=n_theta,
        n_phi=n_phi,
        e_inc=e_inc,
    )
    target = Path(f"{field_name}_{freq_tag(freq)}_rad.vtk")
    target.write_text(_render(pattern))
    return target
