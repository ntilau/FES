"""Tests for the bundled near-field to far-field library (:mod:`n2f`).

The package is bundled under the MIT license; these tests cover the
geometry builders, coordinate transforms, utilities, the scalar Huygens
transformation (checked against the direct point-source far field) and the
vector far-field reference solver.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import n2f

WAVELENGTH = 1.0


def _box_setup(
    n_elems: int = 3,
    spacing: float = 0.5,
    ext: float = 0.25,
    sampling: float = 0.25,
) -> dict:
    """Build an array, a closed sampling box and the scalar near field on it."""
    wavelength = WAVELENGTH
    array_pos = n2f.build_array(wavelength, n_elems, spacing, n_elems, spacing)
    dims = n2f.get_box_dim(wavelength, array_pos, ext, sampling, ext)
    box_pos, box_n, dS, _ = n2f.build_box([1] * 6, *dims, 1, 0, 0)
    r_mag, n_dot_r = n2f.get_box_vectors(array_pos, box_pos, box_n)
    excitation = n2f.sf_excitations(wavelength, array_pos, 0, 0)

    with warnings.catch_warnings():
        # The upstream solver stores the complex integral in a real array.
        warnings.simplefilter("ignore")
        psi, del_psi = n2f.sf_nf_solver(wavelength, excitation, r_mag, n_dot_r)

    return {
        "array_pos": array_pos,
        "box_pos": box_pos,
        "box_n": box_n,
        "dS": dS,
        "r_mag": r_mag,
        "n_dot_r": n_dot_r,
        "excitation": excitation,
        "psi": psi,
        "del_psi": del_psi,
    }


@pytest.fixture(scope="module")
def box_setup() -> dict:
    return _box_setup()


# ---------------------------------------------------------------------------
# Package surface and transforms
# ---------------------------------------------------------------------------


def test_package_surface() -> None:
    for name in (
        "geometry",
        "transforms",
        "scalar",
        "vector",
        "utils",
        "build_array",
        "sf_nf2ff_solver",
        "vf_nf2ff_solver",
    ):
        assert hasattr(n2f, name)
    assert callable(n2f.build_array)
    assert callable(n2f.sf_nf2ff_solver)
    assert callable(n2f.vf_nf2ff_solver)
    assert n2f.__version__ == "1.0.0"


def test_angle_conversions() -> None:
    assert np.isclose(n2f.deg2rad(90), np.pi / 2)
    assert np.isclose(n2f.rad2deg(np.pi / 2), 90)
    angles = np.array([0.0, 30.0, 90.0, 180.0, 359.0])
    assert np.allclose(n2f.rad2deg(n2f.deg2rad(angles)), angles)


def test_coordinate_transform_round_trip() -> None:
    rng = np.random.default_rng(1)
    ax, ay, az = (rng.normal(size=4) for _ in range(3))
    theta = rng.uniform(0, np.pi, size=4)
    phi = rng.uniform(-np.pi, np.pi, size=4)

    ar, at, ap = n2f.cartesian2spherical(ax, ay, az, theta, phi)
    bx, by, bz = n2f.spherical2cartesian(ar, at, ap, theta, phi)

    assert np.allclose([bx, by, bz], [ax, ay, az])
    # The radial component is the projection onto the observation direction.
    r_hat = np.stack(
        [
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ]
    )
    assert np.allclose(ar, np.sum(r_hat * np.stack([ax, ay, az]), axis=0))
    # The theta and phi components are orthogonal to the radial one.
    t_hat = np.stack(
        [
            np.cos(theta) * np.cos(phi),
            np.cos(theta) * np.sin(phi),
            -np.sin(theta),
        ]
    )
    p_hat = np.stack([-np.sin(phi), np.cos(phi), np.zeros_like(phi)])
    assert np.allclose(at, np.sum(t_hat * np.stack([ax, ay, az]), axis=0))
    assert np.allclose(ap, np.sum(p_hat * np.stack([ax, ay, az]), axis=0))


def test_cross_operator_matches_numpy() -> None:
    rng = np.random.default_rng(2)
    a = rng.normal(size=(3, 5))
    b = rng.normal(size=(3, 5))
    cx, cy, cz = n2f.cross_operator(*a, *b)
    assert np.allclose([cx, cy, cz], np.cross(a, b, axis=0))


def test_rotation_matrix_properties() -> None:
    axis = np.array([1.0, 2.0, 3.0])
    rot = n2f.get_rotation_matrix(37.0, axis)
    axis = axis / np.linalg.norm(axis)

    assert rot.shape == (3, 3)
    assert np.allclose(rot @ rot.T, np.eye(3))
    assert np.isclose(np.linalg.det(rot), 1.0)
    # The rotation axis is left invariant.
    assert np.allclose(rot @ axis, axis)
    # A full turn is the identity; 90 degrees about z maps x to y.
    assert np.allclose(n2f.get_rotation_matrix(360.0, axis), np.eye(3))
    assert np.allclose(
        n2f.get_rotation_matrix(90.0, np.array([0.0, 0.0, 1.0]))
        @ np.array([1.0, 0.0, 0.0]),
        [0.0, 1.0, 0.0],
    )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def test_vector2matrix_is_column_major() -> None:
    vec = np.arange(6)
    assert np.array_equal(
        n2f.vector2matrix((2, 3), vec), np.array([[0, 2, 4], [1, 3, 5]])
    )
    assert np.array_equal(
        n2f.vector2matrix([3, 2], vec), np.array([[0, 3], [1, 4], [2, 5]])
    )


def test_error_metrics() -> None:
    ones = np.ones((2, 2))
    zeros = np.zeros((2, 2))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert np.isclose(n2f.get_l1_error(zeros, ones), 1.0)
        assert np.isclose(n2f.get_l2_error(zeros, ones), 1.0)
        assert np.isclose(n2f.get_max_error(zeros, ones), 1.0)
        assert np.isclose(n2f.get_l2_error(ones, ones), 0.0)

    # The L2 metric is the relative Frobenius norm.
    estimate = np.array([[1.0, 1.0], [1.0, 0.0]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert np.isclose(
            n2f.get_l2_error(estimate, ones),
            np.sqrt(1.0) / np.linalg.norm(ones),
        )


def test_free_space_electrical_params() -> None:
    z0, k0, lambda0 = n2f.utils.get_free_space_electrical_params(1e9)
    assert np.isclose(z0, 376.73031346199167)
    assert np.isclose(lambda0, 0.299792458)
    assert np.isclose(k0, 2 * np.pi / lambda0)
    assert np.isclose(lambda0 * 1e9, 0.299792458e9)

    z0_u, k0_u, lambda0_u = n2f.utils.get_free_space_electrical_params(
        123.0, unitary=True
    )
    assert np.isclose(z0_u, z0)
    assert np.isclose(lambda0_u, 1.0)
    assert np.isclose(k0_u, 2 * np.pi)


def test_spherical_sampling_helpers() -> None:
    theta, phi, matrix_size = n2f.utils.get_sph_smpl_angles(30, 45)
    assert matrix_size == (8, 6)  # (N_phi, N_theta)
    assert theta.shape == phi.shape == (48,)
    assert np.all((theta >= 0) & (theta <= np.pi))
    assert np.all((phi >= 0) & (phi < 2 * np.pi))

    theta_plot, phi_plot, _ = n2f.utils.get_sph_smpl_angles_for_plots(30, 45)
    assert theta_plot.shape == phi_plot.shape == (63,)
    assert np.isclose(theta_plot.max(), np.pi)
    assert np.isclose(phi_plot.max(), 2 * np.pi)

    assert np.allclose(n2f.utils.get_sph_smpl_res(2.0, 0.25), (7.5, 7.2))


def test_spanning_and_look_angles() -> None:
    angles, nbr_angles = n2f.utils.get_spanning_angles(40, 90)
    assert angles.shape == (33,)
    assert np.array_equal(nbr_angles, [1, 3, 5, 9, 17, 33])
    assert 0 <= angles.min() and angles.max() <= 90

    ifft_op, phi_ff, theta_ff = n2f.utils.get_look_angle_poly(
        np.array([0.0, 45.0]), 8, 8, 3
    )
    assert ifft_op.shape == (8, 3)
    assert phi_ff.shape == theta_ff.shape == (8, 2)


# ---------------------------------------------------------------------------
# Geometry builders
# ---------------------------------------------------------------------------


def test_build_array() -> None:
    positions = n2f.build_array(1.0, 3, 0.5, 3, 0.5)
    assert positions.shape == (3, 9)
    assert np.all(positions[2] == 0)
    assert np.isclose(np.max(np.abs(positions[0])), 0.5)
    assert np.isclose(np.max(np.abs(positions[1])), 0.5)
    # A 3x3 array centred on the origin has 9 distinct points.
    assert len({(x, y) for x, y in zip(positions[0], positions[1])}) == 9


def test_get_box_dim() -> None:
    array_pos = n2f.build_array(1.0, 3, 0.5, 3, 0.5)
    x_min, x_max, y_min, y_max, z_min, z_max, nx, ny, nz = n2f.get_box_dim(
        1.0, array_pos, 0.5, 0.25, 0.5
    )
    assert (x_min, x_max, y_min, y_max) == (-1.0, 1.0, -1.0, 1.0)
    assert (z_min, z_max) == (-0.5, 0.5)
    assert (nx, ny, nz) == (8, 8, 4)


def test_get_sph_radius() -> None:
    array_pos = n2f.build_array(1.0, 3, 0.5, 3, 0.5)
    assert np.isclose(n2f.get_sph_radius(1.0, array_pos, 0.25), np.sqrt(0.5) + 0.25)
    # With no extension the wavelength factors cancel.
    assert np.isclose(n2f.get_sph_radius(2.0, array_pos, 0.0), np.sqrt(0.5))


def test_build_box() -> None:
    array_pos = n2f.build_array(1.0, 3, 0.5, 3, 0.5)
    dims = n2f.get_box_dim(1.0, array_pos, 0.5, 0.25, 0.5)
    box_pos, box_n, dS, matrix_size = n2f.build_box([1] * 6, *dims, 1, 0, 0)

    # 2 * 8 * 8 + 2 * 8 * 4 + 2 * 8 * 4 = 256 sampling points.
    assert box_pos.shape == (3, 256)
    assert box_n.shape == (3, 256)
    assert dS.shape == (256,)
    assert np.all(dS > 0)
    assert np.allclose(np.linalg.norm(box_n, axis=0), 1.0)
    # Every normal is axis aligned.
    assert np.allclose(np.abs(box_n).sum(axis=0), 1.0)
    # The patches tile the box surface: 4 + 4 + 4 + 4 + 16 + 16.
    assert np.isclose(dS.sum(), 16.0)
    assert matrix_size.shape == (6, 2)


def test_get_box_vectors() -> None:
    array_pos = n2f.build_array(1.0, 3, 0.5, 3, 0.5)
    dims = n2f.get_box_dim(1.0, array_pos, 0.5, 0.25, 0.5)
    box_pos, box_n, _, _ = n2f.build_box([1] * 6, *dims, 1, 0, 0)

    r_mag, n_dot_r = n2f.get_box_vectors(array_pos, box_pos, box_n)
    assert r_mag.shape == (9, 256)
    assert n_dot_r.shape == (9, 256)
    assert np.all(r_mag > 0)
    assert np.all(n_dot_r >= -1 - 1e-12) and np.all(n_dot_r <= 1 + 1e-12)


def test_build_sphere() -> None:
    radius = 2.0
    sphere_pos, dS, theta, phi, matrix_size = n2f.build_sphere(
        radius, 0.25, 30, 30, 0
    )
    assert sphere_pos.shape[0] == 3
    assert sphere_pos.shape[1] == dS.size == theta.size == phi.size
    assert np.allclose(np.linalg.norm(sphere_pos, axis=0), radius)
    assert np.all(dS > 0)
    assert matrix_size == (12, 6)

    # flag=1 resolves the sampling from the wavelength resolution.
    sphere_res, dS_res, *_ = n2f.build_sphere(radius, 0.25, 0, 0, 1)
    assert sphere_res.shape[1] == 1200
    assert sphere_res.shape[1] != sphere_pos.shape[1]
    assert np.allclose(np.linalg.norm(sphere_res, axis=0), radius)
    assert dS_res.shape == (1200,)

    # Rotation preserves the radius and actually rotates the samples.
    rotated, *_ = n2f.build_sphere(
        radius, 0.25, 30, 30, 0, rot_angle=45, rot_axis=[0, 0, 1]
    )
    assert np.allclose(np.linalg.norm(rotated, axis=0), radius)
    assert not np.allclose(rotated, sphere_pos)
    # Rotation about z leaves the z coordinate untouched.
    assert np.allclose(rotated[2], sphere_pos[2])


def test_get_sph_vectors() -> None:
    array_pos = n2f.build_array(1.0, 3, 0.5, 3, 0.5)
    sphere_pos, _, _, _, _ = n2f.build_sphere(1.5, 0.25, 30, 30, 0)
    r_mag, n_dot_r, normals = n2f.get_sph_vectors(array_pos, sphere_pos)

    assert r_mag.shape == (9, sphere_pos.shape[1])
    assert n_dot_r.shape == r_mag.shape
    assert normals.shape == sphere_pos.shape
    assert np.allclose(np.linalg.norm(normals, axis=0), 1.0)
    # The normals point radially outward.
    assert np.allclose(normals, sphere_pos / np.linalg.norm(sphere_pos, axis=0))
    assert np.all(r_mag > 0)
    assert np.all(n_dot_r <= 1 + 1e-12)


def test_triangulated_sphere() -> None:
    points, triangles = n2f.get_tri_sph_mesh(0)
    assert points.shape == (12, 3)
    assert triangles.shape == (20, 3)
    assert np.allclose(np.linalg.norm(points, axis=1), 1.0)
    assert triangles.min() == 0 and triangles.max() == 11

    points_1, triangles_1 = n2f.get_tri_sph_mesh(1)
    assert points_1.shape == (72, 3)
    assert triangles_1.shape == (80, 3)
    assert np.allclose(np.linalg.norm(points_1, axis=1), 1.0)

    radius = 2.0
    centroids, dS, normals = n2f.build_tri_sphere(radius, 1)
    assert centroids.shape == (3, 80)
    assert normals.shape == (3, 80)
    assert dS.shape == (80,)
    assert np.allclose(np.linalg.norm(normals, axis=0), 1.0)
    # A coarse triangulated sphere approximates 4*pi*r^2 from below.
    ratio = dS.sum() / (4 * np.pi * radius**2)
    assert 0.85 < ratio < 1.0
    radii = np.linalg.norm(centroids, axis=0)
    assert np.all(radii > 0.85 * radius) and np.all(radii < radius)


def test_spiraling_trajectory() -> None:
    theta, phi, opp_theta, opp_phi = n2f.get_spiraling_helicoidal_trajectory(20, 10)
    assert theta.shape == phi.shape == (20,)
    assert opp_theta.shape == opp_phi.shape == (10,)
    assert np.isclose(theta[0], 0.0) and np.isclose(theta[-1], 90.0)
    assert np.all((phi >= 0) & (phi < 360))
    assert np.isclose(opp_theta[-1], 90.0)
    assert np.all((opp_phi >= 0) & (opp_phi < 360))


# ---------------------------------------------------------------------------
# Scalar near-field to far-field
# ---------------------------------------------------------------------------


def test_scalar_excitations_are_unit_power() -> None:
    array_pos = n2f.build_array(1.0, 3, 0.5, 3, 0.5)
    excitation = n2f.sf_excitations(1.0, array_pos, 0, 0)

    assert excitation.shape == (array_pos.shape[1],)
    assert np.allclose(np.abs(excitation), 4 * np.pi / np.sqrt(array_pos.shape[1]))
    # The phasors are normalized so that the total emitted power is unitary.
    assert np.isclose(np.sum(np.abs(excitation) ** 2), 16 * np.pi**2)


def test_scalar_near_field_is_finite(box_setup: dict) -> None:
    psi = box_setup["psi"]
    del_psi = box_setup["del_psi"]
    n_points = box_setup["box_pos"].shape[1]
    assert psi.shape == del_psi.shape == (n_points,)
    assert np.isfinite(psi).all() and np.isfinite(del_psi).all()
    assert np.any(psi != 0)


def test_scalar_box_transformation(box_setup: dict) -> None:
    theta = n2f.deg2rad(np.linspace(0, 90, 37))
    phi = n2f.deg2rad([0, 45, 90])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        far_field = n2f.sf_nf2ff_solver(
            WAVELENGTH,
            theta,
            phi,
            box_setup["box_pos"],
            box_setup["box_n"],
            box_setup["dS"],
            box_setup["psi"],
            box_setup["del_psi"],
        )
        reference = n2f.sf_direct_ff_solver(
            WAVELENGTH,
            theta,
            phi,
            box_setup["excitation"],
            box_setup["array_pos"],
        )

    assert far_field.shape == (len(phi), len(theta))
    assert reference.shape == far_field.shape
    assert np.isfinite(far_field).all()
    # Huygens' surface integral reproduces the direct point-source pattern.
    rel_error = np.linalg.norm(np.abs(far_field) - np.abs(reference)) / np.linalg.norm(
        np.abs(reference)
    )
    assert rel_error < 0.05


def test_scalar_operator_matches_solver(box_setup: dict) -> None:
    theta = n2f.deg2rad(np.linspace(0, 90, 19))
    phi = n2f.deg2rad([0, 45, 90])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        far_field = n2f.sf_nf2ff_solver(
            WAVELENGTH,
            theta,
            phi,
            box_setup["box_pos"],
            box_setup["box_n"],
            box_setup["dS"],
            box_setup["psi"],
            box_setup["del_psi"],
        )

    lpsi, ldel_psi = n2f.sf_nf2ff_operator(
        WAVELENGTH,
        theta,
        phi,
        box_setup["box_pos"],
        box_setup["box_n"],
        box_setup["dS"],
    )
    n_points = box_setup["box_pos"].shape[1]
    assert lpsi.shape == (len(theta), n_points, len(phi))
    assert ldel_psi.shape == lpsi.shape

    for i in range(len(phi)):
        applied = lpsi[:, :, i] @ box_setup["psi"] + ldel_psi[:, :, i] @ box_setup[
            "del_psi"
        ]
        # sf_nf2ff_solver stores its (nearly real) result in a real array, so
        # only the real part survives there.
        assert np.allclose(applied.real, far_field[i], atol=1e-10)
        assert np.linalg.norm(applied.imag) < 0.01 * np.linalg.norm(applied.real)


def test_scalar_gain(box_setup: dict) -> None:
    theta = n2f.deg2rad(np.linspace(0, 90, 19))
    phi = n2f.deg2rad([0, 90])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        far_field = n2f.sf_nf2ff_solver(
            WAVELENGTH,
            theta,
            phi,
            box_setup["box_pos"],
            box_setup["box_n"],
            box_setup["dS"],
            box_setup["psi"],
            box_setup["del_psi"],
        )
    gain = n2f.sf_compute_gain(far_field)
    assert gain.shape == far_field.shape
    assert np.allclose(gain, np.abs(far_field) ** 2)
    assert np.all(gain >= 0)


# ---------------------------------------------------------------------------
# Vector far-field reference solver
# ---------------------------------------------------------------------------


def test_vf_excitations() -> None:
    array_pos = n2f.build_array(1.0, 3, 0.5, 3, 0.5)
    z0, k0, _ = n2f.utils.get_free_space_electrical_params(1e9)

    J, M = n2f.vf_excitations(k0, array_pos, 1.0, 0, 0, 0.0, 0, 0, 0, 0)
    assert J.shape == array_pos.shape
    assert M.shape == array_pos.shape
    # A theta-oriented electric current with theta=0 points along +z.
    assert np.allclose(J[0], 0) and np.allclose(J[1], 0)
    assert np.allclose(J[2], 1.0)
    assert np.allclose(M, 0)

    _, M = n2f.vf_excitations(k0, array_pos, 0.0, 0, 0, 1.0, 90, 0, 0, 0)
    # A theta-oriented magnetic current at theta=90 points along +x.
    assert np.allclose(M[0], 1.0, atol=1e-12)
    assert np.allclose(M[1], 0, atol=1e-12)


def test_vf_direct_far_field_and_gain() -> None:
    array_pos = n2f.build_array(1.0, 3, 0.5, 3, 0.5)
    z0, k0, _ = n2f.utils.get_free_space_electrical_params(1e9)
    J, M = n2f.vf_excitations(k0, array_pos, 1.0, 0, 0, 0.0, 0, 0, 0, 0)

    theta = n2f.deg2rad(np.linspace(0, 90, 19))
    phi = n2f.deg2rad([0, 45, 90])
    theta_ff, phi_ff = np.meshgrid(theta, phi)

    et_ff, ep_ff = n2f.vf_direct_ff_solver(
        k0, z0, array_pos, J, M, theta_ff, phi_ff
    )
    assert et_ff.shape == ep_ff.shape == theta_ff.shape
    assert np.iscomplexobj(et_ff)
    assert np.isfinite(et_ff).all() and np.isfinite(ep_ff).all()
    assert np.any(np.abs(et_ff) > 0)

    gain_t, gain_p = n2f.vf_compute_gain(z0, et_ff, ep_ff, 1.0)
    assert np.allclose(gain_t, 2 * np.pi * np.abs(et_ff) ** 2 / z0)
    assert np.allclose(gain_p, 2 * np.pi * np.abs(ep_ff) ** 2 / z0)
    assert np.all(gain_t >= 0) and np.all(gain_p >= 0)


def test_vf_near_field_solver() -> None:
    array_pos = n2f.build_array(1.0, 2, 0.25, 2, 0.25)
    z0, k0, _ = n2f.utils.get_free_space_electrical_params(1e9)
    dims = n2f.get_box_dim(1.0, array_pos, 0.25, 0.25, 0.25)
    box_pos, box_n, dS, _ = n2f.build_box([1] * 6, *dims, 1, 0, 0)
    J, M = n2f.vf_excitations(k0, array_pos, 1.0, 0, 0, 0.0, 0, 0, 0, 0)
    n2f.vf_nf_solver(k0, z0, array_pos, box_pos, box_n, dS, J, M)


def test_vf_n2f_operator() -> None:
    array_pos = n2f.build_array(1.0, 3, 0.5, 3, 0.5)
    z0, k0, _ = n2f.utils.get_free_space_electrical_params(1e9)
    dims = n2f.get_box_dim(1.0, array_pos, 0.25, 0.25, 0.25)
    box_pos, box_n, dS, _ = n2f.build_box([1] * 6, *dims, 1, 0, 0)
    theta = n2f.deg2rad(np.linspace(0, 90, 19))
    phi = n2f.deg2rad([0, 45, 90])
    theta_ff, phi_ff = np.meshgrid(theta, phi)
    n2f.vf_n2f_op_fields(k0, z0, box_pos, box_n, dS, theta_ff, phi_ff)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def test_plotting_helpers() -> None:
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt

    color = n2f.get_color_map()
    assert color.shape == (3,)
    assert np.all((color >= 0) & (color <= 1))

    props = n2f.get_figure_properties()
    assert set(props) == {"fs", "lw", "ms"}

    fig = n2f.plot_svd_error(np.array([1e-1, 1e-2, 1e-3]))
    try:
        assert fig is not None
    finally:
        plt.close(fig)
