"""N2F - Near-field to far-field transformations with model order reduction.

Near-field to far-field transformations with model order reduction, bundled
under the MIT license; see the repository-root ``LICENSE``.
"""

# Import submodules
from . import geometry
from . import transforms
from . import scalar
from . import vector
from . import utils

# Import plotting only if matplotlib is available
try:
    from . import plotting
except ImportError:
    plotting = None

# Main exports for quick access
from .geometry import (
    build_array, build_box, build_sphere, build_tri_sphere,
    get_box_dim, get_sph_radius, get_box_vectors, get_sph_vectors,
    get_tri_sph_mesh, get_spiraling_helicoidal_trajectory
)

from .transforms import (
    cartesian2spherical, spherical2cartesian, cross_operator, get_rotation_matrix
)

from .scalar import (
    sf_excitations, sf_compute_gain, sf_nf_solver, sf_nf2ff_solver,
    sf_direct_ff_solver, sf_nf2ff_operator
)

from .vector import (
    vf_excitations, vf_compute_gain, vf_nf_solver, vf_nf2ff_solver,
    vf_direct_ff_solver, vf_n2f_op_fields, vf_n2f_op_fields_fft
)

from .utils import (
    deg2rad, rad2deg, vector2matrix, get_l1_error, get_l2_error, get_max_error
)

# Plotting imports - only if matplotlib is available
if plotting is not None:
    from .plotting import (
        plot_sph_geom, sf_plot_array_geom, sf_plot_box_nf, sf_plot_sph_nf,
        sf_plot_sph_nf_solid, sf_plot_ff_cut_planes,
        vf_plot_array_geom_3d, vf_plot_ff_3d, vf_plot_ff_3d_radiation,
        vf_plot_ff_cut_planes, vf_plot_ff_polar_cut_planes,
        vf_plot_surface_power_density, plot_svd_error,
        get_color_map, get_figure_properties
    )
else:
    # Dummy exports if plotting is not available
    plot_sph_geom = None
    sf_plot_array_geom = None
    sf_plot_box_nf = None
    sf_plot_sph_nf = None
    sf_plot_sph_nf_solid = None
    sf_plot_ff_cut_planes = None
    vf_plot_array_geom_3d = None
    vf_plot_ff_3d = None
    vf_plot_ff_3d_radiation = None
    vf_plot_ff_cut_planes = None
    vf_plot_ff_polar_cut_planes = None
    vf_plot_surface_power_density = None
    plot_svd_error = None
    get_color_map = None
    get_figure_properties = None

__all__ = [
    # Geometry
    'build_array', 'build_box', 'build_sphere', 'build_tri_sphere',
    'get_box_dim', 'get_sph_radius', 'get_box_vectors', 'get_sph_vectors',
    'get_tri_sph_mesh', 'get_spiraling_helicoidal_trajectory',

    # Transforms
    'cartesian2spherical', 'spherical2cartesian', 'cross_operator', 'get_rotation_matrix',

    # Scalar field
    'sf_excitations', 'sf_compute_gain', 'sf_nf_solver', 'sf_nf2ff_solver',
    'sf_direct_ff_solver', 'sf_nf2ff_operator',

    # Vector field
    'vf_excitations', 'vf_compute_gain', 'vf_nf_solver', 'vf_nf2ff_solver',
    'vf_direct_ff_solver', 'vf_n2f_op_fields', 'vf_n2f_op_fields_fft',

    # Utilities
    'deg2rad', 'rad2deg', 'vector2matrix', 'get_l1_error', 'get_l2_error', 'get_max_error',
    'get_color_map', 'get_figure_properties',

    # Plotting
    'plot_sph_geom', 'sf_plot_array_geom', 'sf_plot_box_nf', 'sf_plot_sph_nf',
    'sf_plot_sph_nf_solid', 'sf_plot_ff_cut_planes', 'vf_plot_array_geom',
    'vf_plot_array_geom_3d', 'vf_plot_ff_3d', 'vf_plot_ff_3d_radiation',
    'vf_plot_ff_cut_planes', 'vf_plot_ff_polar_cut_planes',
    'vf_plot_surface_power_density', 'plot_svd_error',

    # Submodules
    'geometry', 'transforms', 'scalar', 'vector', 'utils', 'plotting'
]

__version__ = "1.0.0"
__author__ = "Laurent Ntibarikure"
