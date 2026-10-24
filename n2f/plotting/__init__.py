"""Plotting functions"""

from .plot_properties import get_color_map, get_figure_properties
from .geometry_plots import (
    plot_sph_geom,
    plot_selected_angles,
    sf_plot_array_geom,
    print_pdf,
    print_eps
)
from .field_plots import (
    sf_plot_box_nf,
    sf_plot_sph_nf,
    sf_plot_sph_nf_solid,
    sf_plot_ff_cut_planes,
    vf_plot_array_geom_3d,
    vf_plot_ff_3d,
    vf_plot_ff_3d_radiation,
    vf_plot_ff_cut_planes,
    vf_plot_ff_polar_cut_planes,
    vf_plot_surface_power_density,
    plot_svd_error
)
from .error_plots import plot_svd_error as plot_svd_error_detailed

__all__ = [
    'plot_sph_geom',
    'sf_plot_ff_cut_planes',
    'plot_selected_angles',
    'plot_svd_error',
    'plot_svd_error_detailed',
    'sf_plot_array_geom',
    'sf_plot_box_nf',
    'sf_plot_sph_nf',
    'sf_plot_sph_nf_solid',
    'vf_plot_array_geom',
    'vf_plot_array_geom_3d',
    'vf_plot_ff_3d',
    'vf_plot_ff_3d_radiation',
    'vf_plot_ff_cut_planes',
    'vf_plot_ff_polar_cut_planes',
    'vf_plot_surface_power_density',
    'get_color_map',
    'get_figure_properties',
    'print_eps',
    'print_pdf',
]
