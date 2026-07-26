# Simulation project examples

from .electrostatics import run_electrostatics
from .waveguide import run_waveguide
from .modal_analysis import modal_analysis_rectangular, modal_analysis_open_strip
from .thermal import thermal_distribution, thermal_distribution_dg
from .filter_design import (
    bilateral_filter, two_post_filter,
    bilateral_filter_hb, two_post_filter_hb
)
from .capacitive import coaxial_capacitance, capacitive_clearance
from .circulator import circulator, circulator_imp, circulator_ddschur
from .scattering import scattering_dd, scattering_dd_iterative, scattering_full_field
from .filter_dnngp import (
    bilateral_filter_dnngp, BilateralFilterDNNGP, plot_surrogate,
    adaptive_lhs_bilateral_filter,
)
