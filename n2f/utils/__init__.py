"""Utility functions"""

from .deg2rad import deg2rad
from .rad2deg import rad2deg
from .vector2matrix import vector2matrix
from .angular_functions import get_spanning_angles, get_look_angle_poly
from .spherical_sampling import (
    get_sph_smpl_angles,
    get_sph_smpl_angles_for_plots,
    get_sph_smpl_res
)
from .error_metrics import get_l1_error, get_l2_error, get_max_error
from .electrical_params import get_free_space_electrical_params, get_reduced_output_system_matrices

__all__ = [
    'deg2rad',
    'rad2deg',
    'vector2matrix',
    'get_spanning_angles',
    'get_look_angle_poly',
    'get_sph_smpl_angles',
    'get_sph_smpl_angles_for_plots',
    'get_sph_smpl_res',
    'get_l1_error',
    'get_l2_error',
    'get_max_error',
    'get_free_space_electrical_params',
    'get_reduced_output_system_matrices',
]
