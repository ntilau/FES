"""Plotting utility functions."""

import numpy as np


def get_color_map():
    """
    Returns a color map array for plotting.

    Returns
    -------
    color : ndarray (3,)
        RGB color components [R, G, B] normalized to [0, 1]
    """
    return np.array([0.1216, 0.2863, 0.4902])


def get_figure_properties():
    """
    Returns figure property dictionary with standard plot formatting.

    Returns
    -------
    fig_props : dict
        Dictionary with the following keys:
        - 'fs': font size (default 12)
        - 'lw': line width (default 1.5)
        - 'ms': marker size (default 7)
    """
    return {
        'fs': 12,      # font size
        'lw': 1.5,     # line width
        'ms': 7        # marker size
    }
