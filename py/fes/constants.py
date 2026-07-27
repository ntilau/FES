import numpy as np


def get_constants():
    """Physical constants and utility functions (port of GetConstants.m)."""
    c0 = 299792458.0
    z0 = 120.0 * np.pi
    eps0 = 1.0 / (z0 * c0)
    mu0 = z0 / c0

    return {
        "c0": c0,
        "z0": z0,
        "eps0": eps0,
        "mu0": mu0,
        "db": lambda x: 20.0 * np.log10(np.abs(x)),
        "arg": lambda x: np.unwrap(np.angle(x)) * 180.0 / np.pi,
    }
