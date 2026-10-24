import numpy as np

#: Values from ``cpp/include/constants.h`` (the frozen v0.1 C++ sources), verbatim.
#:
#: The legacy MATLAB-derived :func:`get_constants` uses ``z0 = 120*pi``
#: (376.99111843077515), while the v0.1 sources use 376.730313461 -- a 7e-4
#: relative difference that is far larger than the tolerance gates.  Port
#: TFE and any other quantity built from ``z0`` must use these values.
C0 = 299792458.0
EPS0 = 8.854187817e-12
MU0 = 1.2566370614359e-6
PI = 3.14159265358979323846
Z0 = 376.730313461


def v01_constants():
    """Return the exact constants used by the v0.1 C++ sources."""
    return {
        "c0": C0,
        "z0": Z0,
        "eps0": EPS0,
        "mu0": MU0,
        "pi": PI,
    }


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
