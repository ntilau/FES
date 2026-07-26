"""Waveguide modal analysis (cutoff frequencies, dispersion).

Port of ProjectModalAnalysis.m and ProjectModalAnalysisOpenStrip.m.
"""

import numpy as np
from scipy.sparse.linalg import eigs

from ..constants import get_constants
from ..mesh.build import build_regular_square
from ..mesh.io_poly import read_poly
from ..fem.assembly import assemble_linear
from ..post.plot import plot_field


def modal_analysis_rectangular(a=22.86e-3, b=10.16e-3, p_ord=3,
                               n_modes=4, epsr=1.0, plot=False):
    """Compute TE mode cutoff frequencies for a rectangular waveguide.

    Parameters
    ----------
    a, b : float
        Waveguide width and height (m).
    p_ord : int
        Polynomial order (1-4).
    n_modes : int
        Number of modes to compute.
    epsr : float
        Relative permittivity.
    plot : bool
        Whether to display the field.

    Returns
    -------
    fc : ndarray
        Cutoff frequencies (Hz).
    """
    consts = get_constants()
    c0 = consts["c0"]

    mesh = build_regular_square(5, 3)
    mesh["node"] = mesh["node"] * np.array([a, b])
    mesh["BC"] = {"Dir": [1]}

    sys = {"pOrd": p_ord, "Hcurl": True}
    sys, mesh = assemble_linear(sys, mesh)

    # Solve TE eigenvalue problem: S u = k0² T u
    # With Dirichlet on TE: use full T and full S
    Tte = sys["Tt"]
    Ste = sys["St"]

    # Remove Dirichlet DOFs (vector version)
    dir_dofs = sys.get("Dir_0", np.array([], dtype=int))

    # For vector problem: Dirichlet removes boundary DOFs
    # slab is already 0-based; slab_zero gives interior edge indices directly
    slab_zero = np.where(mesh["slab"] == 0)[0]
    if sys["pOrd"] == 1:
        vec_free = slab_zero
    elif sys["pOrd"] == 2:
        vec_free = np.concatenate([
            slab_zero,
            mesh["NSPIG"] + slab_zero,
            np.arange(2 * mesh["NSPIG"], 2 * mesh["NSPIG"] + 2 * mesh["NELE"])
        ])
    elif sys["pOrd"] == 3:
        vec_free = np.concatenate([
            slab_zero,
            mesh["NSPIG"] + slab_zero,
            np.arange(2 * mesh["NSPIG"], 2 * mesh["NSPIG"] + 2 * mesh["NELE"]),
            2 * mesh["NSPIG"] + 2 * mesh["NELE"] + slab_zero,
            np.arange(3 * mesh["NSPIG"] + 2 * mesh["NELE"],
                      3 * mesh["NSPIG"] + 6 * mesh["NELE"])
        ])
    else:
        vec_free = np.arange(Ste.shape[0])

    vec_free = vec_free[(vec_free >= 0) & (vec_free < Ste.shape[0])]
    Tte_sub = Tte[vec_free, :][:, vec_free]
    Ste_sub = Ste[vec_free, :][:, vec_free]

    evals = eigs(Ste_sub, M=Tte_sub, k=min(n_modes * 2, len(vec_free) - 2),
                 sigma=(2 * np.pi * 5e9 / c0)**2, which="LM")
    e = evals[0]
    e = e[np.abs(e) > 1e-5]
    fc = np.sqrt(e[:n_modes].real) * c0 / (2 * np.pi)

    # Reference: TE10
    fc_ref = c0 / (2 * a)
    print(f"Computed TE10 cutoff: {fc[0]/1e9:.4f} GHz  (ref: {fc_ref/1e9:.4f} GHz)")
    print(f"  relative error: {abs(fc[0] - fc_ref) / fc_ref:.4e}")

    sys["u"] = np.zeros(sys["NDOFs"])
    if plot:
        plot_field(sys, mesh, cmap="jet")

    return fc


def modal_analysis_open_strip():
    """Modal analysis of a ridged waveguide (open strip).

    Port of ProjectModalAnalysisOpenStrip.m.
    """
    sys = {"pOrd": 2, "hOrd": 1}
    mesh = read_poly("ModelWR90Strip", data_dir="./data")
    mesh["BC"] = {"Dir": [1]}

    sys["Hcurl"] = True
    sys, mesh = assemble_linear(sys, mesh)

    slab_zero = np.where(mesh["slab"] == 0)[0]
    if sys["pOrd"] == 1:
        vec_free = slab_zero
    elif sys["pOrd"] == 2:
        vec_free = np.concatenate([
            slab_zero,
            mesh["NSPIG"] + slab_zero,
            np.arange(2 * mesh["NSPIG"], 2 * mesh["NSPIG"] + 2 * mesh["NELE"])
        ])
    elif sys["pOrd"] == 3:
        vec_free = np.concatenate([
            slab_zero,
            mesh["NSPIG"] + slab_zero,
            np.arange(2 * mesh["NSPIG"], 2 * mesh["NSPIG"] + 2 * mesh["NELE"]),
            2 * mesh["NSPIG"] + 2 * mesh["NELE"] + slab_zero,
            np.arange(3 * mesh["NSPIG"] + 2 * mesh["NELE"],
                      3 * mesh["NSPIG"] + 6 * mesh["NELE"])
        ])
    else:
        vec_free = np.arange(sys["St"].shape[0])

    vec_free = vec_free[(vec_free >= 0) & (vec_free < sys["St"].shape[0])]
    Tte_sub = sys["Tt"][vec_free, :][:, vec_free]
    Ste_sub = sys["St"][vec_free, :][:, vec_free]

    evals = eigs(Ste_sub, M=Tte_sub, k=2)
    fc = np.sqrt(np.abs(evals[0])) * sys["c0"] / (2 * np.pi)
    print(f"Cutoff frequencies: {fc / 1e9} GHz")
    return fc
