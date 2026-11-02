"""Tests for the 3D H(curl) assembler (:mod:`fes.core.assemblers.em_e_fd`).

These pin the building blocks directly: the global-numbering rule used when
collecting the upper triangle, and the port eigenpair selection.
"""

from __future__ import annotations

import numpy as np

from fes.core.assemblers.em_e_fd import (
    _port_eigenpairs,
    _upper_triplets,
)
from fes.core.assemblers.port_pencil import reference_mode_beta


# ── building blocks ─────────────────────────────────────────────────────────


def test_upper_triplets_select_on_global_dof_numbers() -> None:
    """The reference tests ``cdof.v(i) <= cdof.v(j)``, not the local indices."""
    v = np.array([5, 1])  # descending: local and global orders disagree
    block = np.array([[0.0, 1.0], [2.0, 3.0]])

    rows, cols, values = _upper_triplets(v, block)

    # v[0]=5 <= v[0], v[0] > v[1], v[1] <= v[0], v[1] <= v[1]: the stored
    # keys are (5,5), (1,5), (1,1) -- in particular block[1,0], not block[0,1].
    np.testing.assert_array_equal(rows, [5, 1, 1])
    np.testing.assert_array_equal(cols, [5, 5, 1])
    np.testing.assert_array_equal(values, [0.0, 2.0, 3.0])


def test_port_eigenpairs_skip_the_scalar_static_mode() -> None:
    """The scalar sub-block carries a ``lambda = 0`` mode with no vector part.

    With the shift at ``-1`` it is nearer than the real mode (``5``), so a
    selection that ignored the H(curl) block would return it and make the
    ``1/sqrt(beta)`` normalisation singular.
    """
    tmp_a = np.array([[5.0, 0.0], [0.0, 0.0]])
    tmp_b = np.array([[1.0, 0.0], [0.0, 1.0]])

    values, vectors = _port_eigenpairs(tmp_a, tmp_b, numt=1, shift=-1.0, nmodes=1)

    np.testing.assert_allclose(values, [5.0], atol=1e-12)
    assert abs(vectors[0, 0]) > 0.5


# ── the reference's ModeBeta convention ─────────────────────────────────────


def test_reference_mode_beta_matches_the_cpp_branch() -> None:
    """``Eigen.cpp`` stores ``ModeBeta = sqrt(kc^2 - k0^2)``, not beta.

    A propagating mode has a negative pencil eigenvalue, so the reference's
    ``ModeBeta`` is ``+1j*beta``; an evanescent mode's eigenvalue is positive
    and ``ModeBeta`` is the real attenuation constant.  The TFE reduction
    divides ``i*k0*Z0`` by this quantity, so using the physical propagation
    constant instead rotates every port-mode column and destroys the
    S-parameters.
    """
    lam = np.array([
        -24975.0 + 1e-24j,  # propagating: ModeBeta = +1j*158.03...
        30799.0 - 1e-24j,   # evanescent: ModeBeta = 175.49...
        -24975.0 - 1e-24j,  # propagating with noise on the other side
    ])
    expected = np.array([
        1j * np.sqrt(24975.0),
        np.sqrt(30799.0),
        1j * np.sqrt(24975.0),
    ])
    np.testing.assert_allclose(reference_mode_beta(lam), expected, atol=1e-9)

