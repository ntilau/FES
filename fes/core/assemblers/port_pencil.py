"""Shared reduced-pencil eigensolver used by the waveport and cross-section paths.

Both ``em_e_fd``'s wave-port mode computation and ``em_e_tl_eig``'s 2D
cross-section eigenproblem solve the same reduced generalized pencil

.. math::

    A_{tt} v = \\lambda B_{schur} v, \\qquad
    B_{schur} = T_{t2} - G (S_z - k_0^2 T_z)^{-1} G^H

(the Zhu-Cangellaris Schur complement of the scalar block), select the
physical modes ``k_c^2 = k_0^2 + \\lambda > 0``, order them TE10-first by
ascending cutoff wavenumber, and B-orthonormalize the mode vectors.  Keeping
the code here gives the two formulations one source of truth.

The implementation mirrors the reference C++ ``compute_waveport_modes``:
a shift-invert ARPACK solve with a dense fallback, then the ``AssLin.cpp``
normalization ``ModeVec = V @ inv(sqrt(V^T B V))``.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg
from scipy.linalg import solve, sqrtm, inv as _inv
import scipy.sparse as sp
import scipy.sparse.linalg

from ...constants import Z0

__all__ = [
    "_DBL_MIN",
    "_GAMMA_TOL",
    "_port_reduced_pencil",
    "_port_eigenpairs_dense",
    "_port_eigenpairs_arpack",
    "solve_reduced_pencil",
    "postprocess_modes",
]

#: ``DBL_MIN``, the reference's starting value for ``maxepsr``/``maxmur``.
_DBL_MIN = float.fromhex("0x0.0000000000001p-1022")

#: ``gamtol`` from ``eigen.cpp``: components below this become exactly zero.
_GAMMA_TOL = 1e-9


def _port_reduced_pencil(tmpA: np.ndarray, tmpB: np.ndarray, numt: int,
                         numz: int) -> tuple:
    """Reduce the port pencil to edge-only DOFs via the Schur complement.

    Eliminates the scalar face DOFs with the Zhu-Cangellaris Schur complement

        B_schur = Tt2 - G @ inv(Sz - kk*Tz) @ G.T

    and returns the reduced edge pencil ``(A_tt, B_schur)`` where
    ``A_tt = St - kk*Tt``.  Works for ``numz == 0`` too, in which case
    ``B_schur`` is just ``Tt2``.
    """
    Tt2 = tmpB[:numt, :numt]
    G = tmpB[:numt, numt:]
    A_tt = tmpA[:numt, :numt]
    if numz > 0:
        D = tmpB[numt:, numt:]
        B_schur = Tt2 - G @ solve(D, G.conj().T)
    else:
        B_schur = Tt2
    return A_tt, B_schur


def _port_eigenpairs_dense(A_tt: np.ndarray, B_schur: np.ndarray) -> tuple:
    """Dense generalized eigensolve of the reduced edge pencil ``A_tt v = l B_schur v``.

    Returns every pencil eigenvalue; the caller selects the physical port modes
    (``kc^2 > 0``) and orders them TE10-first.  Used by default because it is
    deterministic and bit-comparable with the reference C++ binary.
    """
    lam, vectors = scipy.linalg.eig(A_tt, B_schur)
    return np.asarray(lam, dtype=complex), np.asarray(vectors, dtype=complex)


def _port_eigenpairs_arpack(A_tt, B_schur, shift: float, nmodes: int,
                            eig_tol: float = 0.0):
    """Shift-invert ARPACK for the port pencil ``A_tt v = l B_schur v``.

    Factorises ``A_tt/shift - B_schur`` once with SuperLU and asks ARPACK for
    the Ritz values of ``op = (A_tt/shift - B_schur)^{-1} (B_schur/shift)``
    with largest magnitude, i.e. the pencil eigenvalues nearest *shift* -- the
    lowest-cutoff waveguide modes, since ``shift = -k0^2 * maxepsr * maxmur``
    sits at the bottom of the spectrum.  Each Ritz value ``theta`` is converted
    to ``l = shift + 1/theta``, mirroring ``zneupd``.

    Returns ``(lam, vectors)`` with edge-only ``vectors`` of shape
    ``(numt, k)``, or ``None`` when ARPACK does not converge (the caller falls
    back to the dense solve).
    """
    n = A_tt.shape[0]
    if n < 2:
        return None
    A_tt = A_tt.tocsc() if sp.issparse(A_tt) else sp.csc_matrix(A_tt)
    B_schur = B_schur.tocsc() if sp.issparse(B_schur) else sp.csc_matrix(B_schur)
    A_s = (A_tt / shift - B_schur).tocsc()
    try:
        lu = scipy.sparse.linalg.splu(A_s)
    except RuntimeError:
        return None

    def matvec(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=complex).reshape(n)
        y = np.asarray(B_schur @ x, dtype=complex).reshape(n) / shift
        return lu.solve(y)

    operator = scipy.sparse.linalg.LinearOperator(
        (n, n), matvec=matvec, dtype=complex
    )
    # Deterministic start vector (mirrors the hand-written Arnoldi path), so the
    # Ritz values are reproducible across runs and builds.
    rng = np.random.default_rng(0)
    v0 = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    v0 = v0 / np.linalg.norm(v0)
    k = min(max(nmodes * 6, 16), n - 1)
    if k < 1:
        return None
    ncv = min(2 * k + 1, n - 1)
    if ncv <= k:
        return None
    try:
        theta, vectors = scipy.sparse.linalg.eigs(
            operator,
            k=k,
            which="LM",
            ncv=ncv,
            tol=eig_tol,
            v0=v0,
            return_eigenvectors=True,
        )
    except scipy.sparse.linalg.ArpackNoConvergence:
        return None
    if vectors is None:
        return None
    theta = np.asarray(theta, dtype=complex)
    lam = shift + 1.0 / theta
    vectors = np.asarray(vectors, dtype=complex)
    return lam, vectors


def solve_reduced_pencil(
    A_tt,
    B_schur,
    kk: float,
    shift: float,
    nmodes: int,
    eig_tol: float = 0.0,
    solver: str = "auto",
) -> tuple:
    """Solve ``A_tt v = lam B_schur v`` with the reference's selection rule.

    ``"auto"``/``"arpack"`` try shift-invert ARPACK first and accept the result
    only when it yields at least *nmodes* physical modes (``Re(kk + lam) > 0``);
    otherwise (and for ``"dense"``) the dense generalized solve is used.

    Returns ``(lam, vectors, label)`` where *label* is ``"arpack"`` or
    ``"dense"``.
    """
    lam = vectors = None
    label = "dense"
    if solver in ("arpack", "auto"):
        result = _port_eigenpairs_arpack(A_tt, B_schur, shift, nmodes, eig_tol)
        if result is not None:
            cand_lam, cand_vec = result
            cand_phys = np.count_nonzero(np.real(kk + cand_lam) > 0.0)
            if int(cand_phys) >= nmodes:
                lam, vectors = cand_lam, cand_vec
                label = "arpack"
    # Dense fallback when ARPACK does not converge or yields too few physical
    # modes (the shift sits at the bottom of the spectrum, so spurious
    # negative-kc^2 modes can crowd out the physical ones).
    if lam is None:
        lam, vectors = _port_eigenpairs_dense(A_tt, B_schur)
        label = "dense"
    return (
        np.asarray(lam, dtype=complex),
        np.asarray(vectors, dtype=complex),
        label,
    )


def _reorder_by_previous(beta: np.ndarray, prev_beta) -> np.ndarray:
    """Permutation matching each previous mode to the nearest current ``|beta|``."""
    prev_abs = np.abs(np.asarray(prev_beta, dtype=complex))
    curr_abs = np.abs(beta)
    used = np.zeros(beta.size, dtype=bool)
    perm = np.zeros(beta.size, dtype=np.int64)
    for i in range(beta.size):
        diffs = np.where(used, np.inf, np.abs(curr_abs - prev_abs[i]))
        j = int(np.argmin(diffs))
        perm[i] = j
        used[j] = True
    return perm


def postprocess_modes(
    kk: float,
    k0: float,
    lam,
    vectors,
    apply_B,
    nmodes: int,
    prev_beta=None,
) -> tuple:
    """Select, order, transform and B-orthonormalize the reduced-pencil modes.

    *apply_B* computes ``B_schur @ X`` (a callable so callers can avoid keeping
    a second reference to a large matrix).

    Returns ``(beta, mode_vec, mode_vecf)``.  ``beta = |Re| + 1j*|Im|`` follows
    the waveport convention: the real part is the propagation constant and the
    imaginary part the attenuation.  When *prev_beta* is given, the selected
    modes are reordered to track the previous frequency by nearest
    ``|mode_beta|`` instead of ascending cutoff.
    """
    lam = np.asarray(lam, dtype=complex)
    vectors = np.asarray(vectors, dtype=complex)

    # Cutoff wavenumber squared: kc^2 = kk + lambda = k0^2 + (kc^2 - k0^2).
    # Keep only physical waveguide modes (kc^2 > 0); the indefinite pencil
    # (St - k0^2 Tt, Tt2) also yields spurious negative-kc^2 eigenvalues.
    kc2 = kk + lam
    physical = np.real(kc2) > 0.0
    lam = lam[physical]
    vectors = vectors[:, physical]
    kc2 = kc2[physical]

    # Order TE10-first: ascending cutoff wavenumber, then take nmodes.
    take = min(int(nmodes), kc2.size)
    order_idx = np.argsort(np.real(kc2), kind="stable")[:take]
    lam = lam[order_idx]
    vectors = vectors[:, order_idx]
    kc2 = kc2[order_idx]

    # Propagation constant: beta^2 = k0^2 - kc^2 = -lambda.
    beta = np.sqrt(kk - kc2)
    real = np.abs(beta.real)
    imag = np.abs(beta.imag)
    real[real < _GAMMA_TOL] = 0.0
    imag[imag < _GAMMA_TOL] = 0.0
    beta = real + 1j * imag

    # Optional sweep tracking: preserve the previous frequency's mode order.
    if prev_beta is not None and np.size(prev_beta) == beta.size and beta.size:
        perm = _reorder_by_previous(beta, prev_beta)
        beta = beta[perm]
        vectors = vectors[:, perm]

    # B-orthonormalisation matching C++ (AssLin.cpp line 342-343):
    #   ModeVec = V[:numt,:] @ inv(sqrt(V.st() * B * V))
    # where sqrt is the matrix square root and .st() is the plain transpose
    # (not conjugate).  Using V^T B_schur V (complex symmetric) matches the
    # oracle.  This makes the selected eigenvectors orthonormal in the B inner
    # product, as the reference requires for the TFE reduction.
    gram = vectors.T @ apply_B(vectors)
    scale = _inv(sqrtm(gram))
    mode_vec = vectors @ scale

    coeff = np.zeros((beta.size, beta.size), dtype=complex)
    np.fill_diagonal(coeff, np.sqrt(complex(0.0, k0 * Z0) / beta))
    mode_vecf = apply_B(vectors @ coeff @ scale)

    for m in range(mode_vec.shape[1]):
        pivot = int(np.argmax(np.abs(mode_vec[:, m].real)))
        if mode_vec[pivot, m].real < 0:
            mode_vec[:, m] *= -1.0
            mode_vecf[:, m] *= -1.0

    return beta, mode_vec, mode_vecf
