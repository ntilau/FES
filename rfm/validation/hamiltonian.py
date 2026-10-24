"""
Hamiltonian-based passivity checking and enforcement for the rfm RFM implementation.

A continuous-time LTI multi-port model with transfer matrix

    H(s) = D + sum_k R_k / (s - p_k)

is *passive* (positive real) iff:

  1. It is stable: every pole p_k lies in the open left half-plane (Re p_k < 0);
     poles on the imaginary axis are allowed only if simple with positive
     semi-definite (Hermitian) residues.
  2. D + D^H is positive semi-definite (high-frequency condition).
  3. H(j*w) + H(j*w)^H is Hermitian positive semi-definite for every real
     frequency w (positive realness).

This module implements two independent passivity tests:

* Frequency-domain test: directly evaluate H(j*w) + H(j*w)^H on the supplied
  frequency grid and verify every sampled matrix is positive semi-definite.
  This is the primary test — simple, correct, and independent of any
  state-space conversion.

* Hamiltonian test: form the Hamiltonian matrix

      H_ham = [ A,  -B R^{-1} B^H ;  -C^H R^{-1} C,  -A^H ],   R = D + D^H

  using a block-diagonal state-space realization of the pole-residue model.
  For a minimal realization with nonsingular R, H(j*w) + H(j*w)^H is
  nonsingular for every real w iff H_ham has no eigenvalues on the imaginary
  axis.  This serves as an independent cross-check.

Enforcement reflects right-half-plane poles into the left half-plane
(p -> -conj(p) with conjugated residues), regularizes a singular/near-singular
R matrix, and (for data) shifts each response matrix by delta*I to project its
Hermitian part onto the positive semi-definite cone.

Public API: check_hamiltonian, enforce_passivity, is_passive,
check_data_passivity, enforce_data_passivity.
"""

import copy
import numpy as np
from typing import Tuple, Optional, Union
from scipy import linalg
from ..macromodel import Macromodel


# ---------------------------------------------------------------------------
# Numerical tolerances
# ---------------------------------------------------------------------------

DEFAULT_TOL: float = 1e-8
"""Default numerical tolerance for semi-definiteness and eigenvalue tests."""

DEFAULT_IMAG_TOL: float = 1e-9
"""Tolerance below which a pole's real part is treated as on the imaginary axis."""

REG_EPS: float = 1e-12
"""Regularization floor for singular matrices."""


# ===========================================================================
# Input validation
# ===========================================================================

def _validate_model(model: Macromodel) -> dict:
    """Validate a Macromodel and return its contents normalized.

    Returns a dict with keys: Nports, poles, residues, dc_gain, n_poles.
    Poles are sorted; duplicate poles are merged by summing residues;
    numerically negligible poles are dropped.

    Raises ValueError for malformed input.
    """
    if model is None:
        raise ValueError("model must not be None")

    try:
        Nports = int(getattr(model, "Nports", 0))
    except (TypeError, ValueError):
        raise ValueError("Macromodel.Nports must be an integer") from None
    if Nports < 1:
        raise ValueError("Macromodel.Nports must be a positive integer")

    poles = getattr(model, "poles", None)
    residues = getattr(model, "residues", None)
    dc_gain = getattr(model, "dc_gain", None)

    # --- poles ---
    if poles is None:
        poles_arr = np.array([], dtype=complex)
    else:
        poles_arr = np.asarray(poles, dtype=complex).ravel()
        if poles_arr.size > 0 and not np.all(np.isfinite(poles_arr)):
            raise ValueError("Macromodel.poles must contain only finite values")

    # --- dc_gain ---
    if dc_gain is None:
        dc_arr = np.zeros((Nports, Nports), dtype=complex)
    else:
        dc_arr = np.asarray(dc_gain, dtype=complex)
        if dc_arr.ndim == 0:
            dc_arr = np.full((Nports, Nports), dc_arr.item(), dtype=complex)
        elif dc_arr.shape != (Nports, Nports):
            raise ValueError(
                f"Macromodel.dc_gain must be scalar or (Nports, Nports), "
                f"got shape {dc_arr.shape}"
            )
        if not np.all(np.isfinite(dc_arr)):
            raise ValueError("Macromodel.dc_gain must contain only finite values")

    # --- residues ---
    n_poles = int(poles_arr.size)
    if residues is None:
        res_arr = np.zeros((n_poles, Nports, Nports), dtype=complex)
    else:
        res_arr = np.asarray(residues, dtype=complex)
        if n_poles == 0:
            if res_arr.size != 0:
                raise ValueError("Model has no poles but residues are non-empty")
            res_arr = np.zeros((0, Nports, Nports), dtype=complex)
        elif res_arr.ndim == 1:
            if res_arr.size != n_poles:
                raise ValueError(
                    f"Macromodel.residues has {res_arr.size} entries but model "
                    f"has {n_poles} poles"
                )
            # SISO residue per pole -> diagonal matrix per pole
            res_arr = np.array([
                res_arr[k] * np.eye(Nports, dtype=complex) for k in range(n_poles)
            ])
        elif res_arr.ndim == 2:
            if res_arr.shape == (n_poles, Nports):
                res_arr = np.array([
                    np.diag(res_arr[k].astype(complex)) for k in range(n_poles)
                ])
            elif res_arr.shape == (Nports, Nports) and n_poles == 1:
                res_arr = res_arr.reshape(1, Nports, Nports).astype(complex)
            else:
                raise ValueError(
                    f"Cannot interpret 2D residues shape {res_arr.shape} "
                    f"for n_poles={n_poles}, Nports={Nports}"
                )
        elif res_arr.ndim == 3:
            if res_arr.shape != (n_poles, Nports, Nports):
                raise ValueError(
                    f"residues shape {res_arr.shape} != "
                    f"(n_poles={n_poles}, Nports={Nports}, Nports={Nports})"
                )
        else:
            raise ValueError(
                f"residues has unexpected ndim={res_arr.ndim}"
            )
        if not np.all(np.isfinite(res_arr)):
            raise ValueError("Macromodel.residues must contain only finite values")

    # --- drop numerically negligible poles ---
    if n_poles > 0:
        mag = np.abs(poles_arr)
        scale = max(float(np.max(mag)), 1e-300)
        keep = mag > 100.0 * np.finfo(float).eps * scale
        poles_arr = poles_arr[keep]
        res_arr = res_arr[keep]

    # --- merge duplicate poles ---
    if poles_arr.size > 1:
        order = np.lexsort((poles_arr.imag, poles_arr.real))
        poles_arr = poles_arr[order]
        res_arr = res_arr[order]
        scale = max(float(np.max(np.abs(poles_arr))), 1e-300)
        merged_poles = [poles_arr[0]]
        merged_res = [res_arr[0].copy()]
        for i in range(1, poles_arr.size):
            if abs(poles_arr[i] - merged_poles[-1]) <= 100.0 * np.finfo(float).eps * scale:
                merged_res[-1] = merged_res[-1] + res_arr[i]
            else:
                merged_poles.append(poles_arr[i])
                merged_res.append(res_arr[i].copy())
        poles_arr = np.asarray(merged_poles, dtype=complex)
        res_arr = np.asarray(merged_res, dtype=complex)

    return {
        "Nports": Nports,
        "poles": poles_arr,
        "residues": res_arr,
        "dc_gain": dc_arr,
        "n_poles": int(poles_arr.size),
    }


def _validate_freq(freq: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Validate frequency vector. Returns None if None/empty."""
    if freq is None:
        return None
    f = np.asarray(freq, dtype=float).ravel()
    if f.size == 0:
        return None
    if not np.all(np.isfinite(f)):
        raise ValueError("freq contains non-finite values")
    if np.any(f <= 0):
        raise ValueError("freq must contain strictly positive values")
    return f


def _hermitian_part(M: np.ndarray) -> np.ndarray:
    """Return (M + M^H) / 2."""
    return 0.5 * (M + np.conjugate(M).T)


def _is_psd(M: np.ndarray, tol: float = DEFAULT_TOL) -> bool:
    """Check whether a Hermitian matrix is positive semi-definite."""
    M = _hermitian_part(M)
    try:
        eigvals = linalg.eigvalsh(M)
    except linalg.LinAlgError:
        return False
    if eigvals.size == 0:
        return True
    lam_max = float(np.max(np.abs(eigvals)))
    return bool(np.all(eigvals >= -tol * max(1.0, lam_max)))


# ===========================================================================
# State-space realization (block-diagonal, clean for MIMO)
# ===========================================================================

def _pole_residue_to_state_space(poles: np.ndarray,
                                 residues: np.ndarray,
                                 dc_gain: np.ndarray,
                                 Nports: int
                                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert pole-residue model to block-diagonal state-space (A, B, C, D).

    For H(s) = D + sum_k R_k / (s - p_k):

        A = block_diag(p_1 * I, ..., p_n * I)           (n_p*Nports x n_p*Nports)
        B = [I; I; ...; I]                              (n_p*Nports x Nports)
        C = [R_1, R_2, ..., R_n]                        (Nports x n_p*Nports)
        D = dc_gain                                      (Nports x Nports)

    Then C(sI-A)^{-1}B = sum_k R_k/(s - p_k).
    """
    n_poles = int(poles.size)
    D = np.asarray(dc_gain, dtype=complex).reshape(Nports, Nports)

    if n_poles == 0:
        A = np.zeros((0, 0), dtype=complex)
        B = np.zeros((0, Nports), dtype=complex)
        C = np.zeros((Nports, 0), dtype=complex)
        return A, B, C, D

    # Block-diagonal A: each pole p_k contributes p_k * I_{Nports}
    A_blocks = [poles[k] * np.eye(Nports, dtype=complex) for k in range(n_poles)]
    A = linalg.block_diag(*A_blocks)

    # B: stack identity matrices vertically
    B = np.tile(np.eye(Nports, dtype=complex), (n_poles, 1))

    # C: concatenate residue matrices horizontally
    C = np.hstack([residues[k] for k in range(n_poles)])

    return A, B, C, D


# ===========================================================================
# Frequency-domain passivity test (primary)
# ===========================================================================

def _frequency_passivity_check(model_data: dict,
                               freq: np.ndarray,
                               tol: float = DEFAULT_TOL
                               ) -> Tuple[bool, dict]:
    """Evaluate H(j*w) + H(j*w)^H >= 0 at every frequency point.

    Returns (ok, info) with info holding the minimum eigenvalue seen and the
    frequency at which it occurred.
    """
    Nports = int(model_data["Nports"])
    poles = model_data["poles"]
    residues = model_data["residues"]
    dc_gain = model_data["dc_gain"]

    D = np.asarray(dc_gain, dtype=complex).reshape(Nports, Nports)

    info = {
        "min_eigenvalue": np.inf,
        "min_eigenvalue_freq": None,
        "num_points_checked": 0,
    }

    # High-frequency limit: D + D^H must be PSD.
    R_inf = _hermitian_part(D)
    try:
        r_eigvals = linalg.eigvalsh(R_inf)
        r_min = float(r_eigvals[0]) if r_eigvals.size else 0.0
    except linalg.LinAlgError:
        r_min = 0.0
    if r_min < -tol * max(1.0, abs(r_min)):
        info["min_eigenvalue"] = r_min
        info["min_eigenvalue_freq"] = np.inf
        return False, info

    min_eig = np.inf
    min_freq = None
    checked = 0

    for omega in freq:
        s = 1j * float(omega)
        H_freq = D.copy().astype(complex)
        for k in range(len(poles)):
            denom = s - poles[k]
            if abs(denom) < 1e-14:
                continue
            H_freq += residues[k] / denom

        W = H_freq + H_freq.conj().T
        W = _hermitian_part(W)
        try:
            w_eigvals = linalg.eigvalsh(W)
            e_min = float(w_eigvals[0])
        except linalg.LinAlgError:
            return False, info

        checked += 1
        if e_min < min_eig:
            min_eig = e_min
            min_freq = float(omega)
        if e_min < -tol * max(1.0, abs(e_min)):
            info["min_eigenvalue"] = e_min
            info["min_eigenvalue_freq"] = float(omega)
            info["num_points_checked"] = checked
            return False, info

    info["min_eigenvalue"] = min_eig
    info["min_eigenvalue_freq"] = min_freq
    info["num_points_checked"] = checked
    return True, info


# ===========================================================================
# Stability check
# ===========================================================================

def _check_stability(poles: np.ndarray,
                     residues: np.ndarray,
                     Nports: int,
                     tol: float = DEFAULT_TOL,
                     imag_tol: float = DEFAULT_IMAG_TOL
                     ) -> Tuple[bool, dict]:
    """Check stability and imaginary-axis pole conditions.

    All poles must have Re(p) < 0, except poles within imag_tol of the
    imaginary axis, which must be simple and have PSD Hermitian residues.
    """
    info = {"rhp_poles": [], "imag_axis_poles": []}
    if poles.size == 0:
        return True, info

    real_parts = np.real(poles)
    rhp_mask = real_parts > imag_tol
    imag_mask = np.abs(real_parts) <= imag_tol

    info["rhp_poles"] = [complex(p) for p in poles[rhp_mask]]
    info["imag_axis_poles"] = [complex(p) for p in poles[imag_mask]]

    if np.any(rhp_mask):
        return False, info

    # Imaginary-axis poles: must be simple with PSD residues.
    if np.any(imag_mask):
        scale = max(float(np.max(np.abs(poles))), 1e-300)
        idx = np.where(imag_mask)[0]
        used = np.zeros(idx.size, dtype=bool)
        for a in range(idx.size):
            if used[a]:
                continue
            same = np.where(
                ~used &
                (np.abs(np.imag(poles[idx]) - np.imag(poles[idx[a]]))
                 <= 100.0 * np.finfo(float).eps * scale)
            )[0]
            used[same] = True
            if same.size > 1:
                info["reason"] = "repeated pole on imaginary axis"
                return False, info
            k = idx[same[0]]
            Rk = _hermitian_part(residues[k])
            if not _is_psd(Rk, tol=tol):
                info["reason"] = "imaginary-axis pole with non-PSD residue"
                return False, info

    return True, info


# ===========================================================================
# Hamiltonian matrix test (independent cross-check)
# ===========================================================================

def _form_hamiltonian(A: np.ndarray,
                      B: np.ndarray,
                      C: np.ndarray,
                      D: np.ndarray
                      ) -> np.ndarray:
    """Form the Hamiltonian matrix H = [A, -B R^{-1} B^H; -C^H R^{-1} C, -A^H].

    R = D + D^H.  Regularized if singular.
    """
    R = _hermitian_part(D)
    try:
        r_eigvals = linalg.eigvalsh(R)
        r_min = float(r_eigvals[0]) if r_eigvals.size else 0.0
    except linalg.LinAlgError:
        r_min = 0.0

    if r_min <= REG_EPS:
        reg = max(REG_EPS, abs(r_min) + 10 * REG_EPS)
        R = R + reg * np.eye(R.shape[0], dtype=complex)

    R_inv = linalg.inv(R)
    n = A.shape[0]

    top_right = -B @ R_inv @ np.conjugate(B).T
    # C^H R^{-1} C: C is (Nports, n*Nports), C^H is (n*Nports, Nports)
    bottom_left = -(np.conjugate(C).T @ R_inv @ C)
    bottom_right = -np.conjugate(A).T

    return np.block([[A, top_right],
                     [bottom_left, bottom_right]])


def _hamiltonian_eigenvalue_test(A: np.ndarray,
                                 B: np.ndarray,
                                 C: np.ndarray,
                                 D: np.ndarray,
                                 tol: float = DEFAULT_TOL,
                                 imag_tol: float = DEFAULT_IMAG_TOL
                                 ) -> Tuple[bool, dict]:
    """Test that the Hamiltonian matrix has no imaginary-axis eigenvalues.

    Returns (ok, info).
    """
    info = {}
    n = int(A.shape[0])
    if n == 0:
        info["eigenvalues"] = np.array([], dtype=complex)
        info["min_imag_distance"] = np.inf
        return True, info

    try:
        H_ham = _form_hamiltonian(A, B, C, D)
        eigvals = linalg.eigvals(H_ham)
    except linalg.LinAlgError as exc:
        raise RuntimeError(f"Hamiltonian computation failed: {exc}") from exc

    eigvals = np.sort_complex(eigvals)
    imag_distances = np.abs(np.real(eigvals))
    min_imag_distance = float(np.min(imag_distances))
    info["eigenvalues"] = eigvals
    info["min_imag_distance"] = min_imag_distance

    ok = bool(min_imag_distance > imag_tol)
    return ok, info


def _hamiltonian_passive(model_data: dict,
                         freq: np.ndarray,
                         tol: float = DEFAULT_TOL,
                         imag_tol: float = DEFAULT_IMAG_TOL
                         ) -> Tuple[bool, dict]:
    """Full Hamiltonian passivity test.

    Combines:
      1. Stability (all poles in open LHP, or simple jw-axis poles with PSD residues)
      2. Hamiltonian eigenvalue test (no imaginary-axis eigenvalues)
      3. Frequency-domain sign check (H(jw)+H(jw)^H PSD at sampled points)
    """
    Nports = int(model_data["Nports"])
    poles = model_data["poles"]
    residues = model_data["residues"]
    info = {}

    # 1. Stability
    stable, stability_info = _check_stability(poles, residues, Nports, tol, imag_tol)
    info["stable"] = stable
    info["stability"] = stability_info
    if not stable:
        info["reason"] = "unstable poles (right-half-plane or invalid imaginary-axis poles)"
        return False, info

    # 2. State-space realization
    A, B, C, D = _pole_residue_to_state_space(
        poles, residues, model_data["dc_gain"], Nports)

    # 3. Hamiltonian eigenvalue test
    ham_ok, ham_info = _hamiltonian_eigenvalue_test(
        A, B, C, D, tol=tol, imag_tol=imag_tol)
    info["hamiltonian"] = ham_info
    if not ham_ok:
        info["reason"] = "Hamiltonian matrix has eigenvalues on the imaginary axis"
        return False, info

    # 4. Frequency-domain cross-check
    fd_ok, fd_info = _frequency_passivity_check(model_data, freq, tol=tol)
    info["frequency_domain"] = fd_info
    if not fd_ok:
        info["reason"] = "H(jw) + H(jw)^H not positive semi-definite at sampled frequency"
        return False, info

    info["reason"] = "passive"
    return True, info


# ===========================================================================
# Passivity enforcement
# ===========================================================================

def _reflect_rhp_poles(poles: np.ndarray,
                       residues: np.ndarray
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """Reflect RHP poles into the LHP: p -> -conj(p), residues conjugated."""
    poles = poles.copy()
    residues = residues.copy()
    for k in range(poles.size):
        if poles[k].real > 0.0:
            poles[k] = -np.conj(poles[k])
            residues[k] = np.conjugate(residues[k])
    return poles, residues


def _enforce_passivity_state_space(model_data: dict,
                                   freq: np.ndarray,
                                   tol: float = DEFAULT_TOL,
                                   imag_tol: float = DEFAULT_IMAG_TOL
                                   ) -> dict:
    """Enforce passivity on a validated model dictionary (input not mutated)."""
    out = copy.deepcopy(model_data)
    poles = out["poles"].copy()
    residues = out["residues"].copy()
    D = out["dc_gain"].copy()
    Nports = out["Nports"]

    # Step 1: reflect RHP poles into LHP
    poles, residues = _reflect_rhp_poles(poles, residues)
    out["poles"] = poles
    out["residues"] = residues

    # Step 2: ensure D + D^H >= 0 by shifting D
    R = _hermitian_part(D)
    if not _is_psd(R, tol=tol):
        try:
            eigvals = linalg.eigvalsh(R)
            lam_min = float(eigvals[0]) if eigvals.size else 0.0
        except linalg.LinAlgError:
            lam_min = 0.0
        shift = (tol * max(1.0, abs(lam_min)) - lam_min) + 100.0 * np.finfo(float).eps
        D = D + shift * np.eye(Nports, dtype=complex)
        out["dc_gain"] = D

    # Step 3: verify; if still failing, iteratively shift D at the worst frequency
    passive, info = _hamiltonian_passive(out, freq, tol=tol, imag_tol=imag_tol)
    if not passive:
        for _ in range(50):
            fd_ok, fd_info = _frequency_passivity_check(out, freq, tol=tol)
            if fd_ok:
                passive = True
                break
            e_min = float(fd_info.get("min_eigenvalue", 0.0))
            w_star = float(fd_info.get("min_eigenvalue_freq", 0.0) or 0.0)
            A, B, C, Dcur = _pole_residue_to_state_space(
                out["poles"], out["residues"], out["dc_gain"], Nports)
            s = 1j * w_star
            H = Dcur.copy()
            for k in range(len(out["poles"])):
                denom = s - out["poles"][k]
                if abs(denom) >= 1e-14:
                    H = H + out["residues"][k] / denom
            W = _hermitian_part(H + H.conj().T)
            try:
                eigvals = linalg.eigvalsh(W)
                if eigvals[0] >= -tol:
                    break
            except linalg.LinAlgError:
                break
            delta = (-float(eigvals[0])) + tol * max(1.0, abs(float(eigvals[0]))) + 1e-8
            out["dc_gain"] = out["dc_gain"] + delta * np.eye(Nports, dtype=complex)
        else:
            passive, info = _hamiltonian_passive(out, freq, tol=tol, imag_tol=imag_tol)

    out["passive"] = bool(passive)
    out["enforcement_info"] = info
    return out


# ===========================================================================
# Public API
# ===========================================================================

def check_hamiltonian(model: Macromodel,
                      freq: np.ndarray,
                      tol: float = DEFAULT_TOL,
                      imag_tol: float = DEFAULT_IMAG_TOL
                      ) -> Tuple[bool, Macromodel]:
    """Check passivity of a macromodel using the Hamiltonian matrix approach.

    Performs input validation, stability testing, D + D^H positive
    semi-definiteness, the Hamiltonian imaginary-axis eigenvalue test, and a
    frequency-domain positive-realness cross-check.

    Parameters
    ----------
    model : Macromodel
        The macromodel to check (pole-residue form).
    freq : np.ndarray
        Frequency vector for the independent cross-check (positive, finite).
    tol : float, optional
        Numerical tolerance (default 1e-8).
    imag_tol : float, optional
        Imaginary-axis pole tolerance (default 1e-9).

    Returns
    -------
    is_passive : bool
        True if the model is passive (positive real).
    passive_model : Macromodel
        The original model if passive; otherwise a passivated copy.
    """
    model_data = _validate_model(model)
    f = _validate_freq(freq)

    passive, info = _hamiltonian_passive(model_data, f, tol=tol, imag_tol=imag_tol)

    if passive:
        return True, model

    passive_data = _enforce_passivity_state_space(
        model_data, f, tol=tol, imag_tol=imag_tol)
    passive_model = _model_data_to_model(passive_data, template=model)
    return False, passive_model


def enforce_passivity(model: Macromodel,
                      freq: np.ndarray,
                      tol: float = DEFAULT_TOL,
                      imag_tol: float = DEFAULT_IMAG_TOL
                      ) -> Macromodel:
    """Enforce passivity on a macromodel.

    Returns a passive copy.  If the input is already passive, a deep copy is
    returned unchanged.
    """
    model_data = _validate_model(model)
    f = _validate_freq(freq)

    passive, _ = _hamiltonian_passive(model_data, f, tol=tol, imag_tol=imag_tol)
    if passive:
        return copy.deepcopy(model)

    passive_data = _enforce_passivity_state_space(
        model_data, f, tol=tol, imag_tol=imag_tol)
    return _model_data_to_model(passive_data, template=model)


def is_passive(model: Macromodel,
               freq: np.ndarray,
               tol: float = DEFAULT_TOL,
               imag_tol: float = DEFAULT_IMAG_TOL
               ) -> bool:
    """Check if a macromodel is passive.

    Returns True if the model passes all passivity tests.
    """
    model_data = _validate_model(model)
    f = _validate_freq(freq)
    passive, _ = _hamiltonian_passive(model_data, f, tol=tol, imag_tol=imag_tol)
    return passive


def _normalize_response(response: np.ndarray,
                        n_freq: int) -> Optional[np.ndarray]:
    """Normalize response to shape (Nfreq, Nports, Nports).

    Accepts (Nfreq,) scalar data (Nports=1), (Nports, Nports, Nfreq), or
    (Nfreq, Nports, Nports).  Returns None if unrecognizable.
    """
    H = np.asarray(response, dtype=complex)
    if H.ndim == 1:
        if H.size != n_freq:
            return None
        return H.reshape(n_freq, 1, 1)
    if H.ndim == 3:
        if H.shape[0] == n_freq and H.shape[1] == H.shape[2]:
            return H
        if H.shape[2] == n_freq and H.shape[0] == H.shape[1]:
            return np.transpose(H, (2, 0, 1))
        return None
    return None


def check_data_passivity(freq: np.ndarray,
                         response: np.ndarray,
                         tol: float = DEFAULT_TOL) -> bool:
    """Check passivity of frequency-response data via positive realness.

    For every frequency, verifies H(j*w) + H(j*w)^H is positive semi-definite.
    For scalar data this reduces to 2 * Re H(j*w) >= 0.

    Parameters
    ----------
    freq : np.ndarray
        Frequency vector.
    response : np.ndarray
        Complex frequency response: (Nfreq,), (Nports, Nports, Nfreq), or
        (Nfreq, Nports, Nports).
    tol : float, optional
        Numerical tolerance (default 1e-8).

    Returns
    -------
    bool
        True if data is positive real at every sampled frequency.
    """
    f = _validate_freq(freq)
    H = _normalize_response(response, f.size)
    if H is None:
        raise ValueError(
            "response must have shape (Nfreq,), (Nports, Nports, Nfreq) or "
            "(Nfreq, Nports, Nports)"
        )

    for k in range(f.size):
        W = H[k] + np.conjugate(H[k]).T
        if not _is_psd(W, tol=tol):
            return False
    return True


def enforce_data_passivity(freq: np.ndarray,
                           response: np.ndarray,
                           tol: float = DEFAULT_TOL
                           ) -> Tuple[np.ndarray, np.ndarray]:
    """Enforce passivity on frequency-response data.

    At each frequency where the Hermitian part is not PSD, shifts the
    response by delta*I (minimal Hermitian shift) to make it PSD.

    Parameters
    ----------
    freq : np.ndarray
        Frequency vector.
    response : np.ndarray
        Complex frequency response (same shapes as check_data_passivity).
    tol : float, optional
        Numerical tolerance (default 1e-8).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (freq, passive_response) with the same shape as the input.
    """
    f = _validate_freq(freq)
    H = _normalize_response(response, f.size)
    if H is None:
        raise ValueError(
            "response must have shape (Nfreq,), (Nports, Nports, Nfreq) or "
            "(Nfreq, Nports, Nports)"
        )

    scalar_input = np.asarray(response).ndim == 1
    out = H.copy()
    for k in range(f.size):
        W = _hermitian_part(out[k] + np.conjugate(out[k]).T)
        try:
            eigvals = linalg.eigvalsh(W)
        except linalg.LinAlgError:
            eigvals = np.linalg.eigvalsh(W)
        e_min = float(eigvals[0])
        if e_min < -tol * max(1.0, abs(e_min)):
            delta = (-e_min) + tol * max(1.0, abs(e_min)) + 100.0 * np.finfo(float).eps
            out[k] = out[k] + delta * np.eye(out.shape[-1], dtype=complex)

    if scalar_input:
        out = out[:, 0, 0]
    return f, out


# ===========================================================================
# Internal helpers
# ===========================================================================

def _model_data_to_model(model_data: dict,
                         template: Macromodel) -> Macromodel:
    """Build a Macromodel from a validated model data dictionary."""
    out = copy.deepcopy(template)
    out.Nports = int(model_data["Nports"])
    out.representation = "pole-residue"
    out.poles = model_data["poles"]
    out.residues = model_data["residues"]
    out.dc_gain = model_data["dc_gain"]
    # Populate state-space realization for downstream consumers.
    A, B, C, D = _pole_residue_to_state_space(
        model_data["poles"], model_data["residues"],
        model_data["dc_gain"], int(model_data["Nports"]))
    out.A, out.B, out.C, out.D = A, B, C, D
    return out
