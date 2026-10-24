"""
Vector Fitting algorithm for the rfm RFM implementation.

Implements the Vector Fitting procedure of Gustavsen and Semlyen for rational
approximation of frequency-domain responses.  A response is approximated by

    H(s) = d + sum_k r_k / (s - p_k),        s = j * freq

where the poles ``p_k`` are relocated iteratively by solving the linearized
pole-identification problem.
"""

import numpy as np
from typing import Tuple
from scipy import linalg, optimize

from ..dum import DUM
from ..macromodel import Macromodel


def _initial_poles(normalized_freq: np.ndarray, n_poles: int) -> np.ndarray:
    """Return stable starting poles spread over the normalized band.

    Complex conjugate pairs are used (the usual Vector Fitting choice); an odd
    pole count adds one real pole at the geometric centre of the band.
    """
    fmin = float(np.min(normalized_freq))
    fmax = float(np.max(normalized_freq))

    n_pairs = n_poles // 2
    if n_pairs:
        betas = np.logspace(np.log10(fmin), np.log10(fmax), n_pairs)
    else:
        betas = np.array([], dtype=float)

    poles = []
    for beta in betas:
        alpha = -beta / 100.0
        poles.extend([alpha + 1j * beta, alpha - 1j * beta])
    if n_poles % 2:
        poles.append(-np.sqrt(fmin * fmax))
    return np.asarray(poles, dtype=complex)


def _relocate_poles(s, response, poles, weights, max_iter, conv_tol, relax,
                    enforce_stability):
    """Iterate the Vector Fitting pole-relocation step."""
    n_poles = len(poles)
    n_freq = len(s)
    previous = None

    for _ in range(max_iter):
        phi = 1.0 / (s[:, None] - poles[None, :])
        # [phi, 1, -H*phi] [c; d; c_tilde] = H
        system = np.hstack([
            phi,
            np.ones((n_freq, 1), dtype=complex),
            -response[:, None] * phi,
        ])
        coeffs, _, _, _ = linalg.lstsq(
            system * weights[:, None], response * weights
        )
        c_tilde = coeffs[1 + n_poles:]

        # Zeros of the denominator are the eigenvalues of
        # diag(poles) - 1 * c_tilde^T.
        candidate = linalg.eigvals(
            np.diag(poles) - np.outer(np.ones(n_poles), c_tilde)
        )
        if enforce_stability:
            candidate = np.where(
                candidate.real > 0, -np.conj(candidate), candidate
            )

        poles = relax * candidate + (1.0 - relax) * poles

        if previous is not None:
            change = (np.max(np.abs(poles - previous))
                      / (np.max(np.abs(previous)) + 1e-12))
            if change < conv_tol:
                break
        previous = poles.copy()

    return poles


def vector_fit(freq: np.ndarray, response: np.ndarray,
               max_poles: int = 20,
               max_iter: int = 100,
               conv_tol: float = 1e-8,
               relax: float = 0.5,
               as_poles: bool = True,
               enforce_stability: bool = True
               ) -> Tuple[np.ndarray, np.ndarray, complex]:
    """
    Vector Fitting algorithm for rational approximation of frequency responses.

    Parameters
    ----------
    freq : np.ndarray
        Frequency vector (must be positive and increasing).  Units are
        arbitrary and consistent with ``response``; poles are returned in the
        same units.
    response : np.ndarray
        Complex frequency response to fit.
    max_poles : int, optional
        Maximum number of poles to use, default is 20.  The actual count is
        reduced when there are too few frequency samples to determine the
        model.
    max_iter : int, optional
        Maximum number of pole-relocation iterations, default is 100.
    conv_tol : float, optional
        Relative pole-change convergence tolerance, default is 1e-8.
    relax : float, optional
        Relaxation factor for pole updates (0 < relax <= 1).  Lower values
        (e.g. 0.3-0.7) improve convergence stability, default is 0.5.
    as_poles : bool, optional
        Retained for API compatibility.  When True (default) the returned
        poles are stable.
    enforce_stability : bool, optional
        If True, reflect right-half-plane poles into the left half-plane
        during iteration, default is True.

    Returns
    -------
    poles : np.ndarray
        Complex poles of the fitted model, in the units of ``freq``.
    residues : np.ndarray
        Complex residues corresponding to each pole.
    d_term : complex
        Direct (high-frequency) feedthrough term.

    Notes
    -----
    The pole-relocation step uses the eigenvalue formulation
    ``diag(p) - 1 * c_tilde^T`` instead of ``np.poly``/``np.roots``, which
    avoids numerical overflow for high pole counts.  Frequencies are scaled
    internally to the band for conditioning and the poles and residues are
    scaled back before returning.
    """
    freq = np.asarray(freq, dtype=float).ravel()
    response = np.asarray(response, dtype=complex).ravel()

    if freq.size != response.size:
        raise ValueError("freq and response must have the same length")
    if freq.size < 2:
        raise ValueError("At least two frequency samples are required")
    if not np.all(np.isfinite(freq)) or np.any(freq <= 0):
        raise ValueError("Frequency values must be positive and finite")
    if not np.all(np.isfinite(response)):
        raise ValueError("Response values must be finite")
    if max_poles <= 0:
        raise ValueError("max_poles must be positive")
    if not 0.0 < relax <= 1.0:
        raise ValueError("relax must be in (0, 1]")

    # Keep the linear problem determined; return at most n_poles poles.
    n_poles = min(int(max_poles), freq.size // 2)
    if n_poles < 1:
        raise ValueError("Not enough frequency samples for a rational fit")

    # Normalize the band so that the pole-identification problem is
    # well conditioned for frequencies expressed in Hz or rad/s.
    f_ref = float(np.exp(np.mean(np.log(freq))))
    normalized_freq = freq / f_ref
    s = 1j * normalized_freq

    poles = _initial_poles(normalized_freq, n_poles)
    weights = 1.0 / np.maximum(np.abs(response), 1e-12)
    poles = _relocate_poles(
        s, response, poles, weights, max_iter, conv_tol, relax,
        enforce_stability or as_poles,
    )

    # Final residues and direct term for the identified poles.
    phi = 1.0 / (s[:, None] - poles[None, :])
    system = np.hstack([phi, np.ones((freq.size, 1), dtype=complex)])
    coeffs, _, _, _ = linalg.lstsq(
        system * weights[:, None], response * weights
    )
    residues = coeffs[:-1]
    d_term = coeffs[-1]

    # Sort by pole magnitude for a deterministic ordering.
    order = np.argsort(np.abs(poles))
    poles = poles[order]
    residues = residues[order]

    # Scale back to the caller's frequency units: p = p' * f_ref and
    # r = r' * f_ref leave r/(j*f - p) unchanged.
    return poles * f_ref, residues * f_ref, complex(d_term)


def vector_fit_inf_dc0(freq: np.ndarray, response: np.ndarray,
                       max_poles: int = 20,
                       max_iter: int = 100,
                       relax: float = 0.5
                       ) -> Tuple[np.ndarray, np.ndarray, complex]:
    """
    Vector fitting with a direct (infinity) term.

    The upstream name refers to an infinity/DC-constrained variant; this
    implementation performs the standard fit with a direct term and does not
    impose a DC constraint.  It is kept as a compatibility alias.
    """
    return vector_fit(freq, response, max_poles=max_poles,
                      max_iter=max_iter, relax=relax,
                      as_poles=True, enforce_stability=True)


def _rfm_eval(freq: np.ndarray, poles: np.ndarray, residues: np.ndarray,
              dc_gain: complex) -> np.ndarray:
    """Evaluate the pole-residue RFM H(jw) = sum(r_k / (jw - p_k)) + dc_gain."""
    jw = 1j * np.asarray(freq, dtype=float)
    h = np.zeros_like(jw, dtype=complex)
    residues = np.asarray(residues).ravel()
    dc_gain = np.asarray(dc_gain).ravel()[0]
    for p, r in zip(poles, residues):
        h += r / (jw - p)
    h += dc_gain
    return h


def _rfm_residual(params: np.ndarray, freq: np.ndarray, response: np.ndarray,
                  Np: int) -> np.ndarray:
    """Real-valued residual vector for least-squares fitting.

    Parameters are packed as:
      [pole_real (Np), pole_imag (Np), res_real (Np), res_imag (Np),
       d_real, d_imag]
    """
    poles = params[0:Np] + 1j * params[Np:2 * Np]
    residues = params[2 * Np:3 * Np] + 1j * params[3 * Np:4 * Np]
    d = params[4 * Np] + 1j * params[4 * Np + 1]
    model = _rfm_eval(freq, poles, residues, d)
    error = response - model
    return np.concatenate([error.real, error.imag])


def _unpack_params(params: np.ndarray, Np: int
                   ) -> Tuple[np.ndarray, np.ndarray, complex]:
    poles = params[0:Np] + 1j * params[Np:2 * Np]
    residues = params[2 * Np:3 * Np] + 1j * params[3 * Np:4 * Np]
    d = params[4 * Np] + 1j * params[4 * Np + 1]
    return poles, residues, d


def vector_fit_refined(freq: np.ndarray, response: np.ndarray,
                       max_poles: int = 20,
                       max_iter: int = 100,
                       conv_tol: float = 1e-8,
                       relax: float = 0.5,
                       n_starts: int = 5,
                       use_refinement: bool = True,
                       seed: int = 42,
                       as_poles: bool = True,
                       enforce_stability: bool = True
                       ) -> Tuple[np.ndarray, np.ndarray, complex]:
    """
    Vector Fitting with least-squares refinement via multiple random starts.

    Standard Vector Fitting can settle in a local minimum.  This function runs
    :func:`vector_fit` for an initial model, then refines it with
    ``scipy.optimize.least_squares`` from several random starts, keeping the
    best model.

    Parameters
    ----------
    freq, response, max_poles, max_iter, conv_tol, relax, as_poles,
    enforce_stability :
        See :func:`vector_fit`.
    n_starts : int, optional
        Number of least-squares starts (including the VF start), default 5.
    use_refinement : bool, optional
        If True, refine the VF result, default True.
    seed : int, optional
        Random seed for reproducible starts, default 42.

    Returns
    -------
    poles, residues, d_term :
        As for :func:`vector_fit`.
    """
    freq = np.asarray(freq, dtype=float).ravel()
    response = np.asarray(response, dtype=complex).ravel()

    poles_vf, res_vf, d_vf = vector_fit(
        freq, response, max_poles=max_poles, max_iter=max_iter,
        conv_tol=conv_tol, relax=relax,
        as_poles=as_poles, enforce_stability=enforce_stability,
    )

    if not use_refinement or n_starts <= 1:
        return poles_vf, res_vf, d_vf

    Np = len(poles_vf)
    f_ref = float(np.exp(np.mean(np.log(freq))))
    h_ref = float(np.mean(np.abs(response))) or 1.0

    def _err(poles, residues, d):
        model = _rfm_eval(freq, poles, residues, d)
        error = response - model
        return (np.sqrt(np.mean(np.abs(error) ** 2))
                / np.sqrt(np.mean(np.abs(response) ** 2)))

    best_err = _err(poles_vf, res_vf, d_vf)
    best = (poles_vf, res_vf, d_vf)

    starts = []
    p0_vf = np.concatenate([
        poles_vf.real, poles_vf.imag,
        res_vf.real, res_vf.imag,
        [d_vf.real, d_vf.imag],
    ])
    starts.append(p0_vf)

    rng = np.random.default_rng(seed)
    for _ in range(n_starts - 1):
        pole_real = -rng.uniform(0.01, 30.0, Np) * f_ref
        pole_imag = rng.uniform(-30.0, 30.0, Np) * f_ref
        res_real = rng.standard_normal(Np) * 0.05 * h_ref * f_ref
        res_imag = rng.standard_normal(Np) * 0.05 * h_ref * f_ref
        d_real = np.mean(response.real) + rng.standard_normal() * 0.03 * h_ref
        d_imag = np.mean(response.imag) + rng.standard_normal() * 0.03 * h_ref
        starts.append(np.concatenate([
            pole_real, pole_imag, res_real, res_imag, [d_real, d_imag],
        ]))

    # Reject degenerate solutions far outside the measured band.
    pole_bound = 1000.0 * f_ref
    min_rel_improvement = 0.10

    for p0 in starts:
        try:
            result = optimize.least_squares(
                _rfm_residual, p0, args=(freq, response, Np),
                method='trf', max_nfev=1000,
            )
            poles_fit, res_fit, d_fit = _unpack_params(result.x, Np)

            if (np.any(np.abs(poles_fit.real) > pole_bound)
                    or np.any(np.abs(poles_fit.imag) > pole_bound)):
                continue

            rms_err = _err(poles_fit, res_fit, d_fit)
            if rms_err < best_err * (1.0 - min_rel_improvement):
                best_err = rms_err
                best = (poles_fit, res_fit, d_fit)
        except (ValueError, linalg.LinAlgError):
            # Skip failed starts; keep the best model so far.
            continue

    return best


# Aliases for backward compatibility
VectorFit_inf_dc0 = vector_fit_inf_dc0


def _not_implemented(name):
    def _stub(*args, **kwargs):
        raise NotImplementedError(
            f"{name} is a placeholder in the upstream RFM code and is not "
            "implemented in the bundled package"
        )

    _stub.__name__ = name
    return _stub


En_Skimming = _not_implemented("En_Skimming")
en_skimming = En_Skimming
AddingSkimming = _not_implemented("AddingSkimming")
adding_skimming = AddingSkimming
Final_Skimming = _not_implemented("Final_Skimming")
final_skimming = Final_Skimming
MultiVectorFit = _not_implemented("MultiVectorFit")
multivectorfit = MultiVectorFit
MultiVectorFitPD = _not_implemented("MultiVectorFitPD")


class VectorFit:
    """
    Vector Fitting class interface.
    """

    def __init__(self, max_poles=20, max_iter=100, conv_tol=1e-8,
                 relax=0.5, as_poles=True, enforce_stability=True,
                 use_refinement=False, n_starts=5, seed=42):
        self.max_poles = max_poles
        self.max_iter = max_iter
        self.conv_tol = conv_tol
        self.relax = relax
        self.as_poles = as_poles
        self.enforce_stability = enforce_stability
        self.use_refinement = use_refinement
        self.n_starts = n_starts
        self.seed = seed
        self.poles = None
        self.residues = None
        self.d_term = None

    def fit(self, freq, response):
        if self.use_refinement:
            self.poles, self.residues, self.d_term = vector_fit_refined(
                freq, response,
                max_poles=self.max_poles,
                max_iter=self.max_iter,
                conv_tol=self.conv_tol,
                relax=self.relax,
                n_starts=self.n_starts,
                seed=self.seed,
                as_poles=self.as_poles,
                enforce_stability=self.enforce_stability,
            )
        else:
            self.poles, self.residues, self.d_term = vector_fit(
                freq, response,
                max_poles=self.max_poles,
                max_iter=self.max_iter,
                conv_tol=self.conv_tol,
                relax=self.relax,
                as_poles=self.as_poles,
                enforce_stability=self.enforce_stability,
            )
        return self.poles, self.residues, self.d_term
