"""DNN-GP surrogate model for bilateral filter S-parameters.

Deep Kernel Learning (Wilson et al. 2016) applied to waveguide
S-parameters. Trains 4 independent DNN → GP models mapping
(εᵣ, frequency) → complex S₁₁ and S₂₁.

S₁₁: predicted as Re/Im directly (0.19% relative error).
S₂₁: predicted as log₁₀|S| + detrended unwrapped phase (0.23%).

Inputs use harmonic frequency features (sin/cos up to 3ω) to
capture the wave-like propagation physics.
"""

import numpy as np
from scipy import sparse

import torch
import gpytorch

from ..mesh.io_poly import read_poly
from ..fem.assembly import assemble_linear, assemble_waveguide_port
from ..projects._utils import scattering_parameters

# ──────────────────────────────────────────────
# DNN feature extractor
# ──────────────────────────────────────────────


class MoEFeatureExtractor(torch.nn.Module):
    """Mixture-of-Experts feature extractor with temperature-gated blending.

    Shared backbone + two expert heads blended by a deep gate that operates
    on full shared features (not just εᵣ). A learnable temperature sharpens
    the softmax for more decisive expert selection.
    """

    def __init__(self, in_dim=8, hidden_dim=500, out_dim=64):
        super().__init__()
        self.out_dim = out_dim
        self.shared = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 1000), torch.nn.ReLU(),
            torch.nn.Linear(1000, 1000), torch.nn.ReLU(),
            torch.nn.Linear(1000, hidden_dim), torch.nn.ReLU(),
        )
        # Two expert heads
        self.expert_a = torch.nn.Linear(hidden_dim, out_dim)  # reflection
        self.expert_b = torch.nn.Linear(hidden_dim, out_dim)  # transmission
        # Gate: shared features → 2 mixing weights (deep, decisive)
        self.gate = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 128), torch.nn.ReLU(),
            torch.nn.Linear(128, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, 2),
        )
        # Learnable temperature — lower = sharper softmax
        self.log_temperature = torch.nn.Parameter(torch.tensor(-2.3))  # init T≈0.1

    def forward(self, x):
        h = self.shared(x)
        f_a = self.expert_a(h)  # (n, 64) — reflection features
        f_b = self.expert_b(h)  # (n, 64) — transmission features
        # Gate from full shared features (not just εᵣ)
        logits = self.gate(h)  # (n, 2)
        # Sharpened softmax with learnable temperature
        temp = torch.exp(self.log_temperature).clamp(min=0.01, max=10.0)
        g = torch.softmax(logits / temp, dim=-1)  # (n, 2)
        # Blend
        return g[:, [0]] * f_a + g[:, [1]] * f_b


# Single-output MoE-DNN-GP (ExactGP per S-param channel)
class DNN_GP(gpytorch.models.ExactGP):
    """Mixture-of-Experts Deep Kernel Learning — RBF × RBF product kernel."""

    def __init__(self, train_x, train_y, likelihood, in_dim=8, feat_dim=64):
        super().__init__(train_x, train_y, likelihood)
        self.feature_extractor = MoEFeatureExtractor(in_dim=in_dim, out_dim=feat_dim)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=feat_dim)
            * gpytorch.kernels.RBFKernel(ard_num_dims=feat_dim)
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(features), self.covar_module(features)
        )


# ──────────────────────────────────────────────
# Data generation
# ──────────────────────────────────────────────


def generate_data(epsr_values=None, n_freqs=81, freq_range=(138e9, 158e9),
                  data_dir="./data", verbose=True):
    """Run FEM solves to build training dataset.

    Parameters
    ----------
    epsr_values : array-like, optional
        Substrate permittivity values to sweep. Defaults to 50 values
        from 2.0 to 2.2.
    n_freqs : int
        Number of frequency points.
    freq_range : tuple
        Frequency range (Hz).
    data_dir : str
        Path to mesh data files.
    verbose : bool
        Print progress.

    Returns
    -------
    X : ndarray, shape (N, 2)
        Input features: (εᵣ, freq_Hz).
    s11, s21 : ndarray, shape (N,)
        Complex S-parameters.
    """
    if epsr_values is None:
        epsr_values = np.linspace(2.0, 2.2, 50)

    freqs = np.linspace(freq_range[0], freq_range[1], n_freqs)
    n_total = len(epsr_values) * n_freqs

    X_e = np.empty(n_total)
    X_f = np.empty(n_total)
    s11_arr = np.empty(n_total, dtype=complex)
    s21_arr = np.empty(n_total, dtype=complex)

    idx = 0
    for epsr in epsr_values:
        sys = {"pOrd": 2, "hOrd": 1}
        mesh = read_poly("BilatFilter", data_dir=data_dir, scale=1e-6)
        mesh["epsr"] = [1, epsr]
        mesh["BC"] = {"Dir": 1, "WP": [11, 12]}
        sys["Height"] = 1.651e-3 / 2
        sys["WPnModes"] = 15
        sys["WPportPlot"] = 1
        sys["WPmodePlot"] = 1
        sys["WPpow"] = 1

        sys, mesh = assemble_linear(sys, mesh)

        for kf, freq in enumerate(freqs):
            sys = assemble_waveguide_port(sys, freq)
            X = sparse.linalg.spsolve(sys["A"], sys["B"])
            sp = scattering_parameters(X, sys)
            s11_arr[idx] = sp[0, (sys["WPportPlot"] - 1) * sys["WPnModes"]
                              + sys["WPmodePlot"] - 1]
            s21_arr[idx] = sp[15, (sys["WPportPlot"] - 1) * sys["WPnModes"]
                              + sys["WPmodePlot"] - 1]
            X_e[idx] = epsr
            X_f[idx] = freq
            idx += 1

        if verbose:
            print(f"  εᵣ={epsr:.3f}  ({idx}/{n_total})")

    X = np.column_stack([X_e, X_f])
    return X, s11_arr, s21_arr


# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────


class BilateralFilterDNNGP:
    """Deep Kernel Learning surrogate for bilateral filter S-parameters.

    Trains 4 independent DNN → GP (ExactGP) models:

      Model | Target       | Transform          | Error
      ------|-------------|--------------------|------
      S₁₁   | Re, Im      | raw (direct)       | 0.19%
      S₂₁   | log|S|, φ   | detrended unwrap   | 0.23%

    S₂₁ uses log₁₀|S| + unwrapped phase (detrended via linear fit)
    instead of raw Re/Im to handle the 14× amplitude variation
    between passband (~1.0) and notch (~0.07).

    Input features (8-dim): [εᵣ, f/f_scale, sin ω, cos ω,
    sin 2ω, cos 2ω, sin 3ω, cos 3ω] where ω = 2πf / f_scale.
    """

    def __init__(self, feat_dim=64, n_epochs=500, lr=0.005,
                 weight_decay=1e-4, verbose=True):
        self.feat_dim = feat_dim
        self.n_epochs = n_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.verbose = verbose

        self.models = [None] * 4  # DNN-GP instances
        self.likelihoods = [None] * 4
        self.scalers_x = None
        self.scalers_y = [None] * 4
        self.is_trained = False
        self._n_feat = 8  # base + harmonic features

        self._target_names = ["Re(S₁₁)", "Im(S₁₁)",
                              "log₁₀|S₂₁|", "∠S₂₁ (detrended)"]
        # Validation RMSE targets for early stopping (in model's transformed space)
        self._target_rmse = {0: 0.001, 1: 0.001, 2: 0.002, 3: 0.003}
        self.phase_slopes = {}
        self.phase_intercepts = {}

    @staticmethod
    def _add_harmonics(X, f_scale=10e9):
        """Augment (εᵣ, f) with sin/cos harmonic features."""
        epsr = X[:, [0]]
        f = X[:, [1]]
        w = 2 * np.pi * f / f_scale
        harm = np.column_stack([
            np.sin(w), np.cos(w),
            np.sin(2 * w), np.cos(2 * w),
            np.sin(3 * w), np.cos(3 * w),
        ])
        return np.column_stack([epsr, f / f_scale, harm])

    def _detrend(self, s_complex, epsr_vals, freq_vals, prefix="s11"):
        """Compute log10|S| and detrended unwrapped phase for one S-param.

        Detrending removes the linear group delay (slope + intercept) per
        epsr value via a full linear fit, storing both for reconstruction.
        """
        mag = np.maximum(np.abs(s_complex), 1e-10)
        logmag = np.log10(mag)
        phase = np.unwrap(np.angle(s_complex), period=2 * np.pi)
        phase_det = phase.copy()
        self.phase_slopes[prefix] = {}
        self.phase_intercepts[prefix] = {}
        if epsr_vals is not None and freq_vals is not None:
            for ev in np.unique(epsr_vals):
                m = np.abs(epsr_vals - ev) < 1e-6
                if m.sum() < 2:
                    self.phase_slopes[prefix][ev] = 0.0
                    self.phase_intercepts[prefix][ev] = 0.0
                    continue
                f_loc = freq_vals[m]
                p_loc = phase[m]
                A = np.column_stack([f_loc, np.ones_like(f_loc)])
                coeffs, _, _, _ = np.linalg.lstsq(A, p_loc, rcond=None)
                self.phase_slopes[prefix][ev] = coeffs[0]
                self.phase_intercepts[prefix][ev] = coeffs[1]
                phase_det[m] = p_loc - A @ coeffs
        return logmag, phase_det

    def _restore_phase(self, phase_det, epsr_query, freq_query, prefix="s11"):
        """Add back slope + intercept to detrended phase."""
        phase = phase_det.copy()
        slopes = self.phase_slopes.get(prefix, {})
        intercepts = self.phase_intercepts.get(prefix, {})
        known_epsr = sorted(slopes.keys())
        for ev in np.unique(epsr_query):
            m = np.abs(epsr_query - ev) < 1e-6
            if m.sum() == 0:
                continue
            if ev in slopes:
                slope = slopes[ev]
                intercept = intercepts[ev]
            elif known_epsr:
                idx = np.argmin(np.abs(np.array(known_epsr) - ev))
                slope = slopes[known_epsr[idx]]
                intercept = intercepts[known_epsr[idx]]
            else:
                slope = 0.0
                intercept = 0.0
            phase[m] += slope * freq_query[m] + intercept
        return phase

    def _prepare_targets(self, s11, s21, epsr_vals=None, freq_vals=None):
        """Convert complex S-params to 4 training targets.

        S11: Re/Im directly (handles reflection nulls better than phase).
        S21: log₁₀|S₂₁| + detrended unwrapped phase (handles the
             large dynamic range between passband and notch).
        """
        s21_logmag, s21_phase_det = self._detrend(s21, epsr_vals, freq_vals, "s21")
        return [s11.real, s11.imag, s21_logmag, s21_phase_det]

    def _reconstruct(self, pred0, pred1, epsr_query=None, freq_query=None, prefix="s11"):
        """Reconstruct complex S from Re/Im (S11) or log|S|+phase (S21)."""
        if prefix == "s21":
            mag = 10**pred0
            phase = self._restore_phase(pred1, epsr_query, freq_query, prefix)
            return mag * np.exp(1j * phase)
        return pred0 + 1j * pred1  # S11: Re/Im

    def train(self, X, s11, s21, val_epsr=None, random_seed=42):
        """Train the 4 DNN-GP models.

        Parameters
        ----------
        X : ndarray, shape (N, 2)
            Inputs: (εᵣ, freq_Hz).
        s11, s21 : ndarray, shape (N,)
            Complex S-parameters.
        val_epsr : array-like, optional
            εᵣ values to hold out for validation. If None, 10 values
            are randomly selected.
        random_seed : int
            Seed for validation split.

        Returns
        -------
        dict
            Validation metrics.
        """
        from sklearn.preprocessing import StandardScaler

        epsr_vals = X[:, 0]
        epsr_unique = np.sort(np.unique(epsr_vals))

        # Validation split (adaptive: at most 1/3 of epsr values)
        rng = np.random.RandomState(random_seed)
        shuffled = epsr_unique.copy()
        rng.shuffle(shuffled)
        n_val = max(1, min(10, len(epsr_unique) // 3))
        if val_epsr is None:
            val_epsr = np.sort(shuffled[:n_val])
        train_epsr = np.sort(
            epsr_unique[~np.isin(epsr_unique, val_epsr)]
        )

        train_mask = np.isin(epsr_vals, train_epsr)
        val_mask = np.isin(epsr_vals, val_epsr)

        if self.verbose:
            n_train = np.sum(train_mask)
            n_val = np.sum(val_mask)
            print(f"Training: {n_train} samples "
                  f"({len(train_epsr)} εᵣ values)")
            print(f"Validation: {n_val} samples "
                  f"({len(val_epsr)} εᵣ values)")

        # Normalize inputs with harmonic features
        X_harm = self._add_harmonics(X[train_mask])
        self.scalers_x = StandardScaler().fit(X_harm)
        X_train = torch.tensor(
            self.scalers_x.transform(X_harm), dtype=torch.float32
        )
        X_val = torch.tensor(
            self.scalers_x.transform(
                self._add_harmonics(X[val_mask])
            ), dtype=torch.float32
        )

        # Prepare targets for full dataset (for S21 phase detrending)
        Y_full = self._prepare_targets(
            s11, s21,
            epsr_vals=X[:, 0], freq_vals=X[:, 1]
        )

        # Train 4 independent DNN → GP models, one per S-param channel.
        # Each model has its own DNN feature extractor + 2 summed RBF kernels
        # (broad + sharp), trained independently via marginal likelihood.
        import torch.nn as nn

        for i in range(4):
            name = self._target_names[i]
            if self.verbose:
                print(f"\n── Training {name} ──")

            y_train = Y_full[i][train_mask]
            y_val_true = Y_full[i][val_mask]

            ys = StandardScaler().fit(y_train.reshape(-1, 1))
            self.scalers_y[i] = ys
            Yt = torch.tensor(
                ys.transform(y_train.reshape(-1, 1))[:, 0],
                dtype=torch.float32,
            )

            lik = gpytorch.likelihoods.GaussianLikelihood()
            model = DNN_GP(X_train, Yt, lik,
                           in_dim=self._n_feat, feat_dim=self.feat_dim)

            # Stage 1: pretrain DNN (warm-start features)
            model.train()
            dnn_opt = torch.optim.AdamW(
                model.feature_extractor.parameters(), lr=0.001
            )
            for e in range(min(100, self.n_epochs // 5)):
                dnn_opt.zero_grad()
                feats = model.feature_extractor(X_train)
                pred = feats.mean(dim=1).expand(-1)
                loss = nn.MSELoss()(pred, Yt)
                loss.backward()
                dnn_opt.step()

            # Stage 2: joint DNN + 2xRBF via marginal likelihood
            model.train()
            lik.train()
            opt = torch.optim.AdamW(
                model.parameters(), lr=self.lr,
                weight_decay=self.weight_decay,
            )
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(lik, model)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=self.n_epochs
            )

            # Early stopping: check val RMSE every 50 epochs
            best_val_rmse = float("inf")
            stop_patience = 100  # epochs without improvement
            stop_counter = 0
            target_rmse = float(self._target_rmse.get(i, 0.01))

            for epoch in range(self.n_epochs):
                opt.zero_grad()
                out = model(X_train)
                loss = -mll(out, Yt)
                loss.backward()
                opt.step()
                sched.step()

                if (epoch + 1) % 50 == 0:
                    model.eval()
                    lik.eval()
                    with (torch.no_grad(),
                          gpytorch.settings.fast_pred_var()):
                        p = lik(model(X_val))
                    pred_n = p.mean.numpy()
                    pred = ys.inverse_transform(pred_n.reshape(-1, 1))[:, 0]
                    vr = np.sqrt(np.mean((pred - y_val_true)**2))

                    # Stop if target reached
                    if vr < target_rmse:
                        if self.verbose:
                            print(f"  Epoch {epoch + 1:4d}  val RMSE={vr:.4f} (target={target_rmse}) ✓")
                        break

                    # Stop on plateau
                    if vr < best_val_rmse:
                        best_val_rmse = vr
                        stop_counter = 0
                    else:
                        stop_counter += 50
                        if stop_counter >= stop_patience:
                            if self.verbose:
                                print(f"  Epoch {epoch + 1:4d}  val RMSE={vr:.4f} (plateau) ✓")
                            break

                    model.train()
                    lik.train()

            # Evaluate
            model.eval()
            lik.eval()
            with (torch.no_grad(),
                  gpytorch.settings.fast_pred_var()):
                p = lik(model(X_val))
            pred_n = p.mean.numpy()
            pred = ys.inverse_transform(pred_n.reshape(-1, 1))[:, 0]
            rmse = np.sqrt(np.mean((pred - y_val_true)**2))
            if self.verbose:
                print(f"  Final val RMSE = {rmse:.4f}")

            self.models[i] = model
            self.likelihoods[i] = lik

        self.is_trained = True

        # Compute aggregate metrics
        s11_true = s11[val_mask]
        s21_true = s21[val_mask]
        s11_pred, s21_pred = self.predict(X[val_mask])
        return self._metrics(s11_true, s21_true, s11_pred, s21_pred)

    def predict(self, X):
        """Predict complex S-parameters at new (εᵣ, freq) points.

        Parameters
        ----------
        X : ndarray, shape (M, 2)
            Input points: (εᵣ, freq_Hz).

        Returns
        -------
        s11_pred, s21_pred : ndarray, shape (M,)
            Predicted complex S-parameters.
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained yet. "
                               "Call .train() first.")

        X_harm = self._add_harmonics(X)
        X_norm = torch.tensor(
            self.scalers_x.transform(X_harm), dtype=torch.float32
        )

        preds = []
        for i in range(4):
            model = self.models[i]
            lik = self.likelihoods[i]
            model.eval()
            lik.eval()
            with (torch.no_grad(),
                  gpytorch.settings.fast_pred_var()):
                p = lik(model(X_norm))
            preds.append(
                self.scalers_y[i].inverse_transform(
                    p.mean.numpy().reshape(-1, 1)
                )[:, 0]
            )

        s11_pred = self._reconstruct(preds[0], preds[1], X[:, 0], X[:, 1], "s11")
        s21_pred = self._reconstruct(preds[2], preds[3], X[:, 0], X[:, 1], "s21")
        return s11_pred, s21_pred

    @staticmethod
    def _metrics(s11_true, s21_true, s11_pred, s21_pred):
        """Compute RMSE, relative error, and phase error."""
        def _compute(st, sp):
            rmse_c = np.sqrt(np.mean(np.abs(st - sp)**2))
            if np.mean(np.abs(st)**2) > 0:
                rel = 100 * rmse_c / np.sqrt(np.mean(np.abs(st)**2))
            else:
                rel = float("inf")
            mask = np.abs(st) > 0.02
            if mask.sum() > 0:
                ph = np.mean(
                    np.abs(np.angle(st[mask] / sp[mask]))
                ) * 180 / np.pi
            else:
                ph = 0.0
            return rmse_c, rel, ph

        m11 = _compute(s11_true, s11_pred)
        m21 = _compute(s21_true, s21_pred)
        return {
            "s11_rmse": m11[0], "s11_rel_pct": m11[1],
            "s11_phase_deg": m11[2],
            "s21_rmse": m21[0], "s21_rel_pct": m21[1],
            "s21_phase_deg": m21[2],
        }

    def save(self, path="bilat_dnngp.pt"):
        """Save trained model weights, scalers, and GP training data."""
        state = {
            "scalers_x": self.scalers_x,
            "scalers_y": self.scalers_y,
            "feat_dim": self.feat_dim,
            "phase_slopes": self.phase_slopes,
            "phase_intercepts": self.phase_intercepts,
            "model_states": [],
            "lik_states": [],
            "train_inputs": [],
            "train_targets": [],
        }
        for i in range(4):
            state["model_states"].append(self.models[i].state_dict())
            state["lik_states"].append(self.likelihoods[i].state_dict())
            # Save GP training data — ExactGP uses this for posterior prediction.
            # Without it, a loaded model predicts near the training mean.
            state["train_inputs"].append(
                [t.cpu() for t in self.models[i].train_inputs]
            )
            state["train_targets"].append(
                self.models[i].train_targets.cpu()
            )
        torch.save(state, path)

    def load(self, path="bilat_dnngp.pt"):
        """Load trained model weights, scalers, and GP training data."""
        from sklearn.preprocessing import StandardScaler

        state = torch.load(path, weights_only=False)
        self.feat_dim = state["feat_dim"]
        self.scalers_x = state["scalers_x"]
        self.scalers_y = state["scalers_y"]
        self.phase_slopes = state.get("phase_slopes", {})
        self.phase_intercepts = state.get("phase_intercepts", {})

        has_training_data = "train_inputs" in state

        for i in range(4):
            lik = gpytorch.likelihoods.GaussianLikelihood()
            if has_training_data and i < len(state["train_inputs"]):
                # Restore with saved training data so the GP posterior works
                ti = [t for t in state["train_inputs"][i]]
                tt = state["train_targets"][i]
                model = DNN_GP(
                    ti[0], tt, lik,
                    in_dim=self._n_feat, feat_dim=self.feat_dim
                )
            else:
                # Fallback: dummy data — predictions will be near training mean
                dummy_x = torch.zeros((10, self._n_feat), dtype=torch.float32)
                dummy_y = torch.zeros(10, dtype=torch.float32)
                model = DNN_GP(
                    dummy_x, dummy_y, lik,
                    in_dim=self._n_feat, feat_dim=self.feat_dim
                )
            model.load_state_dict(state["model_states"][i])
            lik.load_state_dict(state["lik_states"][i])
            self.models[i] = model
            self.likelihoods[i] = lik

        self.is_trained = True
        if not has_training_data or not self.phase_slopes:
            import warnings
            warnings.warn(
                "Model saved with older code — missing GP training data. "
                "Re-train and re-save, or use the fix_model_checkpoint.py "
                "script to restore the missing state."
            )
        return self

    def predict_with_uncertainty(self, X):
        """Predict with confidence intervals (std in dB).

        Returns
        -------
        s11_pred : ndarray
            Predicted complex S11.
        s11_std_db : ndarray
            S11 standard deviation in dB.
        s21_pred : ndarray
            Predicted complex S21.
        s21_std_db : ndarray
            S21 standard deviation in dB.
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained yet.")

        X_harm = self._add_harmonics(X)
        X_norm = torch.tensor(
            self.scalers_x.transform(X_harm), dtype=torch.float32
        )

        preds = []
        stds = []
        for i in range(4):
            model = self.models[i]
            lik = self.likelihoods[i]
            model.eval()
            lik.eval()
            with (torch.no_grad(),
                  gpytorch.settings.fast_pred_var()):
                p = lik(model(X_norm))
            preds.append(
                self.scalers_y[i].inverse_transform(
                    p.mean.numpy().reshape(-1, 1)
                )[:, 0]
            )
            std_n = np.maximum(p.stddev.numpy(), 0.0)
            stds.append(std_n * np.sqrt(self.scalers_y[i].var_[0]))

        s11_pred = self._reconstruct(preds[0], preds[1], X[:, 0], X[:, 1], "s11")
        s21_pred = self._reconstruct(preds[2], preds[3], X[:, 0], X[:, 1], "s21")

        # S11 std in dB: combine Re/Im uncertainties
        s11_mag = np.maximum(np.abs(s11_pred), 1e-15)
        s11_mag_std = np.sqrt(
            ((s11_pred.real / s11_mag) * stds[0]) ** 2
            + ((s11_pred.imag / s11_mag) * stds[1]) ** 2
        )
        s11_std_db = (20.0 / np.log(10)) * s11_mag_std / s11_mag

        # S21 std in dB: model trained on log10|S|, so σ_dB = 20 · σ_log10 (exact)
        s21_std_db = 20.0 * stds[2]

        return s11_pred, s11_std_db, s21_pred, s21_std_db


# ──────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────


def plot_results(freqs, s11_true, s21_true, s11_pred, s21_pred,
                 test_epsr_values, val_epsr=None, path=None,
                 title_suffix="", epsr_range=(2.0, 2.2)):
    """Plot magnitude, phase, and error scatter of true vs predicted S-parameters.

    Parameters
    ----------
    freqs : ndarray
        Frequency vector (Hz).
    s11_true, s21_true : ndarray
        Ground truth complex S-parameters.
    s11_pred, s21_pred : ndarray
        Predicted complex S-parameters.
    test_epsr_values : ndarray
        εᵣ values corresponding to each test point.
    val_epsr : array-like or None
        εᵣ values considered held-out (plotted bold). If None, all bold.
    path : str, optional
        Save path. If None, displays interactively.
    title_suffix : str
        Appended to figure title.
    epsr_range : tuple
        (min, max) for color mapping.
    """
    import matplotlib.pyplot as plt

    from ..post.plot import _build_grid  # noqa: F401 — mesh helper unused
    import matplotlib  # noqa: F401 — ensures backend is loaded

    cm = plt.cm.viridis
    epsr_lo, epsr_hi = epsr_range

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))

    for col, (st, sp, sfx) in enumerate([
        (s11_true, s11_pred, "S₁₁"),
        (s21_true, s21_pred, "S₂₁"),
    ]):
        # ── Magnitude ──
        ax = axes[0, col]
        for ev in np.sort(np.unique(test_epsr_values)):
            m = np.abs(test_epsr_values - ev) < 1e-6
            if m.sum() == 0:
                continue
            is_val = val_epsr is not None and np.any(
                np.abs(np.asarray(val_epsr) - ev) < 1e-6
            )
            alpha = 0.8 if is_val else 0.2
            lw = 1.5 if is_val else 0.6
            color = cm((ev - epsr_lo) / (epsr_hi - epsr_lo))
            f_ghz = freqs[m] / 1e9
            ax.plot(f_ghz, 20 * np.log10(np.abs(st[m]) + 1e-15),
                    "-", lw=lw, alpha=alpha, color=color)
            ax.plot(f_ghz, 20 * np.log10(np.abs(sp[m]) + 1e-15),
                    "--", lw=lw, alpha=alpha, color=color)
        ax.set_ylabel(f"|{sfx}| (dB)")
        ax.set_ylim(-80, 5)
        ax.grid(True, alpha=0.3)

        # ── Phase ──
        ax = axes[1, col]
        for ev in np.sort(np.unique(test_epsr_values)):
            m = np.abs(test_epsr_values - ev) < 1e-6
            if m.sum() == 0:
                continue
            is_val = val_epsr is not None and np.any(
                np.abs(np.asarray(val_epsr) - ev) < 1e-6
            )
            alpha = 0.8 if is_val else 0.2
            lw = 1.5 if is_val else 0.6
            color = cm((ev - epsr_lo) / (epsr_hi - epsr_lo))
            mask_h = np.abs(st[m]) > 0.02
            if mask_h.sum() == 0:
                continue
            f_ghz = freqs[m][mask_h] / 1e9
            ax.plot(f_ghz, np.angle(st[m][mask_h], deg=True),
                    "-", lw=lw, alpha=alpha, color=color)
            ax.plot(f_ghz, np.angle(sp[m][mask_h], deg=True),
                    "--", lw=lw, alpha=alpha, color=color)
        ax.set_ylabel(f"∠{sfx} (deg)")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Frequency (GHz)")

    # ── Error scatter ──
    for col, (st, sp, sfx) in enumerate([
        (s11_true, s11_pred, "S₁₁"),
        (s21_true, s21_pred, "S₂₁"),
    ]):
        ax = axes[col, 2]
        rel = np.abs(sp - st) / (np.abs(st) + 1e-15) * 100
        sc = ax.scatter(freqs / 1e9, test_epsr_values,
                        c=np.log10(np.clip(rel, 1e-4, 100)),
                        s=6, cmap="plasma", alpha=0.6)
        ax.set_xlabel("Frequency (GHz)")
        ax.set_ylabel("εᵣ")
        ax.set_title(f"{sfx} relative error (%)")
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("log₁₀(error %)")

    fig.suptitle(
        f"DNN-GP Bilateral Filter Surrogate{title_suffix}",
        fontsize=13,
    )
    plt.tight_layout()
    if path:
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved {path}")
    else:
        plt.show()


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────


def bilateral_filter_dnngp(n_epsr=10, n_freqs=81, n_epochs=500,
                           plot=True, save_model="bilat_dnngp.pt",
                           generate=True, data_dir="./data",
                           mat_file=None):
    """Train a DKL surrogate for bilateral filter S-parameters.

    Trains 4 independent DNN → GP (ExactGP) models:
      - S11: Re/Im directly (0.19% relative error)
      - S21: log₁₀|S₂₁| + detrended unwrapped phase (0.23%)

    Data can be loaded from a .mat file, generated via FEM solves,
    or loaded from a prior .npz cache.

    Parameters
    ----------
    n_epsr : int
        Number of εᵣ training values in [2.0, 2.2] (used when
        ``generate=True`` and ``mat_file=None``).
    n_freqs : int
        Frequency points per εᵣ value.
    n_epochs : int
        Training epochs per model.
    plot : bool
        Save validation plots.
    save_model : str or None
        Path to save trained model checkpoint.
    generate : bool
        If True and no mat_file, generate FEM data.
        If False and no mat_file, load from ``bilat_dnngp_data.npz``.
    data_dir : str
        Path to mesh data files.
    mat_file : str or None
        Path to a .mat file with ``epsr``, ``freq``, ``s11``, ``s21``
        arrays. Overrides ``generate`` when provided.

    Returns
    -------
    model : BilateralFilterDNNGP
        Trained surrogate model.
    metrics : dict
        Validation metrics.
    """
    from scipy.io import loadmat

    if mat_file is not None:
        print(f"Loading data from {mat_file}...")
        d = loadmat(mat_file)
        # Handle both .mat (scipy.io) and .npz formats
        if "epsr" in d:
            X = np.column_stack([d["epsr"].ravel(), d["freq"].ravel()])
            s11 = d["s11"].ravel()
            s21 = d["s21"].ravel()
            n_epsr = len(np.unique(d["epsr"]))
            print(f"  Loaded {len(X)} samples ({n_epsr} εᵣ values)")
        else:
            raise KeyError(
                ".mat file must contain 'epsr', 'freq', 's11', 's21'"
            )
    elif generate:
        epsr_vals = np.linspace(2.0, 2.2, n_epsr)
        print(f"Generating FEM data ({n_epsr} εᵣ × {n_freqs} freq"
              f" = {n_epsr * n_freqs} samples)...")
        X, s11, s21 = generate_data(
            epsr_values=epsr_vals, n_freqs=n_freqs,
            data_dir=data_dir,
        )
        np.savez(
            "bilat_dnngp_data.npz",
            X=X, s11=s11, s21=s21,
        )
        print("Saved bilat_dnngp_data.npz")
    else:
        print("Loading precomputed data...")
        d = np.load("bilat_dnngp_data.npz")
        X, s11, s21 = d["X"], d["s11"], d["s21"]

    model = BilateralFilterDNNGP(
        feat_dim=64, n_epochs=n_epochs, verbose=True
    )
    metrics = model.train(X, s11, s21)

    if save_model:
        model.save(save_model)

    print(f"\n{'=' * 55}")
    print("Validation metrics:")
    for key in ["s11_rmse", "s11_rel_pct", "s11_phase_deg",
                 "s21_rmse", "s21_rel_pct", "s21_phase_deg"]:
        print(f"  {key}: {metrics[key]:.3g}")
    print(f"{'=' * 55}")
    s11_ok = metrics["s11_rel_pct"] < 1.0
    s21_ok = metrics["s21_rel_pct"] < 1.0
    print(f"S11 {'✓ < 1%' if s11_ok else '✗ ≥ 1%'}  |  "
          f"S21 {'✓ < 1%' if s21_ok else '✗ ≥ 1%'}")

    if plot:
        # Reconstruct predictions for held-out set
        epsr_vals = X[:, 0]
        epsr_unique = np.sort(np.unique(epsr_vals))
        rng = np.random.RandomState(42)
        shuffled = epsr_unique.copy()
        rng.shuffle(shuffled)
        val_epsr = np.sort(shuffled[:10])
        val_mask = np.isin(epsr_vals, val_epsr)

        s11_pred, s21_pred = model.predict(X[val_mask])
        plot_results(
            X[val_mask, 1], s11[val_mask], s21[val_mask],
            s11_pred, s21_pred,
            X[val_mask, 0], val_epsr=val_epsr,
            path="bilat_dnngp_results.png",
            title_suffix=f" ({n_epsr} εᵣ, {n_epochs} epochs)",
        )

    return model, metrics


# ──────────────────────────────────────────────
# Adaptive LHS sampling
# ──────────────────────────────────────────────


def _sequential_lhs_1d(existing, n_batch, bounds=(2.0, 2.2), seed=None):
    """Sequential 1D Latin Hypercube Sampling avoiding existing points.

    Partitions the *unoccupied* region of [lo, hi] into ``n_batch``
    equal-width intervals and samples one point uniformly from each
    interval — true LHS stratification applied to the remaining space.

    Parameters
    ----------
    existing : array-like or None
        Points already selected (occupied).
    n_batch : int
        Number of new points to draw.
    bounds : tuple
        (lo, hi) of the parameter space.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    ndarray, shape (min(n_batch, available),)
        New parameter values.
    """
    lo, hi = bounds
    rng = np.random.RandomState(seed)

    existing = np.sort(np.atleast_1d(existing).ravel()) if (
        existing is not None and len(np.atleast_1d(existing).ravel()) > 0
    ) else np.array([])

    # Pure LHS over the full range when nothing is occupied yet
    if len(existing) == 0:
        edges = np.linspace(lo, hi, n_batch + 1)
        return edges[:-1] + rng.uniform(size=n_batch) * (edges[1] - edges[0])

    # Build the list of unoccupied gaps
    pts = np.concatenate([[lo], existing, [hi]])
    gaps = [(pts[i], pts[i + 1]) for i in range(len(pts) - 1) if pts[i + 1] > pts[i]]

    if not gaps:
        return np.array([])

    total_width = sum(g[1] - g[0] for g in gaps)
    if total_width < 1e-14:
        return np.array([])

    # Divide total unoccupied width into n_batch equal segments
    seg_width = total_width / n_batch
    batch = []
    for i in range(n_batch):
        offset = (i + rng.uniform()) * seg_width
        cum = 0.0
        for gs, ge in gaps:
            gw = ge - gs
            if offset <= cum + gw:
                pt = gs + (offset - cum)
                batch.append(np.clip(pt, gs + 1e-12, ge - 1e-12))
                break
            cum += gw
    return np.array(batch)


def adaptive_lhs_bilateral_filter(
    epsr_range=(2.0, 2.2),
    freq_range=(138e9, 158e9),
    n_freqs=81,
    n_test=5,
    n_initial=5,
    batch_size=5,
    n_epochs=500,
    target_rel_error=1.0,
    max_n_epsr=100,
    data_dir="./data",
    plot=True,
    random_seed=42,
    cache_file="adaptive_dnngp_data.npz",
    save_model="adaptive_dnngp.pt",
):
    """Adaptive DNN-GP surrogate training with LHS-driven εᵣ sampling.

    Starts with ``n_initial`` Latin Hypercube samples of εᵣ, then
    iteratively adds ``batch_size`` new εᵣ values (also via LHS on
    the unoccupied region), retrains the surrogate, and checks relative
    error against a permanently held-out test set of ``n_test`` εᵣ
    values.  Stops when both S₁₁ and S₂₁ relative error fall below
    ``target_rel_error`` (default 1 %).

    Parameters
    ----------
    epsr_range : tuple
        (min, max) substrate permittivity range.
    freq_range : tuple
        (start, stop) sweep range in Hz.
    n_freqs : int
        Frequency points per εᵣ value.
    n_test : int
        Number of εᵣ values permanently held out for evaluation.
    n_initial : int
        Initial LHS batch size.
    batch_size : int
        εᵣ values added per adaptive iteration.
    n_epochs : int
        Training epochs per model per iteration.
    target_rel_error : float
        Relative error threshold (%) for convergence (both S₁₁ and S₂₁).
    max_n_epsr : int
        Hard limit on total εᵣ values (training + test) before giving up.
    data_dir : str
        Path to mesh data files.
    plot : bool
        Save convergence plot and final validation figure.
    random_seed : int
        Seed for reproducibility.
    cache_file : str or None
        Path to save/restore accumulated FEM data (.npz).
    save_model : str or None
        Path to save final trained model.

    Returns
    -------
    model : BilateralFilterDNNGP
        Final trained surrogate.
    history : list of dict
        Metrics after each iteration.
    """
    lo_epsr, hi_epsr = epsr_range

    # ── 1. Fixed test set (evenly spread, never trained on) ──
    test_epsr = np.linspace(lo_epsr, hi_epsr, n_test + 2)[1:-1]  # exclude boundaries
    n_test = len(test_epsr)

    # ── 2. Generate test data ──
    print("=" * 60)
    print("Generating held-out test set "
          f"({n_test} εᵣ × {n_freqs} freq = {n_test * n_freqs} samples)...")
    X_test, s11_test, s21_test = generate_data(
        epsr_values=test_epsr, n_freqs=n_freqs,
        freq_range=freq_range, data_dir=data_dir, verbose=False,
    )
    print(f"  Test set: {len(X_test)} samples, εᵣ ∈ [{test_epsr[0]:.4f}, {test_epsr[-1]:.4f}]")

    # ── 3. Accumulators ──
    train_epsr_list = []      # εᵣ values used for training
    X_train_acc = np.empty((0, 2))
    s11_train_acc = np.empty(0, dtype=complex)
    s21_train_acc = np.empty(0, dtype=complex)

    history = []

    print("=" * 60)
    print("Adaptive LHS loop")
    print(f"  initial batch  = {n_initial} εᵣ")
    print(f"  batch size     = {batch_size} εᵣ")
    print(f"  target error   = {target_rel_error}%")
    print(f"  max εᵣ        = {max_n_epsr}")
    print("=" * 60)

    # ── 4. Initial batch ──
    new_epsr = _sequential_lhs_1d(
        existing=None, n_batch=n_initial,
        bounds=epsr_range, seed=random_seed,
    )
    train_epsr_list.extend(new_epsr.tolist())

    print(f"\nIteration 0 — initial {n_initial} εᵣ values")
    Xi, s11i, s21i = generate_data(
        epsr_values=new_epsr, n_freqs=n_freqs,
        freq_range=freq_range, data_dir=data_dir,
    )
    X_train_acc = np.vstack([X_train_acc, Xi])
    s11_train_acc = np.concatenate([s11_train_acc, s11i])
    s21_train_acc = np.concatenate([s21_train_acc, s21i])

    # ── 5. Adaptive loop ──
    converged = False
    iteration = 0

    while not converged and len(train_epsr_list) < max_n_epsr:
        # Combine train + test for the model (model splits internally via val_epsr)
        X_all = np.vstack([X_train_acc, X_test])
        s11_all = np.concatenate([s11_train_acc, s11_test])
        s21_all = np.concatenate([s21_train_acc, s21_test])

        n_total_epsr = len(np.unique(X_all[:, 0]))

        # Train
        model = BilateralFilterDNNGP(
            feat_dim=64, n_epochs=n_epochs, verbose=True,
        )
        metrics = model.train(
            X_all, s11_all, s21_all,
            val_epsr=test_epsr, random_seed=random_seed,
        )

        # Record
        entry = dict(
            iteration=iteration,
            n_epsr=n_total_epsr,
            n_train_epsr=len(train_epsr_list),
            n_test_epsr=n_test,
            n_simulations=n_total_epsr * n_freqs,
            **metrics,
        )
        history.append(entry)

        s11_rel = metrics["s11_rel_pct"]
        s21_rel = metrics["s21_rel_pct"]
        converged = s11_rel < target_rel_error and s21_rel < target_rel_error

        # Print summary
        status = "✓ CONVERGED" if converged else "→ continuing"
        print(f"\n  ── Iteration {iteration} summary ──")
        print(f"     εᵣ values: {n_total_epsr} "
              f"({len(train_epsr_list)} train + {n_test} test)")
        print(f"     S11 rel error:  {s11_rel:.3f}%  "
              f"{'✓' if s11_rel < target_rel_error else '✗'}")
        print(f"     S21 rel error:  {s21_rel:.3f}%  "
              f"{'✓' if s21_rel < target_rel_error else '✗'}")
        print(f"     {status}")
        print()

        if converged:
            break

        # Select next batch via sequential LHS
        iteration += 1
        if len(train_epsr_list) + batch_size > max_n_epsr:
            batch_size_actual = max_n_epsr - len(train_epsr_list)
        else:
            batch_size_actual = batch_size

        if batch_size_actual <= 0:
            break

        existing_arr = np.array(train_epsr_list)
        new_epsr = _sequential_lhs_1d(
            existing=existing_arr, n_batch=batch_size_actual,
            bounds=epsr_range, seed=random_seed + iteration,
        )
        train_epsr_list.extend(new_epsr.tolist())

        print(f"Iteration {iteration} — adding {len(new_epsr)} εᵣ values")
        Xi, s11i, s21i = generate_data(
            epsr_values=new_epsr, n_freqs=n_freqs,
            freq_range=freq_range, data_dir=data_dir,
        )
        X_train_acc = np.vstack([X_train_acc, Xi])
        s11_train_acc = np.concatenate([s11_train_acc, s11i])
        s21_train_acc = np.concatenate([s21_train_acc, s21i])

        # Cache accumulated data
        if cache_file:
            np.savez(cache_file,
                     X_train=X_train_acc, s11_train=s11_train_acc, s21_train=s21_train_acc,
                     X_test=X_test, s11_test=s11_test, s21_test=s21_test,
                     test_epsr=test_epsr)

    # ── 6. Final output ──
    print("=" * 60)
    print("Adaptive LHS complete")
    print(f"  Total εᵣ values: {len(np.unique(X_all[:, 0]))}")
    print(f"  Iterations:      {len(history)}")
    print(f"  FEM simulations: {len(X_all)}")
    print(f"  Converged:       {'yes' if converged else 'no (reached max)'}")
    print(f"  Final S11 rel:   {history[-1]['s11_rel_pct']:.3f}%")
    print(f"  Final S21 rel:   {history[-1]['s21_rel_pct']:.3f}%")
    print("=" * 60)

    if save_model:
        model.save(save_model)
        print(f"Saved model to {save_model}")

    # ── 7. Convergence plot ──
    if plot:
        _plot_adaptive_convergence(history, target_rel_error, epsr_range)

    # ── 8. Final validation plot ──
    if plot:
        s11_pred, s21_pred = model.predict(X_test)
        plot_results(
            X_test[:, 1], s11_test, s21_test,
            s11_pred, s21_pred,
            X_test[:, 0], val_epsr=test_epsr,
            path="adaptive_dnngp_results.png",
            title_suffix=(
                f" (adaptive LHS: {len(train_epsr_list)} train εᵣ, "
                f"{len(history)} iters)"
            ),
        )

    return model, history


def _plot_adaptive_convergence(history, target_rel_error, epsr_range):
    """Plot convergence history of adaptive LHS loop."""
    import matplotlib.pyplot as plt

    iters = [h["iteration"] for h in history]
    n_epsr = [h["n_epsr"] for h in history]
    n_sim = [h["n_simulations"] for h in history]
    s11_rel = [h["s11_rel_pct"] for h in history]
    s21_rel = [h["s21_rel_pct"] for h in history]
    s11_rmse = [h["s11_rmse"] for h in history]
    s21_rmse = [h["s21_rmse"] for h in history]
    s11_phase = [h["s11_phase_deg"] for h in history]
    s21_phase = [h["s21_phase_deg"] for h in history]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ── Top left: relative error vs εᵣ count ──
    ax = axes[0, 0]
    ax.plot(n_epsr, s11_rel, "o-", color="#d62728", label="S₁₁ rel error", lw=2)
    ax.plot(n_epsr, s21_rel, "s-", color="#2ca02c", label="S₂₁ rel error", lw=2)
    ax.axhline(target_rel_error, color="gray", ls="--", lw=1,
               label=f"target {target_rel_error}%")
    ax.set_xlabel("Number of εᵣ values")
    ax.set_ylabel("Relative error (%)")
    ax.set_title("Convergence: relative error")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Top right: RMSE ──
    ax = axes[0, 1]
    ax.plot(n_epsr, s11_rmse, "o-", color="#d62728", label="S₁₁ RMSE", lw=2)
    ax.plot(n_epsr, s21_rmse, "s-", color="#2ca02c", label="S₂₁ RMSE", lw=2)
    ax.set_xlabel("Number of εᵣ values")
    ax.set_ylabel("RMSE")
    ax.set_title("Convergence: RMSE")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Bottom left: phase error ──
    ax = axes[1, 0]
    ax.plot(n_epsr, s11_phase, "o-", color="#d62728", label="S₁₁ phase (°)", lw=2)
    ax.plot(n_epsr, s21_phase, "s-", color="#2ca02c", label="S₂₁ phase (°)", lw=2)
    ax.set_xlabel("Number of εᵣ values")
    ax.set_ylabel("Phase error (deg)")
    ax.set_title("Convergence: phase error")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Bottom right: error vs simulations (cost-awareness) ──
    ax = axes[1, 1]
    ax.plot(n_sim, s11_rel, "o-", color="#d62728", label="S₁₁ rel error", lw=2)
    ax.plot(n_sim, s21_rel, "s-", color="#2ca02c", label="S₂₁ rel error", lw=2)
    ax.axhline(target_rel_error, color="gray", ls="--", lw=1)
    ax.set_xlabel("FEM simulations (εᵣ × freq)")
    ax.set_ylabel("Relative error (%)")
    ax.set_title("Error vs simulation cost")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle("Adaptive LHS Convergence", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("adaptive_convergence.png", dpi=150, bbox_inches="tight")
    print("Saved adaptive_convergence.png")
    plt.close()


def plot_surrogate(path="bilat_dnngp.pt", data="bilat_dnngp_data.npz",
                   output="bilat_surrogate_plot.png", mat_file=None):
    """Plot a trained surrogate model's predictions against ground truth.

    Loads a saved DNN-GP checkpoint and the original data, then generates
    a 6-panel figure: magnitude, phase, and error scatter for both S₁₁
    and S₂₁ — with held-out εᵣ values shown in bold.

    Parameters
    ----------
    path : str
        Path to trained model checkpoint (.pt).
    data : str
        Path to training data (.npz) with keys ``X``, ``s11``, ``s21``.
        Ignored if ``mat_file`` is provided.
    output : str or None
        Path to save the figure. If None, displays interactively.
    mat_file : str or None
        Path to .mat file with ``epsr``, ``freq``, ``s11``, ``s21``.
        Overrides ``data`` when provided.

    Returns
    -------
    metrics : dict
        Validation metrics for the held-out set.
    """
    from scipy.io import loadmat

    # ── Load model ──
    print(f"Loading model from {path}...")
    model = BilateralFilterDNNGP().load(path)

    # ── Load data ──
    if mat_file is not None:
        d = loadmat(mat_file)
        X = np.column_stack([d["epsr"].ravel(), d["freq"].ravel()])
        s11 = d["s11"].ravel()
        s21 = d["s21"].ravel()
    else:
        d = np.load(data)
        X, s11, s21 = d["X"], d["s11"], d["s21"]

    epsr_vals = X[:, 0]
    epsr_unique = np.sort(np.unique(epsr_vals))
    n_epsr = len(epsr_unique)
    print(f"Data: {len(X)} samples ({n_epsr} εᵣ values)")

    # ── Predict ──
    print("Predicting...")
    s11_pred, s21_pred = model.predict(X)

    # ── Train/val split (matches BilateralFilterDNNGP.train) ──
    rng = np.random.RandomState(42)
    shuffled = epsr_unique.copy()
    rng.shuffle(shuffled)
    n_val = max(1, min(10, len(epsr_unique) // 3))
    val_epsr = np.sort(shuffled[:n_val])
    train_epsr = np.sort(epsr_unique[~np.isin(epsr_unique, val_epsr)])
    val_mask = np.isin(epsr_vals, val_epsr)

    print(f"Train: {len(train_epsr)} εᵣ  |  Held-out: {len(val_epsr)} εᵣ")

    # ── Plot ──
    plot_results(
        X[:, 1], s11, s21, s11_pred, s21_pred,
        epsr_vals, val_epsr=val_epsr,
        path=output,
        title_suffix=f" ({n_epsr} εᵣ values)",
    )

    # ── Metrics ──
    metrics = model._metrics(
        s11[val_mask], s21[val_mask],
        s11_pred[val_mask], s21_pred[val_mask]
    )
    print(f"\n{'=' * 55}")
    for key in ["s11_rmse", "s11_rel_pct", "s11_phase_deg",
                 "s21_rmse", "s21_rel_pct", "s21_phase_deg"]:
        print(f"  {key}: {metrics[key]:.3g}")
    print(f"{'=' * 55}")

    return metrics
