"""Bayesian VAR (BVAR) with Minnesota prior.

Implements a BVAR(p) using the standard Minnesota prior without external
Bayesian libraries. The posterior mean of the coefficient matrix is used
for point forecasts; Monte Carlo draws from the posterior distribution
produce the quantile fan chart.

Minnesota prior intuition:
  - Own lags are informative (variance shrinks with lag order).
  - Cross-variable lags are less informative (tighter prior).
  - Overall tightness λ1 controls how much we trust the prior over data.

References:
  Litterman (1986), "Forecasting with Bayesian Vector Autoregressions",
  Review of Economics and Statistics.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.linalg import LinAlgError

from .base import BaseModel, EvaluationResult, ForecastResult
from .utils import make_forecast_dates, walk_forward_eval


class BVARModel(BaseModel):
    """Bayesian VAR with Minnesota prior.

    Hyperparameters:
        n_lags:   VAR lag order (fixed; not AIC-selected to keep priors simple).
        lambda1:  Overall tightness. Lower = stronger shrinkage. Default 0.2.
        lambda2:  Cross-variable tightness relative to own lags. Default 0.5.
        n_draws:  Monte Carlo draws from posterior for quantile estimation.
    """

    model_id = "bvar"

    def __init__(
        self,
        variable_id: str,
        horizon_years: int = 3,
        quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
        random_state: int = 42,
        n_lags: int = 2,
        lambda1: float = 0.2,
        lambda2: float = 0.5,
        n_draws: int = 500,
        **kwargs: Any,
    ) -> None:
        super().__init__(variable_id, horizon_years, quantiles, random_state)
        self.n_lags = n_lags
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.n_draws = n_draws

        # Set at fit time
        self._B_post: np.ndarray | None = None       # (k × n) posterior mean
        self._V_post: np.ndarray | None = None       # (k*n × k*n) posterior cov
        self._sigma_ols: np.ndarray | None = None    # (n,) per-variable OLS residual std
        self._Sigma: np.ndarray | None = None        # (n × n) error cov estimate
        self._y_panel: pd.DataFrame | None = None
        self._col_idx: int = 0
        self._n_vars: int = 0
        self._k: int = 0                             # total predictors per equation

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit(self, y: pd.Series | pd.DataFrame, X: pd.DataFrame | None = None) -> "BVARModel":
        """Fit BVAR on a panel of target variables.

        Args:
            y:  Series or DataFrame. Multi-column panel is recommended.
                Single-series input degrades to Bayesian AR.
            X:  Ignored (all variables are endogenous in VAR).
        """
        if isinstance(y, pd.Series):
            panel = y.to_frame(name=self.variable_id)
        else:
            panel = y.copy()

        panel = panel.dropna()
        self._y_panel = panel
        self._n_vars = panel.shape[1]
        self._col_idx = (
            list(panel.columns).index(self.variable_id)
            if self.variable_id in panel.columns
            else 0
        )

        Y, X_mat = _build_companion_matrices(panel.values, self.n_lags)
        n, T = self._n_vars, Y.shape[0]
        k = X_mat.shape[1]  # n_lags * n + 1 (constant)
        self._k = k

        # Step 1: OLS residual std per variable (used to scale Minnesota prior)
        try:
            B_ols = np.linalg.lstsq(X_mat, Y, rcond=None)[0]
            resid = Y - X_mat @ B_ols
            self._sigma_ols = resid.std(axis=0, ddof=k)
        except LinAlgError:
            self._sigma_ols = np.ones(n)

        sigma = np.where(self._sigma_ols > 0, self._sigma_ols, 1.0)

        # Step 2: Build Minnesota prior precision matrix V0_inv (block-diagonal)
        V0_inv = _minnesota_prior_precision(n, self.n_lags, sigma, self.lambda1, self.lambda2)

        # Step 3: Estimate Sigma (error covariance) via OLS residuals
        resid_ols = Y - X_mat @ B_ols if 'B_ols' in dir() else Y
        try:
            self._Sigma = (resid_ols.T @ resid_ols) / max(T - k, 1)
        except Exception:  # noqa: BLE001
            self._Sigma = np.eye(n)

        Sigma_inv = _safe_inv(self._Sigma)

        # Step 4: Posterior mean and covariance (conditional on Sigma)
        # V_post = (V0_inv + Sigma_inv ⊗ X'X)^{-1}
        # b_post = V_post (V0_inv b0 + (Sigma_inv ⊗ X') vec(Y))
        # With b0 = 0 (Minnesota prior mean), this simplifies:
        XtX = X_mat.T @ X_mat  # (k × k)
        SiXtX = np.kron(Sigma_inv, XtX)  # (k*n × k*n)

        V_post_inv = V0_inv + SiXtX
        self._V_post = _safe_inv(V_post_inv)

        # Right-hand side: (Sigma_inv ⊗ X') vec(Y) = vec(X' Y Sigma_inv)
        rhs = np.kron(Sigma_inv, X_mat.T) @ Y.T.ravel(order="F")
        b_post = self._V_post @ rhs

        # Reshape to (k × n)
        self._B_post = b_post.reshape(k, n, order="F")

        self._is_fitted = True
        return self

    # ── Predict ───────────────────────────────────────────────────────────────

    def predict(self) -> ForecastResult:
        self._require_fitted()
        assert self._y_panel is not None and self._B_post is not None

        rng = np.random.default_rng(self.random_state)

        freq = pd.infer_freq(self._y_panel.index)
        steps_per_year = _steps_per_year(freq)
        steps = self.horizon_years * steps_per_year
        n = self._n_vars

        # Draw B matrices from the posterior: vec(B) ~ N(b_post, V_post)
        b_post_vec = self._B_post.ravel(order="F")
        try:
            draws = rng.multivariate_normal(b_post_vec, self._V_post, size=self.n_draws)
        except LinAlgError:
            # Fall back to diagonal approximation if V_post is ill-conditioned
            draws = rng.normal(b_post_vec, np.sqrt(np.abs(np.diag(self._V_post))),
                               size=(self.n_draws, len(b_post_vec)))

        # For each draw, simulate h steps ahead
        Y_vals = self._y_panel.values  # (T × n)
        all_forecasts = np.zeros((self.n_draws, steps, n))

        for d in range(self.n_draws):
            B_draw = draws[d].reshape(self._k, n, order="F")
            all_forecasts[d] = _simulate_forward(Y_vals, B_draw, self.n_lags, steps, self._Sigma, rng)

        # Aggregate to annual growth rates and extract quantiles
        col_idx = self._col_idx
        rows = []
        for yr in range(1, self.horizon_years + 1):
            sl = slice((yr - 1) * steps_per_year, yr * steps_per_year)
            annual_means = all_forecasts[:, sl, col_idx].mean(axis=1)  # (n_draws,)
            rows.append({
                "date": make_forecast_dates(self._y_panel.index[-1], yr),
                "q10": float(np.percentile(annual_means, 10)),
                "q50": float(np.percentile(annual_means, 50)),
                "q90": float(np.percentile(annual_means, 90)),
            })

        return ForecastResult(
            variable_id=self.variable_id,
            model_id=self.model_id,
            forecasts=pd.DataFrame(rows),
            metadata={
                "n_lags": self.n_lags,
                "lambda1": self.lambda1,
                "lambda2": self.lambda2,
                "n_draws": self.n_draws,
                "n_vars": n,
            },
        )

    # ── Evaluate ──────────────────────────────────────────────────────────────

    def evaluate(
        self,
        y: pd.Series,
        X: pd.DataFrame | None = None,
        min_train_years: int = 10,
    ) -> EvaluationResult:
        return walk_forward_eval(self, y, X, min_train_years)


# ── Helper functions ───────────────────────────────────────────────────────────

def _build_companion_matrices(
    data: np.ndarray, n_lags: int
) -> tuple[np.ndarray, np.ndarray]:
    """Build Y and X matrices for VAR(p) estimation.

    Y: (T-p × n) matrix of observations.
    X: (T-p × p*n+1) matrix of lagged values plus a constant column.
    """
    T, n = data.shape
    rows = T - n_lags
    Y = data[n_lags:]  # (rows × n)
    X = np.zeros((rows, n_lags * n + 1))
    for lag in range(1, n_lags + 1):
        X[:, (lag - 1) * n : lag * n] = data[n_lags - lag : T - lag]
    X[:, -1] = 1.0  # constant
    return Y, X


def _minnesota_prior_precision(
    n: int, n_lags: int, sigma: np.ndarray, lambda1: float, lambda2: float
) -> np.ndarray:
    """Build the block-diagonal Minnesota prior precision matrix V0_inv.

    Prior variance for coefficient on variable j at lag l in equation i:
      - Own lag (i==j): (λ1 / l)²
      - Cross lag (i≠j): (λ1 * λ2 / l)² * (σi² / σj²)
      - Constant: very large variance (diffuse prior → small precision)
    """
    k = n_lags * n + 1  # predictors per equation
    total = k * n
    V0_inv = np.zeros((total, total))

    for eq in range(n):          # equation index
        for lag in range(1, n_lags + 1):
            for var in range(n):  # predictor variable index
                coef_idx = (lag - 1) * n + var   # position within one equation's block
                global_idx = eq * k + coef_idx

                if var == eq:
                    prior_var = (lambda1 / lag) ** 2
                else:
                    prior_var = (lambda1 * lambda2 / lag) ** 2 * (sigma[eq] ** 2 / sigma[var] ** 2)

                V0_inv[global_idx, global_idx] = 1.0 / max(prior_var, 1e-10)

        # Constant: near-zero precision (diffuse)
        const_idx = eq * k + (k - 1)
        V0_inv[const_idx, const_idx] = 1e-4

    return V0_inv


def _safe_inv(M: np.ndarray) -> np.ndarray:
    """Invert a matrix with fallback to pseudo-inverse if singular."""
    try:
        return np.linalg.inv(M)
    except LinAlgError:
        return np.linalg.pinv(M)


def _simulate_forward(
    history: np.ndarray,
    B: np.ndarray,
    n_lags: int,
    steps: int,
    Sigma: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """H-step-ahead simulation given coefficient matrix B.

    Args:
        history: (T × n) historical values.
        B:       (k × n) coefficient matrix (lagged vars + constant).
        n_lags:  Number of lags in the VAR.
        steps:   Number of steps to forecast.
        Sigma:   (n × n) error covariance. When provided together with rng,
                 innovations ε_t ~ N(0, Σ) are drawn at each step so that the
                 resulting fan chart reflects both coefficient and innovation
                 uncertainty.
        rng:     Seeded random generator used for innovation draws.

    Returns:
        (steps × n) array of simulated values.
    """
    n = history.shape[1]
    buf = list(history[-n_lags:])   # rolling window of last n_lags observations
    forecasts = np.zeros((steps, n))

    for t in range(steps):
        x = np.concatenate([buf[-(lag)] for lag in range(1, n_lags + 1)] + [np.ones(1)])
        y_hat = x @ B
        if Sigma is not None and rng is not None:
            y_hat = y_hat + rng.multivariate_normal(np.zeros(n), Sigma)
        forecasts[t] = y_hat
        buf.append(y_hat)

    return forecasts


def _steps_per_year(freq: str | None) -> int:
    if freq is None:
        return 4
    freq = freq.upper()
    if freq in ("A", "AS", "A-DEC", "YE", "YS", "Y"):
        return 1
    if freq in ("Q", "QS", "Q-DEC", "QE", "QS-OCT"):
        return 4
    if freq in ("M", "MS", "ME"):
        return 12
    return 4
