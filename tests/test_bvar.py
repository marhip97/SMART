"""Unit tests for BVARModel on synthetic data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.base import ForecastResult, EvaluationResult
from src.models.bvar import (
    BVARModel,
    _build_companion_matrices,
    _minnesota_prior_precision,
    _simulate_forward,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _make_annual_series(n: int = 30, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    vals = [2.0]
    for _ in range(n - 1):
        vals.append(0.7 * vals[-1] + rng.normal(0, 0.5))
    idx = pd.date_range("1993-01-01", periods=n, freq="YS")
    return pd.Series(vals, index=idx, name="test_var")


def _make_panel(n: int = 30, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y1 = _make_annual_series(n, seed)
    y2 = pd.Series(
        0.5 * y1.values + rng.normal(0, 0.3, n),
        index=y1.index,
        name="other_var",
    )
    return pd.DataFrame({"test_var": y1, "other_var": y2})


# ── Helper function tests ──────────────────────────────────────────────────────

def test_build_companion_matrices_shape():
    data = np.random.default_rng(0).normal(size=(20, 3))
    Y, X = _build_companion_matrices(data, n_lags=2)
    assert Y.shape == (18, 3)          # T - n_lags rows
    assert X.shape == (18, 2 * 3 + 1)  # p*n + 1 columns


def test_build_companion_matrices_constant():
    data = np.ones((10, 2))
    _, X = _build_companion_matrices(data, n_lags=1)
    assert (X[:, -1] == 1.0).all()


def test_minnesota_prior_precision_shape():
    n, p = 3, 2
    sigma = np.array([0.5, 0.3, 0.4])
    k = n * p + 1
    V0_inv = _minnesota_prior_precision(n, p, sigma, lambda1=0.2, lambda2=0.5)
    assert V0_inv.shape == (k * n, k * n)
    # Diagonal elements should be non-negative
    assert (np.diag(V0_inv) >= 0).all()


def test_minnesota_own_lags_larger_variance_than_cross():
    """Minnesota prior: own lags have LARGER variance (smaller precision) than cross lags.

    λ2 < 1 → cross-variable prior variance = (λ1*λ2/l)² < own variance = (λ1/l)²
    Equivalently, cross-lag precision > own-lag precision.
    This encodes the belief that own-variable dynamics are more informative.
    """
    n, p = 2, 1
    sigma = np.ones(n)
    V0_inv = _minnesota_prior_precision(n, p, sigma, lambda1=0.2, lambda2=0.5)
    # Equation 0, own lag (var=0, lag=1): global index 0
    # Equation 0, cross lag (var=1, lag=1): global index 1
    own_prec = V0_inv[0, 0]
    cross_prec = V0_inv[1, 1]
    # Own lag has LOWER precision (higher variance) than cross lag when λ2 < 1
    assert own_prec <= cross_prec


def test_simulate_forward_shape():
    history = np.random.default_rng(0).normal(size=(20, 2))
    n_lags, steps, n = 2, 5, 2
    k = n_lags * n + 1
    B = np.zeros((k, n))
    result = _simulate_forward(history, B, n_lags=n_lags, steps=steps)
    assert result.shape == (steps, n)


def test_simulate_forward_zero_coefficients_gives_constant():
    """With B=0 (except constant=c), forecast should converge to c."""
    history = np.ones((10, 1))
    n_lags, n = 2, 1
    k = n_lags * n + 1
    B = np.zeros((k, n))
    B[-1, 0] = 3.0  # constant term
    result = _simulate_forward(history, B, n_lags=n_lags, steps=4)
    assert result.shape == (4, 1)
    # After many steps from zero autoregression, converges to constant
    assert abs(result[-1, 0] - 3.0) < 1e-9


# ── BVARModel tests ────────────────────────────────────────────────────────────

class TestBVARModel:

    def test_fit_predict_single_series(self):
        y = _make_annual_series()
        model = BVARModel("test_var", horizon_years=3, n_lags=1, n_draws=50)
        model.fit(y)
        result = model.predict()
        assert isinstance(result, ForecastResult)
        assert result.variable_id == "test_var"
        assert result.model_id == "bvar"
        assert set(result.forecasts.columns) >= {"date", "q10", "q50", "q90"}
        assert len(result.forecasts) == 3

    def test_fit_predict_panel(self):
        panel = _make_panel()
        model = BVARModel("test_var", horizon_years=2, n_lags=1, n_draws=50)
        model.fit(panel)
        result = model.predict()
        assert len(result.forecasts) == 2

    def test_quantile_ordering(self):
        """q10 <= q50 <= q90 for all forecast years."""
        panel = _make_panel(n=30)
        model = BVARModel("test_var", horizon_years=3, n_lags=1, n_draws=100)
        model.fit(panel)
        result = model.predict()
        df = result.forecasts
        assert (df["q10"] <= df["q50"]).all(), "q10 must be <= q50"
        assert (df["q50"] <= df["q90"]).all(), "q50 must be <= q90"

    def test_predict_before_fit_raises(self):
        model = BVARModel("test_var")
        with pytest.raises(RuntimeError, match="fit()"):
            model.predict()

    def test_metadata_contains_hyperparams(self):
        y = _make_annual_series()
        model = BVARModel("test_var", n_lags=2, lambda1=0.1, n_draws=50)
        model.fit(y)
        result = model.predict()
        assert result.metadata["n_lags"] == 2
        assert result.metadata["lambda1"] == 0.1
        assert result.metadata["n_draws"] == 50

    def test_tighter_prior_reduces_forecast_variance(self):
        """Tighter lambda1 should give narrower fan chart."""
        panel = _make_panel(n=30)

        model_tight = BVARModel("test_var", n_lags=1, lambda1=0.05, n_draws=200, random_state=0)
        model_loose = BVARModel("test_var", n_lags=1, lambda1=0.5, n_draws=200, random_state=0)

        model_tight.fit(panel)
        model_loose.fit(panel)

        tight_spread = (model_tight.predict().forecasts["q90"] - model_tight.predict().forecasts["q10"]).mean()
        loose_spread = (model_loose.predict().forecasts["q90"] - model_loose.predict().forecasts["q10"]).mean()

        assert tight_spread <= loose_spread, "Tighter prior should give narrower intervals"

    def test_evaluate_returns_metrics(self):
        y = _make_annual_series(n=25)
        model = BVARModel("test_var", horizon_years=1, n_lags=1, n_draws=30)
        result = model.evaluate(y, min_train_years=10)
        assert isinstance(result, EvaluationResult)
        assert result.variable_id == "test_var"
        if result.n_obs > 0:
            assert np.isfinite(result.rmse)
            assert result.rmse >= 0

    def test_reproducible_with_same_seed(self):
        panel = _make_panel()
        m1 = BVARModel("test_var", n_lags=1, n_draws=50, random_state=7)
        m2 = BVARModel("test_var", n_lags=1, n_draws=50, random_state=7)
        m1.fit(panel)
        m2.fit(panel)
        r1 = m1.predict().forecasts
        r2 = m2.predict().forecasts
        pd.testing.assert_frame_equal(r1, r2)

    def test_quarterly_series(self):
        rng = np.random.default_rng(5)
        idx = pd.date_range("2000-01-01", periods=60, freq="QS")
        y = pd.Series(rng.normal(2, 0.5, 60), index=idx, name="test_var")
        model = BVARModel("test_var", horizon_years=2, n_lags=1, n_draws=50)
        model.fit(y)
        result = model.predict()
        assert len(result.forecasts) == 2
