"""Unit tests for all SMART forecast models on synthetic data.

Tests verify:
- fit() runs without error
- predict() returns a valid ForecastResult with correct schema
- evaluate() returns an EvaluationResult with numeric metrics
- ForecastResult quantile ordering: q10 <= q50 <= q90 (on average)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.base import EvaluationResult, ForecastResult
from src.models.arima import ARIMAModel
from src.models.var import VARModel
from src.models.dfm import DFMModel
from src.models.arx import ARXModel
from src.models.ml_baseline import MLBaselineModel


# ── Synthetic data fixtures ────────────────────────────────────────────────────

def _make_annual_series(n: int = 30, seed: int = 0) -> pd.Series:
    """AR(1) annual series with known structure."""
    rng = np.random.default_rng(seed)
    vals = [2.0]
    for _ in range(n - 1):
        vals.append(0.7 * vals[-1] + rng.normal(0, 0.5))
    idx = pd.date_range("1993-01-01", periods=n, freq="YS")
    return pd.Series(vals, index=idx, name="test_var")


def _make_quarterly_series(n: int = 80, seed: int = 1) -> pd.Series:
    rng = np.random.default_rng(seed)
    vals = [2.0]
    for _ in range(n - 1):
        vals.append(0.6 * vals[-1] + rng.normal(0, 0.4))
    idx = pd.date_range("2003-01-01", periods=n, freq="QS")
    return pd.Series(vals, index=idx, name="test_var")


def _make_exog(y: pd.Series, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {"exog1": rng.normal(0, 1, len(y)), "exog2": rng.normal(1, 0.5, len(y))},
        index=y.index,
    )


def _assert_forecast_result(result: ForecastResult, variable_id: str, model_id: str, horizon: int) -> None:
    assert isinstance(result, ForecastResult)
    assert result.variable_id == variable_id
    assert result.model_id == model_id
    assert set(result.forecasts.columns) >= {"date", "q10", "q50", "q90"}
    assert len(result.forecasts) == horizon
    assert result.forecasts["q10"].le(result.forecasts["q90"]).all(), "q10 must be <= q90"


def _assert_eval_result(result: EvaluationResult, variable_id: str) -> None:
    assert isinstance(result, EvaluationResult)
    assert result.variable_id == variable_id
    assert result.n_obs >= 0
    if result.n_obs > 0:
        assert np.isfinite(result.rmse)
        assert np.isfinite(result.mae)
        assert result.rmse >= 0
        assert result.mae >= 0


# ── ARIMA ──────────────────────────────────────────────────────────────────────

class TestARIMAModel:
    def test_fit_predict_annual(self):
        y = _make_annual_series()
        model = ARIMAModel("test_var", horizon_years=3, max_p=2, max_q=1, max_d=1)
        model.fit(y)
        result = model.predict()
        _assert_forecast_result(result, "test_var", "arima", 3)

    def test_fit_predict_quarterly(self):
        y = _make_quarterly_series()
        model = ARIMAModel("test_var", horizon_years=2, max_p=2, max_q=1, max_d=1)
        model.fit(y)
        result = model.predict()
        _assert_forecast_result(result, "test_var", "arima", 2)

    def test_predict_before_fit_raises(self):
        model = ARIMAModel("test_var")
        with pytest.raises(RuntimeError, match="fit()"):
            model.predict()

    def test_metadata_contains_order(self):
        y = _make_annual_series()
        model = ARIMAModel("test_var", max_p=2, max_q=1, max_d=1)
        model.fit(y)
        result = model.predict()
        assert "order" in result.metadata
        assert "aic" in result.metadata

    def test_evaluate_returns_metrics(self):
        y = _make_annual_series(n=25)
        model = ARIMAModel("test_var", horizon_years=1, max_p=2, max_q=1, max_d=1)
        result = model.evaluate(y, min_train_years=10)
        _assert_eval_result(result, "test_var")

    def test_d_is_capped_at_one(self):
        """T1: max_d > 1 must be silently capped to 1 to avoid divergent forecasts."""
        model = ARIMAModel("test_var", max_d=3)
        assert model.max_d == 1

    def test_short_series_uses_safe_fallback(self):
        """T1: series shorter than SHORT_SERIES_THRESHOLD uses ARIMA(1,1,0) trend='n'."""
        y = _make_annual_series(n=10)
        model = ARIMAModel("test_var", horizon_years=3)
        model.fit(y)
        assert model._result.model.order == (1, 1, 0)

    def test_no_explosive_forecast_short_series(self):
        """T1: short, unstable series must not produce divergent forecasts.

        Reproduces the styringsrente/lønnsvekst failure mode (n=9–11 obs).
        """
        # Trending series that previously caused d=2 to be selected
        idx = pd.date_range("2017-01-01", periods=9, freq="YS")
        y = pd.Series([0.5, 0.5, 0.75, 1.0, 1.5, 4.5, 4.25, 3.75, 3.5], index=idx)
        model = ARIMAModel("styringsrente", horizon_years=3, max_d=2)  # config asks for 2
        model.fit(y)
        result = model.predict()

        hist_max  = float(y.max())
        hist_std  = float(y.std(ddof=1))
        ceiling   = hist_max + 3 * hist_std
        assert (result.forecasts["q50"] <= ceiling).all(), \
            f"q50 above {ceiling:.1f}: {result.forecasts['q50'].tolist()}"
        # Specifically: must not predict 41.9 % (real-world failure mode)
        assert (result.forecasts["q50"] < 15).all(), \
            f"q50 unrealistically high: {result.forecasts['q50'].tolist()}"


# ── VAR ────────────────────────────────────────────────────────────────────────

class TestVARModel:
    def test_fit_predict_single_series(self):
        y = _make_annual_series()
        model = VARModel("test_var", horizon_years=3, max_lags=2)
        model.fit(y)
        result = model.predict()
        _assert_forecast_result(result, "test_var", "var", 3)

    def test_fit_predict_panel(self):
        y = _make_annual_series()
        panel = pd.DataFrame({"test_var": y, "other_var": y * 0.5 + 0.3})
        model = VARModel("test_var", horizon_years=2, max_lags=2)
        model.fit(panel)
        result = model.predict()
        _assert_forecast_result(result, "test_var", "var", 2)

    def test_predict_before_fit_raises(self):
        model = VARModel("test_var")
        with pytest.raises(RuntimeError):
            model.predict()

    def test_insufficient_data_raises(self):
        y = _make_annual_series(n=3)
        model = VARModel("test_var", max_lags=4)
        with pytest.raises(ValueError, match="insufficient"):
            model.fit(y)


# ── DFM ────────────────────────────────────────────────────────────────────────

class TestDFMModel:
    def test_fit_predict_single_series(self):
        y = _make_annual_series()
        model = DFMModel("test_var", horizon_years=2, n_factors=1, factor_order=1, error_order=0)
        model.fit(y)
        result = model.predict()
        _assert_forecast_result(result, "test_var", "dfm", 2)

    def test_fit_predict_with_exog(self):
        y = _make_annual_series()
        X = _make_exog(y)
        model = DFMModel("test_var", horizon_years=2, n_factors=1, factor_order=1, error_order=0)
        model.fit(y, X)
        result = model.predict()
        _assert_forecast_result(result, "test_var", "dfm", 2)

    def test_predict_before_fit_raises(self):
        model = DFMModel("test_var")
        with pytest.raises(RuntimeError):
            model.predict()


# ── AR-X ───────────────────────────────────────────────────────────────────────

class TestARXModel:
    def test_fit_predict_no_exog(self):
        y = _make_annual_series()
        model = ARXModel("test_var", horizon_years=3, max_lags=2)
        model.fit(y)
        result = model.predict()
        _assert_forecast_result(result, "test_var", "arx", 3)

    def test_fit_predict_with_exog(self):
        y = _make_annual_series()
        X = _make_exog(y)
        model = ARXModel("test_var", horizon_years=2, max_lags=2, exog_lags=1)
        model.fit(y, X)
        result = model.predict()
        _assert_forecast_result(result, "test_var", "arx", 2)

    def test_predict_before_fit_raises(self):
        model = ARXModel("test_var")
        with pytest.raises(RuntimeError):
            model.predict()

    def test_metadata_contains_ar_order(self):
        y = _make_annual_series()
        model = ARXModel("test_var", max_lags=2)
        model.fit(y)
        result = model.predict()
        assert "ar_order" in result.metadata

    def test_evaluate_returns_metrics(self):
        y = _make_annual_series(n=25)
        model = ARXModel("test_var", horizon_years=1, max_lags=2)
        result = model.evaluate(y, min_train_years=10)
        _assert_eval_result(result, "test_var")

    def test_clip_extreme_predictions(self):
        """T2/T6: predictions must be clipped to ±10σ from historical mean."""
        y = _make_annual_series(n=20)  # μ ≈ 2, σ ≈ 1
        X = _make_exog(y)
        model = ARXModel("test_var", horizon_years=3, max_lags=2)
        model.fit(y, X)
        result = model.predict()

        mu = float(y.mean())
        sigma = float(y.std(ddof=1))
        upper = mu + 10 * sigma
        lower = mu - 10 * sigma
        assert (result.forecasts["q50"] <= upper).all(), \
            f"q50 above clip ceiling {upper:.1f}: {result.forecasts['q50'].tolist()}"
        assert (result.forecasts["q50"] >= lower).all()
        # Must not produce values like 7672 (real-world boligprisvekst failure)
        assert (result.forecasts["q90"].abs() < 100).all()


# ── ML Baseline ────────────────────────────────────────────────────────────────

class TestMLBaselineModel:
    def test_fit_predict_no_exog(self):
        y = _make_annual_series()
        model = MLBaselineModel("test_var", horizon_years=3, n_estimators=20, lags=[1, 2])
        model.fit(y)
        result = model.predict()
        _assert_forecast_result(result, "test_var", "ml_baseline", 3)

    def test_fit_predict_with_exog(self):
        y = _make_annual_series()
        X = _make_exog(y)
        model = MLBaselineModel("test_var", horizon_years=2, n_estimators=20, lags=[1, 2])
        model.fit(y, X)
        result = model.predict()
        _assert_forecast_result(result, "test_var", "ml_baseline", 2)

    def test_predict_before_fit_raises(self):
        model = MLBaselineModel("test_var")
        with pytest.raises(RuntimeError):
            model.predict()

    def test_feature_importances_in_metadata(self):
        y = _make_annual_series()
        model = MLBaselineModel("test_var", horizon_years=1, n_estimators=20, lags=[1, 2])
        model.fit(y)
        result = model.predict()
        assert "feature_importances" in result.metadata
        assert 1 in result.metadata["feature_importances"]

    def test_evaluate_returns_metrics(self):
        y = _make_annual_series(n=25)
        model = MLBaselineModel("test_var", horizon_years=1, n_estimators=20, lags=[1, 2])
        result = model.evaluate(y, min_train_years=10)
        _assert_eval_result(result, "test_var")

    def test_quarterly_input_resampled_to_annual(self):
        y = _make_quarterly_series()
        model = MLBaselineModel("test_var", horizon_years=2, n_estimators=20, lags=[1, 2])
        model.fit(y)
        result = model.predict()
        assert len(result.forecasts) == 2


# ── ForecastResult validation ──────────────────────────────────────────────────

class TestForecastResult:
    def test_missing_columns_raises(self):
        df = pd.DataFrame({"date": pd.to_datetime(["2025-01-01"]), "q50": [1.0]})
        with pytest.raises(ValueError, match="missing columns"):
            ForecastResult("v", "m", df)

    def test_valid_result(self):
        df = pd.DataFrame({
            "date": pd.to_datetime(["2025-01-01"]),
            "q10": [-0.5], "q50": [1.0], "q90": [2.5],
        })
        r = ForecastResult("v", "m", df)
        assert r.variable_id == "v"
        assert r.model_id == "m"
