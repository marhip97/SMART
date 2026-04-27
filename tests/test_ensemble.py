"""Unit tests for the ensemble cross-check layer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.base import EvaluationResult, ForecastResult
from src.ensemble.disagreement import (
    DisagreementReport,
    compute_disagreement,
    disagreement_to_dataframe,
)
from src.ensemble.forecaster import EnsembleForecaster


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _make_forecast(model_id: str, q10: float, q50: float, q90: float,
                   horizon: int = 3, variable_id: str = "kpi") -> ForecastResult:
    dates = pd.date_range("2025-01-01", periods=horizon, freq="YS")
    df = pd.DataFrame({
        "date": dates,
        "q10": [q10] * horizon,
        "q50": [q50] * horizon,
        "q90": [q90] * horizon,
    })
    return ForecastResult(variable_id=variable_id, model_id=model_id, forecasts=df)


def _sample_results(variable_id: str = "kpi") -> list[ForecastResult]:
    return [
        _make_forecast("arima",       q10=1.0, q50=2.0, q90=3.0, variable_id=variable_id),
        _make_forecast("var",         q10=1.5, q50=2.5, q90=3.5, variable_id=variable_id),
        _make_forecast("bvar",        q10=1.2, q50=2.2, q90=3.2, variable_id=variable_id),
        _make_forecast("arx",         q10=0.8, q50=1.8, q90=2.8, variable_id=variable_id),
        _make_forecast("ml_baseline", q10=2.0, q50=5.0, q90=8.0, variable_id=variable_id),  # outlier
    ]


# ── DisagreementReport tests ───────────────────────────────────────────────────

class TestComputeDisagreement:

    def test_returns_one_report_per_horizon(self):
        results = _sample_results()
        reports = compute_disagreement(results)
        assert len(reports) == 3

    def test_spread_is_max_minus_min(self):
        results = _sample_results()
        reports = compute_disagreement(results)
        r = reports[0]
        q50s = list(r.point_forecasts.values())
        expected_spread = max(q50s) - min(q50s)
        assert abs(r.spread - expected_spread) < 1e-9

    def test_outlier_detection(self):
        results = _sample_results()
        reports = compute_disagreement(results, outlier_z_threshold=1.5)
        # ml_baseline has q50=5.0 vs others ~2.0-2.5 → should be flagged
        r = reports[0]
        assert "ml_baseline" in r.outlier_models

    def test_no_outliers_when_all_agree(self):
        results = [
            _make_forecast("arima", q10=1.8, q50=2.0, q90=2.2),
            _make_forecast("var",   q10=1.9, q50=2.1, q90=2.3),
            _make_forecast("bvar",  q10=1.7, q50=1.9, q90=2.1),
        ]
        reports = compute_disagreement(results)
        assert reports[0].outlier_models == []

    def test_high_disagreement_flag(self):
        results = _sample_results()
        reports = compute_disagreement(results)
        assert reports[0].high_disagreement is True

    def test_low_disagreement_flag(self):
        results = [
            _make_forecast("arima", q10=1.9, q50=2.0, q90=2.1),
            _make_forecast("var",   q10=1.8, q50=2.05, q90=2.2),
        ]
        reports = compute_disagreement(results)
        assert reports[0].high_disagreement is False

    def test_ensemble_q50_is_mean_of_q50s(self):
        results = [
            _make_forecast("arima", q10=1.0, q50=2.0, q90=3.0),
            _make_forecast("var",   q10=1.0, q50=4.0, q90=7.0),
        ]
        reports = compute_disagreement(results)
        assert abs(reports[0].ensemble_q50 - 3.0) < 1e-9

    def test_ensemble_q10_is_10th_percentile_of_model_q10s(self):
        """Ensemble q10 is the 10th percentile across model q10s (not the minimum)."""
        results = _sample_results()
        reports = compute_disagreement(results)
        q10s = [float(r.forecasts.iloc[0]["q10"]) for r in results]
        expected = float(np.percentile(q10s, 10))
        assert abs(reports[0].ensemble_q10 - expected) < 1e-9

    def test_empty_input_returns_empty(self):
        assert compute_disagreement([]) == []

    def test_variable_id_carried_through(self):
        results = _sample_results(variable_id="bnp_fastland")
        reports = compute_disagreement(results)
        assert all(r.variable_id == "bnp_fastland" for r in reports)

    def test_z_scores_sum_to_zero(self):
        results = _sample_results()
        reports = compute_disagreement(results)
        z_sum = sum(reports[0].z_scores.values())
        assert abs(z_sum) < 1e-9


class TestDisagreementToDataframe:

    def test_returns_dataframe(self):
        reports = compute_disagreement(_sample_results())
        df = disagreement_to_dataframe(reports)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3

    def test_contains_required_columns(self):
        df = disagreement_to_dataframe(compute_disagreement(_sample_results()))
        for col in ("variable_id", "horizon_year", "spread", "std",
                    "ensemble_q50", "high_disagreement"):
            assert col in df.columns

    def test_per_model_q50_columns(self):
        df = disagreement_to_dataframe(compute_disagreement(_sample_results()))
        assert "q50_arima" in df.columns
        assert "q50_ml_baseline" in df.columns


# ── EnsembleForecaster tests ───────────────────────────────────────────────────

class TestEnsembleForecaster:

    def test_equal_weighting_is_mean(self):
        results = [
            _make_forecast("arima", q10=1.0, q50=2.0, q90=3.0),
            _make_forecast("var",   q10=1.0, q50=4.0, q90=7.0),
        ]
        fc = EnsembleForecaster("kpi", weighting="equal").combine(results)
        assert abs(fc.forecasts["q50"].iloc[0] - 3.0) < 1e-9

    def test_equal_weights_sum_to_one(self):
        results = _sample_results()
        fc = EnsembleForecaster("kpi").combine(results)
        assert abs(sum(fc.weights.values()) - 1.0) < 1e-9

    def test_performance_weighting_favours_low_rmse(self):
        results = [
            _make_forecast("arima", q10=1.0, q50=2.0, q90=3.0),
            _make_forecast("var",   q10=2.0, q50=4.0, q90=6.0),
        ]
        evals = [
            EvaluationResult("kpi", "arima", rmse=0.5, mae=0.4, n_obs=10),
            EvaluationResult("kpi", "var",   rmse=2.0, mae=1.5, n_obs=10),
        ]
        fc = EnsembleForecaster("kpi", weighting="performance", eval_results=evals).combine(results)
        # arima has lower RMSE → higher weight → ensemble q50 closer to 2.0 than 4.0
        assert fc.forecasts["q50"].iloc[0] < 3.0
        assert fc.weights["arima"] > fc.weights["var"]

    def test_trimmed_weighting_ignores_extremes(self):
        results = [
            _make_forecast("m1", q10=1.0, q50=2.0, q90=3.0),
            _make_forecast("m2", q10=1.0, q50=2.1, q90=3.1),
            _make_forecast("m3", q10=1.0, q50=2.2, q90=3.2),
            _make_forecast("m4", q10=1.0, q50=10.0, q90=15.0),  # outlier high
        ]
        fc_equal   = EnsembleForecaster("kpi", weighting="equal").combine(results)
        fc_trimmed = EnsembleForecaster("kpi", weighting="trimmed").combine(results)
        # Trimmed should be lower than equal (outlier dropped)
        assert fc_trimmed.forecasts["q50"].iloc[0] < fc_equal.forecasts["q50"].iloc[0]

    def test_quantile_ordering_preserved(self):
        results = _sample_results()
        fc = EnsembleForecaster("kpi").combine(results)
        assert (fc.forecasts["q10"] <= fc.forecasts["q50"]).all()
        assert (fc.forecasts["q50"] <= fc.forecasts["q90"]).all()

    def test_forecast_has_correct_schema(self):
        fc = EnsembleForecaster("kpi").combine(_sample_results())
        assert set(fc.forecasts.columns) >= {"date", "q10", "q50", "q90"}
        assert len(fc.forecasts) == 3

    def test_disagreement_reports_attached(self):
        fc = EnsembleForecaster("kpi").combine(_sample_results())
        assert len(fc.disagreement) == 3
        assert all(isinstance(r, DisagreementReport) for r in fc.disagreement)

    def test_empty_results_raises(self):
        with pytest.raises(ValueError, match="empty"):
            EnsembleForecaster("kpi").combine([])

    def test_variable_id_mismatch_raises(self):
        results = [
            _make_forecast("arima", q10=1.0, q50=2.0, q90=3.0, variable_id="kpi"),
            _make_forecast("var",   q10=1.0, q50=2.0, q90=3.0, variable_id="bnp_fastland"),
        ]
        with pytest.raises(ValueError, match="mismatch"):
            EnsembleForecaster("kpi").combine(results)

    def test_performance_weights_floor_applied(self):
        results = [
            _make_forecast("arima", q10=1.0, q50=2.0, q90=3.0),
            _make_forecast("var",   q10=1.0, q50=2.0, q90=3.0),
        ]
        evals = [
            EvaluationResult("kpi", "arima", rmse=0.01, mae=0.01, n_obs=10),
            EvaluationResult("kpi", "var",   rmse=100.0, mae=80.0, n_obs=10),
        ]
        fc = EnsembleForecaster("kpi", weighting="performance",
                                eval_results=evals, min_weight=0.1).combine(results)
        assert fc.weights["var"] >= 0.1 - 1e-9
