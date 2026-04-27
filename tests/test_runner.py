"""Tests for the model runner (src/runner.py)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.runner import (
    _applies_to,
    _build_exog,
    _load_series,
    _records,
    run_variable,
    save_results,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _annual_series(variable_id: str = "kpi", n: int = 20, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2000-01-01", periods=n, freq="YS")
    return pd.Series(rng.normal(2.0, 0.5, n), index=dates, name=variable_id)


def _raw_parquet(tmp_path: Path, variable_id: str, s: pd.Series) -> Path:
    raw_dir = tmp_path / variable_id
    raw_dir.mkdir(parents=True)
    df = pd.DataFrame({"date": s.index, "value": s.values})
    p = raw_dir / "2026-04-27.parquet"
    df.to_parquet(p, index=False)
    return p


MINIMAL_MODELS_CFG = [
    {
        "id": "arima",
        "class": "ARIMAModel",
        "applies_to": "all",
        "uses_exog": False,
        "params": {"max_p": 2, "max_q": 1, "max_d": 1},
    }
]


# ── _applies_to ────────────────────────────────────────────────────────────────

class TestAppliesTo:

    def test_all_applies_to_any_variable(self):
        assert _applies_to({"applies_to": "all"}, "kpi") is True

    def test_explicit_list_match(self):
        cfg = {"applies_to": ["kpi", "bnp_fastland"]}
        assert _applies_to(cfg, "kpi") is True

    def test_explicit_list_no_match(self):
        cfg = {"applies_to": ["kpi", "bnp_fastland"]}
        assert _applies_to(cfg, "styringsrente") is False

    def test_default_is_all(self):
        assert _applies_to({}, "kpi") is True


# ── _records ───────────────────────────────────────────────────────────────────

class TestRecords:

    def test_output_shape(self):
        df = pd.DataFrame({
            "date": pd.date_range("2027-01-01", periods=3, freq="YS"),
            "q10": [1.0, 1.1, 1.2],
            "q50": [2.0, 2.1, 2.2],
            "q90": [3.0, 3.1, 3.2],
        })
        recs = _records(df)
        assert len(recs) == 3
        assert set(recs[0].keys()) == {"date", "q10", "q50", "q90"}

    def test_date_formatted_as_iso(self):
        df = pd.DataFrame({
            "date": [pd.Timestamp("2027-01-01")],
            "q10": [1.0], "q50": [2.0], "q90": [3.0],
        })
        assert _records(df)[0]["date"] == "2027-01-01"


# ── _load_series ───────────────────────────────────────────────────────────────

class TestLoadSeries:

    def test_returns_none_when_no_data(self, tmp_path):
        with patch("src.runner.RAW_DATA_DIR", tmp_path):
            assert _load_series("missing_var") is None

    def test_loads_latest_parquet(self, tmp_path):
        s = _annual_series("kpi")
        _raw_parquet(tmp_path, "kpi", s)
        with patch("src.runner.RAW_DATA_DIR", tmp_path):
            result = _load_series("kpi")
        assert result is not None
        assert len(result) == len(s)

    def test_series_has_datetime_index(self, tmp_path):
        s = _annual_series("kpi")
        _raw_parquet(tmp_path, "kpi", s)
        with patch("src.runner.RAW_DATA_DIR", tmp_path):
            result = _load_series("kpi")
        assert isinstance(result.index, pd.DatetimeIndex)


# ── _build_exog ────────────────────────────────────────────────────────────────

class TestBuildExog:

    def test_returns_none_when_no_conditioning_data(self, tmp_path):
        variables_cfg = [{"id": "kpi", "type": "target"}]
        with patch("src.runner.RAW_DATA_DIR", tmp_path):
            result = _build_exog(variables_cfg)
        assert result is None

    def test_returns_dataframe_with_conditioning_vars(self, tmp_path):
        s = _annual_series("oljepris")
        _raw_parquet(tmp_path, "oljepris", s)
        variables_cfg = [
            {"id": "kpi", "type": "target"},
            {"id": "oljepris", "type": "conditioning"},
        ]
        with patch("src.runner.RAW_DATA_DIR", tmp_path):
            result = _build_exog(variables_cfg)
        assert result is not None
        assert "oljepris" in result.columns


# ── run_variable ───────────────────────────────────────────────────────────────

class TestRunVariable:

    def test_returns_dict_with_required_keys(self):
        y = _annual_series("kpi", n=20)
        result = run_variable("kpi", y, None, MINIMAL_MODELS_CFG)
        assert result is not None
        assert set(result.keys()) >= {"variable_id", "run_date", "ensemble", "models", "disagreement"}

    def test_ensemble_forecasts_have_three_horizons(self):
        y = _annual_series("kpi", n=20)
        result = run_variable("kpi", y, None, MINIMAL_MODELS_CFG)
        assert len(result["ensemble"]["forecasts"]) == 3

    def test_returns_none_for_insufficient_data(self):
        y = _annual_series("kpi", n=3)
        result = run_variable("kpi", y, None, MINIMAL_MODELS_CFG)
        assert result is None

    def test_quantile_ordering_preserved(self):
        y = _annual_series("kpi", n=20)
        result = run_variable("kpi", y, None, MINIMAL_MODELS_CFG)
        for fc in result["ensemble"]["forecasts"]:
            assert fc["q10"] <= fc["q50"] <= fc["q90"]

    def test_weights_sum_to_one(self):
        y = _annual_series("kpi", n=20)
        result = run_variable("kpi", y, None, MINIMAL_MODELS_CFG)
        assert abs(sum(result["ensemble"]["weights"].values()) - 1.0) < 1e-6

    def test_unknown_model_class_skipped_gracefully(self):
        bad_cfg = [{"id": "ghost", "class": "GhostModel", "applies_to": "all",
                    "uses_exog": False, "params": {}}]
        y = _annual_series("kpi", n=20)
        result = run_variable("kpi", y, None, bad_cfg)
        assert result is None  # no models succeeded


# ── save_results ───────────────────────────────────────────────────────────────

class TestSaveResults:

    def test_writes_latest_json(self, tmp_path):
        results = {
            "kpi": {
                "variable_id": "kpi",
                "run_date": "2026-04-27",
                "ensemble": {"weighting": "performance", "forecasts": [], "weights": {}},
                "models": {},
                "disagreement": [],
            }
        }
        with patch("src.runner.FORECASTS_DIR", tmp_path / "forecasts"):
            save_results(results)
        assert (tmp_path / "forecasts" / "kpi" / "latest.json").exists()

    def test_writes_manifest(self, tmp_path):
        results = {"kpi": {"variable_id": "kpi", "run_date": "2026-04-27",
                           "ensemble": {}, "models": {}, "disagreement": []}}
        with patch("src.runner.FORECASTS_DIR", tmp_path / "forecasts"):
            save_results(results)
        manifest_path = tmp_path / "forecasts" / "manifest.json"
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text())
        assert "kpi" in manifest["variables"]

    def test_archive_copy_written(self, tmp_path):
        results = {"kpi": {"variable_id": "kpi", "run_date": "2026-04-27",
                           "ensemble": {}, "models": {}, "disagreement": []}}
        with patch("src.runner.FORECASTS_DIR", tmp_path / "forecasts"):
            save_results(results)
        archive_files = list((tmp_path / "forecasts" / "kpi").glob("2*.json"))
        assert len(archive_files) == 1
