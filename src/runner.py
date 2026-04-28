"""SMART model runner – reads latest raw data, runs all applicable models,
produces ensemble forecasts, and writes results to data/processed/forecasts/.

Called after the data pipeline has fetched and stored raw vintages.
"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from src.data.base import PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.ensemble.forecaster import EnsembleForecaster
from src.models.arima import ARIMAModel
from src.models.arx import ARXModel
from src.models.base import EvaluationResult, ForecastResult
from src.models.bvar import BVARModel
from src.models.dfm import DFMModel
from src.models.ml_baseline import MLBaselineModel
from src.models.utils import resample_to_annual
from src.models.var import VARModel

logger = logging.getLogger(__name__)

_MODEL_REGISTRY: dict[str, type] = {
    "ARIMAModel": ARIMAModel,
    "VARModel": VARModel,
    "BVARModel": BVARModel,
    "DFMModel": DFMModel,
    "ARXModel": ARXModel,
    "MLBaselineModel": MLBaselineModel,
}

FORECASTS_DIR = PROCESSED_DATA_DIR / "forecasts"
CONFIG_DIR = Path(__file__).parents[1] / "config"


# ── Config loading ─────────────────────────────────────────────────────────────

def _load_config() -> tuple[list[dict], list[dict]]:
    with open(CONFIG_DIR / "variables.yaml") as f:
        variables = yaml.safe_load(f)["variables"]
    with open(CONFIG_DIR / "models.yaml") as f:
        models = yaml.safe_load(f)["models"]
    return variables, models


# ── Data loading ───────────────────────────────────────────────────────────────

def _load_series(variable_id: str) -> pd.Series | None:
    """Load latest raw vintage for a variable as a pd.Series with DatetimeIndex."""
    raw_dir = RAW_DATA_DIR / variable_id
    if not raw_dir.exists():
        return None
    files = sorted(raw_dir.glob("*.parquet"))
    if not files:
        return None
    df = pd.read_parquet(files[-1])
    if "date" not in df.columns or "value" not in df.columns:
        logger.warning("Parquet for %s missing 'date' or 'value' column.", variable_id)
        return None
    s = pd.Series(
        df["value"].values,
        index=pd.DatetimeIndex(df["date"]),
        name=variable_id,
    )
    return s.dropna().sort_index()


def _build_exog(variables_cfg: list[dict]) -> pd.DataFrame | None:
    """Load all conditioning variables and combine into a single DataFrame."""
    frames: dict[str, pd.Series] = {}
    for var in variables_cfg:
        if var["type"] != "conditioning":
            continue
        s = _load_series(var["id"])
        if s is not None:
            frames[var["id"]] = s
    if not frames:
        return None
    return pd.DataFrame(frames)


# ── Per-variable model execution ───────────────────────────────────────────────

def _applies_to(model_cfg: dict, variable_id: str) -> bool:
    applies_to = model_cfg.get("applies_to", "all")
    if applies_to == "all":
        return True
    return variable_id in applies_to


def _isnan(v: float) -> bool:
    try:
        return math.isnan(v)
    except (TypeError, ValueError):
        return True


def _records(forecasts_df: pd.DataFrame) -> list[dict]:
    return [
        {
            "date": row["date"].strftime("%Y-%m-%d"),
            "q10": round(float(row["q10"]), 4),
            "q50": round(float(row["q50"]), 4),
            "q90": round(float(row["q90"]), 4),
        }
        for _, row in forecasts_df.iterrows()
    ]


def _apply_transform(y_annual: pd.Series, var_cfg: dict) -> pd.Series:
    """Convert annual index-level series to YoY % change when the config requires it."""
    if var_cfg.get("transform") == "yoy_pct" and var_cfg.get("unit") == "index":
        return (y_annual.pct_change() * 100).dropna()
    return y_annual


def run_variable(
    variable_id: str,
    y_raw: pd.Series,
    X_raw: pd.DataFrame | None,
    models_cfg: list[dict],
    var_cfg: dict | None = None,
) -> dict | None:
    """Fit all applicable models for one target variable and return a results dict."""
    y = resample_to_annual(y_raw).dropna()
    y = _apply_transform(y, var_cfg or {})
    X = X_raw.resample("YS").mean() if X_raw is not None else None

    if len(y) < 5:
        logger.warning("%s: only %d annual obs – skipping.", variable_id, len(y))
        return None

    forecast_results: list[ForecastResult] = []
    eval_results: list[EvaluationResult] = []
    model_forecasts: dict[str, list[dict]] = {}
    model_evals: dict[str, EvaluationResult] = {}

    for model_cfg in models_cfg:
        if not _applies_to(model_cfg, variable_id):
            continue

        model_cls = _MODEL_REGISTRY.get(model_cfg["class"])
        if model_cls is None:
            logger.warning("Unknown model class '%s' – skipping.", model_cfg["class"])
            continue

        params = {k: v for k, v in model_cfg.get("params", {}).items()}
        uses_exog = model_cfg.get("uses_exog", False)
        X_fit = X if uses_exog else None
        model_id = model_cfg["id"]

        try:
            model = model_cls(variable_id=variable_id, **params)
            model.fit(y, X_fit)
            fc = model.predict()
            ev = model.evaluate(y, X_fit)
            forecast_results.append(fc)
            eval_results.append(ev)
            model_forecasts[model_id] = _records(fc.forecasts)
            model_evals[model_id] = ev
            logger.info("  ✓ %s / %s", variable_id, model_id)
        except Exception as exc:
            logger.warning("  ✗ %s / %s: %s", variable_id, model_id, exc)

    if not forecast_results:
        logger.warning("%s: no models produced a forecast.", variable_id)
        return None

    ensemble = EnsembleForecaster(
        variable_id=variable_id,
        weighting="performance",
        eval_results=eval_results,
    ).combine(forecast_results)

    disagreement = [
        {
            "horizon_year": dr.horizon_year,
            "spread": round(dr.spread, 4),
            "std": round(dr.std, 4),
            "ensemble_q50": round(dr.ensemble_q50, 4),
            "high_disagreement": dr.high_disagreement,
            "outlier_models": dr.outlier_models,
        }
        for dr in ensemble.disagreement
    ]

    # Historical observations (transformed, annual) – last 20 years max
    history = [
        {"date": ts.strftime("%Y-%m-%d"), "value": round(float(val), 4)}
        for ts, val in y.iloc[-20:].items()
    ]

    evaluation = {
        mid: {
            "rmse":  round(ev.rmse, 4) if not _isnan(ev.rmse) else None,
            "mae":   round(ev.mae, 4) if not _isnan(ev.mae) else None,
            "r2":    round(ev.r2, 4) if not _isnan(ev.r2) else None,
            "n_obs": ev.n_obs,
            "backtest": ev.details.get("backtest", []),
        }
        for mid, ev in model_evals.items()
    }

    return {
        "variable_id": variable_id,
        "run_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "history": history,
        "ensemble": {
            "weighting": ensemble.weighting,
            "forecasts": _records(ensemble.forecasts),
            "weights": {k: round(v, 4) for k, v in ensemble.weights.items()},
        },
        "models": model_forecasts,
        "evaluation": evaluation,
        "disagreement": disagreement,
    }


# ── Top-level pipeline ─────────────────────────────────────────────────────────

def run_all(
    variables_cfg: list[dict],
    models_cfg: list[dict],
) -> dict[str, dict]:
    """Run models for every target variable. Returns results keyed by variable_id."""
    X_all = _build_exog(variables_cfg)

    results: dict[str, dict] = {}
    for var in variables_cfg:
        if var["type"] != "target":
            continue
        variable_id = var["id"]
        logger.info("── %s", variable_id)
        y = _load_series(variable_id)
        if y is None:
            logger.warning("%s: no raw data found – skipping.", variable_id)
            continue
        result = run_variable(variable_id, y, X_all, models_cfg, var_cfg=var)
        if result:
            results[variable_id] = result

    return results


def save_results(results: dict[str, dict]) -> None:
    """Write per-variable JSON forecasts and a manifest file."""
    FORECASTS_DIR.mkdir(parents=True, exist_ok=True)
    run_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    for variable_id, data in results.items():
        var_dir = FORECASTS_DIR / variable_id
        var_dir.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(data, indent=2, ensure_ascii=False)
        (var_dir / "latest.json").write_text(payload)
        (var_dir / f"{run_date}.json").write_text(payload)
        logger.info("Saved %s", variable_id)

    manifest = {
        "run_date": run_date,
        "variables": sorted(results.keys()),
    }
    (FORECASTS_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False)
    )
    logger.info("manifest.json written – %d variables", len(results))


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    variables_cfg, models_cfg = _load_config()
    results = run_all(variables_cfg, models_cfg)
    save_results(results)
    logger.info("Runner complete: %d / %d target variables processed.",
                len(results),
                sum(1 for v in variables_cfg if v["type"] == "target"))


if __name__ == "__main__":
    main()
