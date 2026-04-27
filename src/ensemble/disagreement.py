"""Cross-model disagreement metrics – the core of SMART's cross-check layer.

When models disagree substantially, that disagreement is itself informative:
it signals that the forecast is uncertain or model-dependent, and should be
interpreted with extra caution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.models.base import ForecastResult


@dataclass
class DisagreementReport:
    """Disagreement statistics across models for one variable and horizon.

    Attributes:
        variable_id:    Target variable.
        horizon_year:   Forecast year (1, 2, or 3).
        forecast_date:  First day of the forecast year.
        model_ids:      Ordered list of model IDs included.
        point_forecasts: Dict mapping model_id → q50 point forecast.
        spread:         Max q50 – Min q50 across models (absolute disagreement).
        std:            Standard deviation of q50s across models.
        iqr:            Interquartile range of q50s across models (Q75–Q25).
        z_scores:       Dict mapping model_id → normalised distance from ensemble mean.
        outlier_models: Models whose |z-score| > threshold (default 1.5).
        ensemble_q50:   Equal-weight mean of all model q50s.
        ensemble_q10:   10th percentile of model q10s (conservative lower bound).
        ensemble_q90:   90th percentile of model q90s (conservative upper bound).
    """

    variable_id: str
    horizon_year: int
    forecast_date: pd.Timestamp
    model_ids: list[str]
    point_forecasts: dict[str, float]
    spread: float
    std: float
    iqr: float
    z_scores: dict[str, float]
    outlier_models: list[str]
    ensemble_q50: float
    ensemble_q10: float
    ensemble_q90: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def high_disagreement(self) -> bool:
        """True if spread exceeds 1 percentage point OR any model is an outlier."""
        return self.spread > 1.0 or len(self.outlier_models) > 0


def compute_disagreement(
    results: list[ForecastResult],
    outlier_z_threshold: float = 1.5,
) -> list[DisagreementReport]:
    """Compute cross-model disagreement for each forecast horizon.

    Args:
        results:              List of ForecastResult objects (one per model).
        outlier_z_threshold:  Z-score threshold for flagging outlier models.

    Returns:
        One DisagreementReport per forecast year across all models.
    """
    if not results:
        return []

    variable_id = results[0].variable_id

    # Align all forecasts to the same dates
    horizon_years = sorted({len(r.forecasts) for r in results}, reverse=True)[0]

    reports = []
    for yr_idx in range(horizon_years):
        row_data: dict[str, dict[str, float]] = {}
        forecast_date = None

        for result in results:
            if yr_idx >= len(result.forecasts):
                continue
            row = result.forecasts.iloc[yr_idx]
            row_data[result.model_id] = {
                "q10": float(row["q10"]),
                "q50": float(row["q50"]),
                "q90": float(row["q90"]),
            }
            forecast_date = row["date"]

        if not row_data or forecast_date is None:
            continue

        active_models = list(row_data.keys())
        q50s = np.array([row_data[m]["q50"] for m in active_models])
        q10s = np.array([row_data[m]["q10"] for m in active_models])
        q90s = np.array([row_data[m]["q90"] for m in active_models])

        mean_q50 = float(q50s.mean())
        std_q50 = float(q50s.std(ddof=0)) if len(q50s) > 1 else 0.0
        spread = float(q50s.max() - q50s.min())
        iqr = float(np.percentile(q50s, 75) - np.percentile(q50s, 25))

        z_scores = {}
        for m, q50 in zip(active_models, q50s):
            z_scores[m] = float((q50 - mean_q50) / std_q50) if std_q50 > 0 else 0.0

        outliers = [m for m, z in z_scores.items() if abs(z) > outlier_z_threshold]

        reports.append(DisagreementReport(
            variable_id=variable_id,
            horizon_year=yr_idx + 1,
            forecast_date=forecast_date,
            model_ids=active_models,
            point_forecasts={m: row_data[m]["q50"] for m in active_models},
            spread=spread,
            std=std_q50,
            iqr=iqr,
            z_scores=z_scores,
            outlier_models=outliers,
            ensemble_q50=mean_q50,
            ensemble_q10=float(np.percentile(q10s, 10)),
            ensemble_q90=float(np.percentile(q90s, 90)),
        ))

    return reports


def disagreement_to_dataframe(reports: list[DisagreementReport]) -> pd.DataFrame:
    """Convert a list of DisagreementReports to a tidy DataFrame."""
    rows = []
    for r in reports:
        base = {
            "variable_id": r.variable_id,
            "horizon_year": r.horizon_year,
            "forecast_date": r.forecast_date,
            "spread": r.spread,
            "std": r.std,
            "iqr": r.iqr,
            "high_disagreement": r.high_disagreement,
            "outlier_models": ",".join(r.outlier_models) if r.outlier_models else "",
            "ensemble_q10": r.ensemble_q10,
            "ensemble_q50": r.ensemble_q50,
            "ensemble_q90": r.ensemble_q90,
            "n_models": len(r.model_ids),
        }
        for model_id, q50 in r.point_forecasts.items():
            base[f"q50_{model_id}"] = q50
        rows.append(base)
    return pd.DataFrame(rows)
