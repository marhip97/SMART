"""EnsembleForecaster – aggregates individual model forecasts into a combined view.

Supports three weighting schemes:
  - equal:       Simple mean of all model q50s (transparent, robust baseline).
  - performance: Weights inversely proportional to each model's historical RMSE.
  - trimmed:     Equal-weight mean after dropping the single highest and lowest
                 point forecasts (robust to outlier models).

The ensemble also produces a combined quantile fan chart: the q10 is the
10th percentile of all model q10s and q90 is the 90th percentile of all
model q90s, giving a conservative (wider) uncertainty band that reflects
both within-model and across-model uncertainty.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from src.models.base import EvaluationResult, ForecastResult
from .disagreement import DisagreementReport, compute_disagreement

WeightingScheme = Literal["equal", "performance", "trimmed"]


def _apply_weight_floor(weights: dict[str, float], min_weight: float) -> dict[str, float]:
    """Redistribute weights so every model gets at least min_weight.

    Uses iterative pinning: models below the floor are pinned to min_weight and
    the remaining budget is distributed proportionally among the free models.
    """
    weights = dict(weights)
    for _ in range(len(weights) + 1):
        total = sum(weights.values())
        weights = {mid: w / total for mid, w in weights.items()}
        below = [mid for mid, w in weights.items() if w < min_weight]
        if not below:
            break
        pinned = {mid: min_weight for mid in below}
        free = {mid: w for mid, w in weights.items() if mid not in below}
        if not free:
            n = len(weights)
            return {mid: 1.0 / n for mid in weights}
        remaining = 1.0 - len(pinned) * min_weight
        if remaining <= 0:
            n = len(weights)
            return {mid: 1.0 / n for mid in weights}
        free_total = sum(free.values())
        weights = {**pinned, **{mid: w / free_total * remaining for mid, w in free.items()}}
    return weights


@dataclass
class EnsembleForecast:
    """Aggregated forecast produced by EnsembleForecaster.

    Attributes:
        variable_id:    Target variable.
        weighting:      Weighting scheme used.
        forecasts:      DataFrame with ['date', 'q10', 'q50', 'q90'].
        weights:        Dict mapping model_id → weight used for q50.
        disagreement:   Per-horizon DisagreementReport objects.
        model_forecasts: Original per-model ForecastResult objects.
    """

    variable_id: str
    weighting: WeightingScheme
    forecasts: pd.DataFrame
    weights: dict[str, float]
    disagreement: list[DisagreementReport]
    model_forecasts: list[ForecastResult] = field(default_factory=list)


class EnsembleForecaster:
    """Combines forecasts from multiple models into an ensemble.

    Args:
        variable_id:   Target variable ID (must match all ForecastResult objects).
        weighting:     Weighting scheme: 'equal', 'performance', or 'trimmed'.
        eval_results:  Per-model EvaluationResult objects (required for 'performance').
        min_weight:    Minimum weight floor per model (avoids zero-weight models).
    """

    def __init__(
        self,
        variable_id: str,
        weighting: WeightingScheme = "equal",
        eval_results: list[EvaluationResult] | None = None,
        min_weight: float = 0.05,
    ) -> None:
        self.variable_id = variable_id
        self.weighting = weighting
        self.eval_results = eval_results or []
        self.min_weight = min_weight

    def combine(self, results: list[ForecastResult]) -> EnsembleForecast:
        """Combine a list of ForecastResult objects into an ensemble forecast.

        Args:
            results: One ForecastResult per model, all for the same variable_id.

        Returns:
            EnsembleForecast with aggregated q10/q50/q90 and disagreement metrics.

        Raises:
            ValueError: If results is empty or variable_ids do not match.
        """
        if not results:
            raise ValueError("Cannot combine an empty list of ForecastResult objects.")

        mismatched = [r.variable_id for r in results if r.variable_id != self.variable_id]
        if mismatched:
            raise ValueError(
                f"variable_id mismatch: expected '{self.variable_id}', "
                f"got {mismatched}"
            )

        weights = self._compute_weights(results)
        disagreement = compute_disagreement(results)
        forecasts_df = self._aggregate(results, weights)

        return EnsembleForecast(
            variable_id=self.variable_id,
            weighting=self.weighting,
            forecasts=forecasts_df,
            weights=weights,
            disagreement=disagreement,
            model_forecasts=results,
        )

    # ── Private helpers ────────────────────────────────────────────────────────

    def _compute_weights(self, results: list[ForecastResult]) -> dict[str, float]:
        if self.weighting == "equal":
            n = len(results)
            return {r.model_id: 1.0 / n for r in results}

        if self.weighting == "trimmed":
            # Weights are equal; trimming happens in _aggregate
            n = len(results)
            return {r.model_id: 1.0 / n for r in results}

        if self.weighting == "performance":
            return self._performance_weights(results)

        raise ValueError(f"Unknown weighting scheme: '{self.weighting}'")

    def _performance_weights(self, results: list[ForecastResult]) -> dict[str, float]:
        """Weights inversely proportional to RMSE; equal weight for missing evals."""
        eval_map = {e.model_id: e.rmse for e in self.eval_results if np.isfinite(e.rmse)}
        model_ids = [r.model_id for r in results]

        raw_weights = {}
        for mid in model_ids:
            rmse = eval_map.get(mid)
            if rmse is None or rmse == 0:
                raw_weights[mid] = 1.0  # no eval → equal weight
            else:
                raw_weights[mid] = 1.0 / rmse

        total = sum(raw_weights.values())
        weights = {mid: w / total for mid, w in raw_weights.items()}
        return _apply_weight_floor(weights, self.min_weight)

    def _aggregate(
        self, results: list[ForecastResult], weights: dict[str, float]
    ) -> pd.DataFrame:
        """Build the ensemble forecast DataFrame."""
        horizon = max(len(r.forecasts) for r in results)
        rows = []

        for yr_idx in range(horizon):
            valid = [r for r in results if yr_idx < len(r.forecasts)]
            if not valid:
                continue

            dates = [r.forecasts.iloc[yr_idx]["date"] for r in valid]
            forecast_date = dates[0]

            q10s = np.array([float(r.forecasts.iloc[yr_idx]["q10"]) for r in valid])
            q50s = np.array([float(r.forecasts.iloc[yr_idx]["q50"]) for r in valid])
            q90s = np.array([float(r.forecasts.iloc[yr_idx]["q90"]) for r in valid])

            if self.weighting == "trimmed" and len(valid) >= 4:
                # Drop highest and lowest q50 forecasters
                ranks = np.argsort(q50s)
                keep = ranks[1:-1]
                q50_agg = float(q50s[keep].mean())
                q10_agg = float(np.percentile(q10s[keep], 10))
                q90_agg = float(np.percentile(q90s[keep], 90))
            else:
                w = np.array([weights.get(r.model_id, 1.0 / len(valid)) for r in valid])
                w = w / w.sum()
                q50_agg = float(w @ q50s)
                # Conservative bounds: percentile across models
                q10_agg = float(np.percentile(q10s, 10))
                q90_agg = float(np.percentile(q90s, 90))

            rows.append({
                "date": forecast_date,
                "q10": q10_agg,
                "q50": q50_agg,
                "q90": q90_agg,
            })

        return pd.DataFrame(rows)
