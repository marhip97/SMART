"""Shared utilities for SMART model implementations."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .base import BaseModel, EvaluationResult


def make_forecast_dates(last_obs: pd.Timestamp, years_ahead: int) -> pd.Timestamp:
    """Return the first day of the calendar year `years_ahead` after `last_obs`."""
    return pd.Timestamp(year=last_obs.year + years_ahead, month=1, day=1)


def annual_to_quarterly(annual_value: float, n_quarters: int = 4) -> list[float]:
    """Distribute an annual growth rate evenly across quarters."""
    return [annual_value] * n_quarters


def resample_to_annual(y: pd.Series) -> pd.Series:
    """Resample a monthly/quarterly series to annual mean."""
    return y.resample("YS").mean()


def build_lag_features(
    y: pd.Series,
    X: pd.DataFrame | None,
    lags: list[int],
    exog_lags: int = 1,
) -> pd.DataFrame:
    """Build a feature matrix of lagged y values and optionally lagged X columns."""
    frames = []
    for lag in lags:
        s = y.shift(lag)
        s.name = f"y_lag{lag}"
        frames.append(s)

    if X is not None:
        for col in X.columns:
            for lag in range(1, exog_lags + 1):
                s = X[col].shift(lag)
                s.name = f"{col}_lag{lag}"
                frames.append(s)

    return pd.concat(frames, axis=1)


def walk_forward_eval(
    model: "BaseModel",
    y: pd.Series,
    X: pd.DataFrame | None,
    min_train_years: int,
) -> "EvaluationResult":
    """Walk-forward (expanding window) backtesting – one-year-ahead forecasts.

    For each evaluation period:
      1. Fit the model on data up to and including year t.
      2. Predict year t+1.
      3. Compare q50 forecast with realised value.

    Args:
        model:           An unfitted BaseModel instance.
        y:               Full target series.
        X:               Optional exogenous variables.
        min_train_years: Minimum years of training data before first forecast.

    Returns:
        EvaluationResult with RMSE and MAE across all evaluation periods.
    """
    from .base import EvaluationResult  # avoid circular at module level

    y_annual = resample_to_annual(y).dropna()
    if X is not None:
        X_annual = X.resample("YS").mean()
    else:
        X_annual = None

    errors: list[float] = []

    for cutoff in y_annual.index[min_train_years:-1]:
        y_train = y_annual.loc[:cutoff]
        X_train = X_annual.loc[:cutoff] if X_annual is not None else None

        # Clone model to avoid state bleed between iterations
        m = copy.deepcopy(model)
        m._is_fitted = False

        try:
            m.fit(y_train, X_train)
            result = m.predict()
            q50_next_year = float(result.forecasts["q50"].iloc[0])
        except Exception:  # noqa: BLE001
            continue

        next_year = pd.Timestamp(year=cutoff.year + 1, month=1, day=1)
        if next_year not in y_annual.index:
            continue
        actual = float(y_annual.loc[next_year])
        errors.append(q50_next_year - actual)

    if not errors:
        return EvaluationResult(
            variable_id=model.variable_id,
            model_id=model.model_id,
            rmse=float("nan"),
            mae=float("nan"),
            n_obs=0,
        )

    arr = np.array(errors)
    return EvaluationResult(
        variable_id=model.variable_id,
        model_id=model.model_id,
        rmse=float(np.sqrt(np.mean(arr**2))),
        mae=float(np.mean(np.abs(arr))),
        n_obs=len(arr),
    )
