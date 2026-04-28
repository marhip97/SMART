"""Abstract base class for all forecast models in SMART."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class ForecastResult:
    """Standardised output from a model's predict() call.

    Attributes:
        variable_id:  ID of the target variable (matches config).
        model_id:     ID of the model that produced the forecast.
        forecasts:    DataFrame with columns ['date', 'q10', 'q50', 'q90'].
                      'date' is the first day of the forecast year.
                      q10/q50/q90 are annual growth rate forecasts (%).
        metadata:     Optional dict with model-specific diagnostics.
    """

    variable_id: str
    model_id: str
    forecasts: pd.DataFrame
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        required = {"date", "q10", "q50", "q90"}
        missing = required - set(self.forecasts.columns)
        if missing:
            raise ValueError(f"ForecastResult.forecasts missing columns: {missing}")


@dataclass
class EvaluationResult:
    """Out-of-sample evaluation metrics from backtesting.

    Attributes:
        variable_id:  Target variable ID.
        model_id:     Model ID.
        rmse:         Root mean squared error of point forecasts (q50).
        mae:          Mean absolute error of point forecasts.
        r2:           Out-of-sample R² (can be negative if worse than naïve mean).
        n_obs:        Number of out-of-sample observations evaluated.
        details:      Optional per-horizon or per-period breakdown.
                      details["backtest"] holds per-year actual vs forecast pairs.
    """

    variable_id: str
    model_id: str
    rmse: float
    mae: float
    n_obs: int
    r2: float = field(default_factory=lambda: float("nan"))
    details: dict[str, Any] = field(default_factory=dict)


class BaseModel(ABC):
    """Abstract base class every forecast model must subclass.

    Subclasses implement fit(), predict(), and evaluate().
    The model_id class attribute must be set on each subclass.
    """

    model_id: str = ""

    def __init__(
        self,
        variable_id: str,
        horizon_years: int = 3,
        quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        self.variable_id = variable_id
        self.horizon_years = horizon_years
        self.quantiles = quantiles
        self.random_state = random_state
        self._is_fitted = False

    @abstractmethod
    def fit(self, y: pd.Series, X: pd.DataFrame | None = None) -> "BaseModel":
        """Fit the model on historical data.

        Args:
            y:  Target time series (index: DatetimeIndex, values: annual growth rates %).
            X:  Optional DataFrame of conditioning/exogenous variables (same index as y).

        Returns:
            self
        """

    @abstractmethod
    def predict(self) -> ForecastResult:
        """Generate forecasts for horizon_years ahead.

        Must be called after fit(). Returns a ForecastResult with annual
        growth rate forecasts for each of the next horizon_years years,
        at quantiles 10/50/90.
        """

    @abstractmethod
    def evaluate(
        self,
        y: pd.Series,
        X: pd.DataFrame | None = None,
        min_train_years: int = 10,
    ) -> EvaluationResult:
        """Walk-forward out-of-sample backtesting.

        Args:
            y:               Full target series.
            X:               Optional exogenous variables.
            min_train_years: Minimum years of data before first forecast.

        Returns:
            EvaluationResult with RMSE and MAE on one-year-ahead forecasts.
        """

    def _require_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(f"{self.__class__.__name__}.fit() must be called before predict().")
