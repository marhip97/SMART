"""ARIMA model – univariate benchmark using auto-selection via AIC/BIC."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

from .base import BaseModel, EvaluationResult, ForecastResult
from .utils import make_forecast_dates, walk_forward_eval


class ARIMAModel(BaseModel):
    """Auto-selected ARIMA(p,d,q) model for univariate forecasting.

    Fits an ARIMA model with order selected by AIC over a grid of
    (p, d, q) combinations. Produces point forecasts (q50) and
    approximate quantiles via forecast standard errors.
    """

    model_id = "arima"

    def __init__(
        self,
        variable_id: str,
        horizon_years: int = 3,
        quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
        random_state: int = 42,
        max_p: int = 4,
        max_q: int = 2,
        max_d: int = 2,
        information_criterion: str = "aic",
        **kwargs: Any,
    ) -> None:
        super().__init__(variable_id, horizon_years, quantiles, random_state)
        self.max_p = max_p
        self.max_q = max_q
        self.max_d = max_d
        self.information_criterion = information_criterion
        self._result: Any = None
        self._y: pd.Series | None = None

    def fit(self, y: pd.Series, X: pd.DataFrame | None = None) -> "ARIMAModel":
        """Fit ARIMA with order selection by AIC grid search."""
        self._y = y.dropna()
        d = self._select_d()
        best_model, best_ic = None, np.inf

        for p in range(0, self.max_p + 1):
            for q in range(0, self.max_q + 1):
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        mod = SARIMAX(
                            self._y,
                            order=(p, d, q),
                            trend="c",
                            enforce_stationarity=False,
                            enforce_invertibility=False,
                        )
                        res = mod.fit(disp=False)
                    ic = getattr(res, self.information_criterion)
                    if ic < best_ic:
                        best_ic = ic
                        best_model = res
                except Exception:  # noqa: BLE001
                    continue

        if best_model is None:
            raise RuntimeError(f"ARIMA grid search failed for variable '{self.variable_id}'.")

        self._result = best_model
        self._is_fitted = True
        return self

    def predict(self) -> ForecastResult:
        self._require_fitted()
        assert self._y is not None

        # Forecast horizon_years years ahead at the series frequency
        freq = pd.infer_freq(self._y.index)
        steps_per_year = _steps_per_year(freq)
        steps = self.horizon_years * steps_per_year

        forecast = self._result.get_forecast(steps=steps)
        mean = forecast.predicted_mean
        conf = forecast.conf_int(alpha=0.2)  # 80% CI → approx 10/90 quantiles

        # Aggregate to annual growth rates
        rows = []
        for yr in range(1, self.horizon_years + 1):
            sl = slice((yr - 1) * steps_per_year, yr * steps_per_year)
            q50 = float(mean.iloc[sl].mean())
            q10 = float(conf.iloc[sl, 0].mean())
            q90 = float(conf.iloc[sl, 1].mean())
            forecast_date = make_forecast_dates(self._y.index[-1], yr)
            rows.append({"date": forecast_date, "q10": q10, "q50": q50, "q90": q90})

        forecasts = pd.DataFrame(rows)
        return ForecastResult(
            variable_id=self.variable_id,
            model_id=self.model_id,
            forecasts=forecasts,
            metadata={"order": self._result.model.order, "aic": self._result.aic},
        )

    def evaluate(
        self,
        y: pd.Series,
        X: pd.DataFrame | None = None,
        min_train_years: int = 10,
    ) -> EvaluationResult:
        return walk_forward_eval(self, y, X, min_train_years)

    def _select_d(self) -> int:
        """Select differencing order d via ADF test (max self.max_d)."""
        assert self._y is not None
        for d in range(self.max_d + 1):
            series = self._y.diff(d).dropna() if d > 0 else self._y
            try:
                p_value = adfuller(series, autolag="AIC")[1]
                if p_value < 0.05:
                    return d
            except Exception:  # noqa: BLE001
                pass
        return 1  # default


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
