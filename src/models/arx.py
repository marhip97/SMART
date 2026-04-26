"""AR-X model – autoregressive model with exogenous variables."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from .base import BaseModel, EvaluationResult, ForecastResult
from .utils import make_forecast_dates, walk_forward_eval


class ARXModel(BaseModel):
    """Autoregressive model with exogenous (conditioning) variables (AR-X).

    Fits an AR(p) model augmented with lagged values of the exogenous
    variables. Lag order p is selected by AIC. Conditioning variables
    are supplied at fit time; for forecasting, exogenous variables are
    assumed to remain at their last observed value (naive assumption –
    acceptable for annual horizons where scenarios can later be swapped in).
    """

    model_id = "arx"

    def __init__(
        self,
        variable_id: str,
        horizon_years: int = 3,
        quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
        random_state: int = 42,
        max_lags: int = 4,
        exog_lags: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(variable_id, horizon_years, quantiles, random_state)
        self.max_lags = max_lags
        self.exog_lags = exog_lags
        self._result: Any = None
        self._y: pd.Series | None = None
        self._X_train: pd.DataFrame | None = None
        self._best_p: int = 1

    def fit(self, y: pd.Series, X: pd.DataFrame | None = None) -> "ARXModel":
        """Fit AR-X with AIC-selected lag order."""
        y_clean = y.dropna()

        # Build lagged exogenous matrix and align y to the common non-NaN index
        exog = self._build_exog(y_clean, X)
        if exog is not None:
            common_idx = y_clean.index.intersection(exog.index)
            y_clean = y_clean.loc[common_idx]
            exog = exog.loc[common_idx]

        self._y = y_clean

        best_res, best_aic = None, np.inf
        for p in range(1, self.max_lags + 1):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    mod = SARIMAX(
                        self._y,
                        exog=exog,
                        order=(p, 0, 0),
                        trend="c",
                        enforce_stationarity=False,
                    )
                    res = mod.fit(disp=False)
                if res.aic < best_aic:
                    best_aic = res.aic
                    best_res = res
                    self._best_p = p
            except Exception:  # noqa: BLE001
                continue

        if best_res is None:
            raise RuntimeError(f"AR-X fitting failed for variable '{self.variable_id}'.")

        self._result = best_res
        self._X_train = X
        self._is_fitted = True
        return self

    def predict(self) -> ForecastResult:
        self._require_fitted()
        assert self._y is not None

        freq = pd.infer_freq(self._y.index)
        steps_per_year = _steps_per_year(freq)
        steps = self.horizon_years * steps_per_year

        # Naïve assumption: exogenous variables stay at last observed value
        if self._X_train is not None:
            last_exog = self._build_exog(self._y, self._X_train)
            if last_exog is not None and len(last_exog) > 0:
                exog_forecast = pd.DataFrame(
                    [last_exog.iloc[-1].values] * steps,
                    columns=last_exog.columns,
                )
            else:
                exog_forecast = None
        else:
            exog_forecast = None

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            forecast = self._result.get_forecast(steps=steps, exog=exog_forecast)

        mean = forecast.predicted_mean
        conf = forecast.conf_int(alpha=0.2)

        rows = []
        for yr in range(1, self.horizon_years + 1):
            sl = slice((yr - 1) * steps_per_year, yr * steps_per_year)
            rows.append({
                "date": make_forecast_dates(self._y.index[-1], yr),
                "q10": float(conf.iloc[sl, 0].mean()),
                "q50": float(mean.iloc[sl].mean()),
                "q90": float(conf.iloc[sl, 1].mean()),
            })

        return ForecastResult(
            variable_id=self.variable_id,
            model_id=self.model_id,
            forecasts=pd.DataFrame(rows),
            metadata={"ar_order": self._best_p, "n_exog": 0 if self._X_train is None else self._X_train.shape[1]},
        )

    def evaluate(
        self,
        y: pd.Series,
        X: pd.DataFrame | None = None,
        min_train_years: int = 10,
    ) -> EvaluationResult:
        return walk_forward_eval(self, y, X, min_train_years)

    def _build_exog(self, y: pd.Series, X: pd.DataFrame | None) -> pd.DataFrame | None:
        """Build lagged exogenous feature matrix on y's index, without dropping NaN rows.

        Caller is responsible for intersecting y and exog indexes.
        """
        if X is None:
            return None
        frames = []
        for col in X.columns:
            for lag in range(1, self.exog_lags + 1):
                s = X[col].shift(lag)
                s.name = f"{col}_lag{lag}"
                frames.append(s)
        exog = pd.concat(frames, axis=1).reindex(y.index).dropna()
        return exog if len(exog) > 0 else None


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
