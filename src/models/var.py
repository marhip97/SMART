"""VAR (Vector Autoregression) model for joint multivariate forecasting."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.var_model import VAR

from .base import BaseModel, EvaluationResult, ForecastResult
from .utils import make_forecast_dates, walk_forward_eval


class VARModel(BaseModel):
    """Vector Autoregression model (VAR).

    Fits a VAR(p) on all target variables jointly, with lag order
    selected by AIC. Forecasts each variable's annual growth rate
    for horizon_years ahead.

    Note: requires y to contain the full panel of target variables
    (passed as a DataFrame with one column per variable). When called
    from the pipeline on a single target variable, the VAR uses the
    full panel stored at fit time to produce joint forecasts, then
    returns the column for variable_id.
    """

    model_id = "var"

    def __init__(
        self,
        variable_id: str,
        horizon_years: int = 3,
        quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
        random_state: int = 42,
        max_lags: int = 4,
        ic: str = "aic",
        **kwargs: Any,
    ) -> None:
        super().__init__(variable_id, horizon_years, quantiles, random_state)
        self.max_lags = max_lags
        self.ic = ic
        self._result: Any = None
        self._columns: list[str] = []
        self._y_panel: pd.DataFrame | None = None

    def fit(self, y: pd.Series | pd.DataFrame, X: pd.DataFrame | None = None) -> "VARModel":
        """Fit VAR on a panel of variables.

        Args:
            y:  Either a single Series (variable_id column only – used in walk-forward)
                or a DataFrame with one column per target variable. When a Series is
                passed, the VAR reduces to a univariate AR, which degrades gracefully.
            X:  Ignored for VAR (all variables are endogenous).
        """
        if isinstance(y, pd.Series):
            panel = y.to_frame(name=self.variable_id)
        else:
            panel = y.copy()

        panel = panel.dropna()

        # VAR requires at least max_lags + 1 observations per variable
        if len(panel) < self.max_lags + 2:
            raise ValueError(
                f"VAR: insufficient data ({len(panel)} obs) for max_lags={self.max_lags}."
            )

        self._columns = list(panel.columns)
        self._y_panel = panel

        # statsmodels VAR requires at least 2 endogenous variables
        if panel.shape[1] < 2:
            self._result = _fit_ar_fallback(panel.iloc[:, 0], self.max_lags)
            self._is_fitted = True
            return self

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            var_model = VAR(panel)
            try:
                self._result = var_model.fit(maxlags=self.max_lags, ic=self.ic, trend="c")
            except Exception:
                # IC-based selection can fail (e.g. Cholesky); fall back to fixed lag=1
                self._result = var_model.fit(maxlags=1, trend="c")

        self._is_fitted = True
        return self

    def predict(self) -> ForecastResult:
        self._require_fitted()
        assert self._y_panel is not None

        # AR fallback path (single-variable case)
        if hasattr(self._result, "get_forecast"):
            return _predict_from_ar_fallback(
                self._result, self._y_panel, self.variable_id, self.model_id, self.horizon_years
            )

        freq = pd.infer_freq(self._y_panel.index)
        steps_per_year = _steps_per_year(freq)
        steps = self.horizon_years * steps_per_year

        fc = self._result.forecast(self._y_panel.values[-self._result.k_ar :], steps=steps)
        fc_df = pd.DataFrame(fc, columns=self._columns)

        # Extract target variable column; fall back to first column if not present
        col = self.variable_id if self.variable_id in self._columns else self._columns[0]
        mean_series = fc_df[col]

        # Forecast intervals via MSE (statsmodels provides forecast_interval)
        try:
            lower, upper = self._result.forecast_interval(
                self._y_panel.values[-self._result.k_ar :], steps=steps, alpha=0.2
            )
            lo_series = pd.Series(lower[:, self._columns.index(col)])
            hi_series = pd.Series(upper[:, self._columns.index(col)])
        except Exception:  # noqa: BLE001
            lo_series = mean_series - 1.0
            hi_series = mean_series + 1.0

        rows = []
        for yr in range(1, self.horizon_years + 1):
            sl = slice((yr - 1) * steps_per_year, yr * steps_per_year)
            rows.append({
                "date": make_forecast_dates(self._y_panel.index[-1], yr),
                "q10": float(lo_series.iloc[sl].mean()),
                "q50": float(mean_series.iloc[sl].mean()),
                "q90": float(hi_series.iloc[sl].mean()),
            })

        return ForecastResult(
            variable_id=self.variable_id,
            model_id=self.model_id,
            forecasts=pd.DataFrame(rows),
            metadata={"lag_order": self._result.k_ar, "ic": self.ic},
        )

    def evaluate(
        self,
        y: pd.Series,
        X: pd.DataFrame | None = None,
        min_train_years: int = 10,
    ) -> EvaluationResult:
        return walk_forward_eval(self, y, X, min_train_years)


def _fit_ar_fallback(y: pd.Series, max_lags: int) -> Any:
    """Fit a univariate SARIMAX(p,0,0) as fallback when VAR needs ≥2 series."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    best, best_aic = None, np.inf
    for p in range(1, max_lags + 1):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = SARIMAX(y, order=(p, 0, 0), trend="c", enforce_stationarity=False).fit(disp=False)
            if res.aic < best_aic:
                best_aic = res.aic
                best = res
        except Exception:  # noqa: BLE001
            continue
    if best is None:
        raise RuntimeError("VAR AR-fallback fitting failed.")
    return best


def _predict_from_ar_fallback(
    result: Any,
    y_panel: pd.DataFrame,
    variable_id: str,
    model_id: str,
    horizon_years: int,
) -> ForecastResult:
    freq = pd.infer_freq(y_panel.index)
    steps_per_year = _steps_per_year(freq)
    steps = horizon_years * steps_per_year
    fc = result.get_forecast(steps=steps)
    mean = fc.predicted_mean
    conf = fc.conf_int(alpha=0.2)
    rows = []
    for yr in range(1, horizon_years + 1):
        sl = slice((yr - 1) * steps_per_year, yr * steps_per_year)
        rows.append({
            "date": make_forecast_dates(y_panel.index[-1], yr),
            "q10": float(conf.iloc[sl, 0].mean()),
            "q50": float(mean.iloc[sl].mean()),
            "q90": float(conf.iloc[sl, 1].mean()),
        })
    return ForecastResult(
        variable_id=variable_id,
        model_id=model_id,
        forecasts=pd.DataFrame(rows),
        metadata={"fallback": "AR", "reason": "single-variable input"},
    )


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
