"""Dynamic Factor Model (DFM) using statsmodels DynamicFactorMQ."""

from __future__ import annotations

import warnings
from typing import Any

import pandas as pd
from statsmodels.tsa.statespace.dynamic_factor_mq import DynamicFactorMQ

from .base import BaseModel, EvaluationResult, ForecastResult
from .utils import make_forecast_dates, walk_forward_eval


class DFMModel(BaseModel):
    """Dynamic Factor Model for extracting latent common factors.

    Uses statsmodels DynamicFactorMQ which supports mixed-frequency data.
    When fitted on a single series, it reduces to a simple factor model
    and degrades gracefully. For best results, pass a multi-column panel
    (target + conditioning variables) as y.
    """

    model_id = "dfm"

    def __init__(
        self,
        variable_id: str,
        horizon_years: int = 3,
        quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
        random_state: int = 42,
        n_factors: int = 2,
        factor_order: int = 1,
        error_order: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(variable_id, horizon_years, quantiles, random_state)
        self.n_factors = n_factors
        self.factor_order = factor_order
        self.error_order = error_order
        self._result: Any = None
        self._y_panel: pd.DataFrame | None = None
        self._col_idx: int = 0

    def fit(self, y: pd.Series | pd.DataFrame, X: pd.DataFrame | None = None) -> "DFMModel":
        """Fit DFM on a panel of variables.

        Args:
            y:  Series or DataFrame. Multi-column panel improves factor extraction.
            X:  Optional additional variables; merged into the panel.
        """
        if isinstance(y, pd.Series):
            panel = y.to_frame(name=self.variable_id)
        else:
            panel = y.copy()

        if X is not None:
            panel = panel.join(X, how="left")

        panel = panel.dropna(how="all")

        # Clamp n_factors to number of columns
        n_factors = min(self.n_factors, panel.shape[1])
        self._col_idx = list(panel.columns).index(self.variable_id) if self.variable_id in panel.columns else 0
        self._y_panel = panel

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mod = DynamicFactorMQ(
                panel,
                factors=n_factors,
                factor_orders=self.factor_order,
                error_order=self.error_order,
            )
            self._result = mod.fit(disp=False, maxiter=200)

        self._is_fitted = True
        return self

    def predict(self) -> ForecastResult:
        self._require_fitted()
        assert self._y_panel is not None

        freq = pd.infer_freq(self._y_panel.index)
        steps_per_year = _steps_per_year(freq)
        steps = self.horizon_years * steps_per_year

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            forecast_obj = self._result.get_forecast(steps=steps)

        mean = forecast_obj.predicted_mean
        conf = forecast_obj.conf_int(alpha=0.2)

        # mean may be a Series (single variable) or DataFrame (multi-variable)
        if isinstance(mean, pd.Series):
            mean_vals = mean
            lo_vals = conf.iloc[:, 0]
            hi_vals = conf.iloc[:, 1]
        else:
            col = self.variable_id if self.variable_id in mean.columns else mean.columns[self._col_idx]
            mean_vals = mean[col]
            lo_vals = conf[f"lower {col}"]
            hi_vals = conf[f"upper {col}"]

        rows = []
        for yr in range(1, self.horizon_years + 1):
            sl = slice((yr - 1) * steps_per_year, yr * steps_per_year)
            rows.append({
                "date": make_forecast_dates(self._y_panel.index[-1], yr),
                "q10": float(lo_vals.iloc[sl].mean()),
                "q50": float(mean_vals.iloc[sl].mean()),
                "q90": float(hi_vals.iloc[sl].mean()),
            })

        return ForecastResult(
            variable_id=self.variable_id,
            model_id=self.model_id,
            forecasts=pd.DataFrame(rows),
            metadata={"n_factors": self.n_factors},
        )

    def evaluate(
        self,
        y: pd.Series,
        X: pd.DataFrame | None = None,
        min_train_years: int = 10,
    ) -> EvaluationResult:
        return walk_forward_eval(self, y, X, min_train_years)


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
