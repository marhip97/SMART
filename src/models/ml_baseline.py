"""ML baseline model – Random Forest with lagged features."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from .base import BaseModel, EvaluationResult, ForecastResult
from .utils import build_lag_features, make_forecast_dates, resample_to_annual, walk_forward_eval


class MLBaselineModel(BaseModel):
    """Random Forest regression model for non-linear baseline forecasts.

    Features: lagged values of the target variable and (optionally) lagged
    values of conditioning variables. Quantiles are estimated via quantile
    regression forests (sklearn's predict with return_decision_path trick is
    unavailable, so we approximate using the distribution of tree predictions).
    """

    model_id = "ml_baseline"

    def __init__(
        self,
        variable_id: str,
        horizon_years: int = 3,
        quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
        random_state: int = 42,
        n_estimators: int = 200,
        max_depth: int = 4,
        lags: list[int] | None = None,
        exog_lags: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(variable_id, horizon_years, quantiles, random_state)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.lags = lags or [1, 2, 4]
        self.exog_lags = exog_lags
        self._models: dict[int, RandomForestRegressor] = {}  # horizon → model
        self._y: pd.Series | None = None
        self._X: pd.DataFrame | None = None
        self._feature_cols: list[str] = []

    def fit(self, y: pd.Series, X: pd.DataFrame | None = None) -> "MLBaselineModel":
        """Fit one Random Forest per forecast horizon (direct multi-step strategy)."""
        y_ann = resample_to_annual(y).dropna()
        X_ann = X.resample("YS").mean() if X is not None else None

        self._y = y_ann
        self._X = X_ann
        self._models = {}

        features = build_lag_features(y_ann, X_ann, self.lags, self.exog_lags)
        self._feature_cols = list(features.columns)

        for h in range(1, self.horizon_years + 1):
            target = y_ann.shift(-h)  # h-step-ahead target
            df = features.join(target.rename("target")).dropna()
            if len(df) < 5:
                continue

            rf = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
                n_jobs=-1,
            )
            rf.fit(df[self._feature_cols], df["target"])
            self._models[h] = rf

        if not self._models:
            raise RuntimeError(f"ML baseline: no models fitted for '{self.variable_id}'.")

        self._is_fitted = True
        return self

    def predict(self) -> ForecastResult:
        self._require_fitted()
        assert self._y is not None

        features = build_lag_features(self._y, self._X, self.lags, self.exog_lags)
        last_features = features.iloc[[-1]][self._feature_cols]

        rows = []
        for h in range(1, self.horizon_years + 1):
            if h not in self._models:
                continue
            rf = self._models[h]

            # Quantile approximation via per-tree predictions
            tree_preds = np.array([tree.predict(last_features)[0] for tree in rf.estimators_])
            q10 = float(np.percentile(tree_preds, 10))
            q50 = float(np.percentile(tree_preds, 50))
            q90 = float(np.percentile(tree_preds, 90))

            rows.append({
                "date": make_forecast_dates(self._y.index[-1], h),
                "q10": q10,
                "q50": q50,
                "q90": q90,
            })

        return ForecastResult(
            variable_id=self.variable_id,
            model_id=self.model_id,
            forecasts=pd.DataFrame(rows),
            metadata={
                "n_estimators": self.n_estimators,
                "lags": self.lags,
                "feature_importances": {
                    h: dict(zip(self._feature_cols, self._models[h].feature_importances_))
                    for h in self._models
                },
            },
        )

    def evaluate(
        self,
        y: pd.Series,
        X: pd.DataFrame | None = None,
        min_train_years: int = 10,
    ) -> EvaluationResult:
        return walk_forward_eval(self, y, X, min_train_years)
