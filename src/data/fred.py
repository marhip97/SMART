"""FRED (St. Louis Fed) data source – no API key required for public series."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import requests

from .base import DataSource, _assert_columns, _assert_no_nulls

logger = logging.getLogger(__name__)

# FRED public data API (no key required for the observations endpoint)
FRED_OBS_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"


class FREDDataSource(DataSource):
    """Fetch public time series from FRED via the CSV download endpoint.

    This endpoint requires no API key and is suitable for public series.

    source_params:
        series_id:  FRED series identifier (e.g. "DCOILBRENTEU", "ECBDFR")
    """

    def __init__(self, variable_id: str, source_params: dict[str, Any]) -> None:
        super().__init__(variable_id, source_params)
        self.series_id: str = source_params["series_id"]

    def fetch(self) -> pd.DataFrame:
        logger.info(
            "Fetching FRED series '%s' for variable '%s'.",
            self.series_id, self.variable_id,
        )
        params = {"id": self.series_id}
        response = requests.get(FRED_OBS_URL, params=params, timeout=60)
        response.raise_for_status()

        from io import StringIO
        df_raw = pd.read_csv(StringIO(response.text))

        # FRED CSV: columns are 'DATE' and the series ID
        df_raw.columns = ["date_raw", "value_raw"]
        df_raw["date"] = pd.to_datetime(df_raw["date_raw"], errors="coerce")
        # FRED uses "." for missing observations
        df_raw["value"] = pd.to_numeric(
            df_raw["value_raw"].replace(".", float("nan")), errors="coerce"
        )
        df = df_raw[["date", "value"]].dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        return df

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        _assert_columns(df, ["date", "value"])
        if len(df) == 0:
            raise ValueError(f"Empty DataFrame for FRED series '{self.series_id}'.")
        _assert_no_nulls(df, ["date"])
        null_pct = df["value"].isna().mean()
        if null_pct > 0.1:
            raise ValueError(
                f"Variable '{self.variable_id}': {null_pct:.0%} null values in 'value'."
            )
        return df
