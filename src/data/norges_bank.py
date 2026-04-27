"""Norges Bank data source – styringsrente, EUR/NOK, handelspartnervekst, K2."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import requests

from .base import DataSource, _assert_columns, _assert_no_nulls

logger = logging.getLogger(__name__)

# Norges Bank Data (NBD) REST API
# Docs: https://data.norges-bank.no/api/
NBD_BASE = "https://data.norges-bank.no/api/data"

# Series codes used in variables.yaml → NBD dataset/series mapping
# Norges Bank SDMX-JSON REST API: https://data.norges-bank.no/api/data/{flow}/{key}
# Confirmed dataflows: SHORT_RATES (interest rates), EXR (exchange rates)
_SERIES_MAP: dict[str, tuple[str, str]] = {
    # series key → (NBD dataflow, series key within that flow)
    "SIREN":             ("SHORT_RATES", "B.SIGHT_DEP_RATE.NOK.D"),   # Styringsrente (daily → monthly)
    "EURNOK":            ("EXR",         "B.EUR.NOK.SP"),              # EUR/NOK spot rate
    "TPGDP":             ("MPM",         "TPGDP_Q"),                   # Handelspartnervekst (quarterly)
    "K2_HOUSEHOLDS_YOY": ("CR",          "K2.H.12M.NOK"),             # K2 husholdninger 12mnd-vekst
}


class NorgesBankDataSource(DataSource):
    """Fetch data from Norges Bank Data (data.norges-bank.no).

    source_params:
        series:  one of the keys defined in _SERIES_MAP above
    """

    def __init__(self, variable_id: str, source_params: dict[str, Any]) -> None:
        super().__init__(variable_id, source_params)
        self.series_key: str = source_params["series"]
        if self.series_key not in _SERIES_MAP:
            raise ValueError(
                f"Unknown Norges Bank series '{self.series_key}'. "
                f"Valid options: {list(_SERIES_MAP)}"
            )
        self.dataflow, self.series_id = _SERIES_MAP[self.series_key]

    def fetch(self) -> pd.DataFrame:
        logger.info(
            "Fetching Norges Bank series '%s' (flow=%s) for variable '%s'.",
            self.series_key, self.dataflow, self.variable_id,
        )
        url = f"{NBD_BASE}/{self.dataflow}/{self.series_id}"
        params = {
            "format": "sdmx-json",
            "startPeriod": "1990-01-01",
            "locale": "no",
        }
        response = requests.get(url, params=params, timeout=60)
        response.raise_for_status()

        data = response.json()
        df = _parse_sdmx_json(data)

        # Styringsrente is published as daily observations; resample to monthly mean
        if self.series_key == "SIREN":
            df = df.set_index("date").resample("MS").mean().reset_index()

        return df

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        _assert_columns(df, ["date", "value"])
        if len(df) == 0:
            raise ValueError(f"Empty DataFrame for variable '{self.variable_id}'.")
        _assert_no_nulls(df, ["date"])
        null_pct = df["value"].isna().mean()
        if null_pct > 0.1:
            raise ValueError(
                f"Variable '{self.variable_id}': {null_pct:.0%} null values in 'value'."
            )
        return df


def _parse_sdmx_json(data: dict) -> pd.DataFrame:
    """Extract a date/value DataFrame from a Norges Bank SDMX-JSON response."""
    try:
        structure = data["data"]["dataSets"][0]["series"]
        # Typically a single series; take the first key
        series_key = next(iter(structure))
        observations = structure[series_key]["observations"]

        time_periods = data["data"]["structure"]["dimensions"]["observation"]
        time_values = next(d for d in time_periods if d["id"] == "TIME_PERIOD")["values"]

        rows = []
        for idx_str, obs_list in observations.items():
            idx = int(idx_str)
            period_str = time_values[idx]["id"]
            value = obs_list[0]
            rows.append({"date_raw": period_str, "value": value})

        df = pd.DataFrame(rows)
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df["date"] = pd.to_datetime(df["date_raw"], errors="coerce")
        df = df[["date", "value"]].sort_values("date").reset_index(drop=True)
        return df
    except (KeyError, StopIteration, TypeError) as exc:
        raise ValueError(f"Could not parse Norges Bank SDMX-JSON response: {exc}") from exc
