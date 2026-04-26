"""NAV open data source – registered unemployment rate."""

from __future__ import annotations

import logging
from typing import Any

import requests

from .base import DataSource, _assert_columns, _assert_no_nulls

logger = logging.getLogger(__name__)

# NAV publishes monthly labour market statistics as JSON.
# Endpoint returns an array of {year, month, value} objects.
NAV_API_URL = "https://data.nav.no/api/v1/arbeidsmarked/ledigestillinger"

# Fallback: NAV also exposes data via SSB (table 05111 – registrerte ledige).
# We use a direct NAV endpoint where available and fall back to SSB otherwise.
NAV_SSB_TABLE = "05111"
NAV_SSB_URL = "https://data.ssb.no/api/v0/no/table"


class NAVDataSource(DataSource):
    """Fetch registered unemployment rate from NAV.

    source_params:
        series: identifier string (currently only "registrert_ledige_pct" supported)
    """

    def __init__(self, variable_id: str, source_params: dict[str, Any]) -> None:
        super().__init__(variable_id, source_params)
        self.series: str = source_params["series"]

    def fetch(self) -> pd.DataFrame:
        logger.info("Fetching registered unemployment (NAV/SSB) for '%s'.", self.variable_id)
        # NAV's preferred open-data endpoint for registered unemployment as share of labour force
        # is currently served through SSB table 05111 (Registrerte arbeidsledige).
        return self._fetch_via_ssb()

    def _fetch_via_ssb(self) -> pd.DataFrame:
        query = {
            "query": [
                {"code": "Region", "selection": {"filter": "item", "values": ["0"]}},
                {"code": "Kjonn", "selection": {"filter": "item", "values": ["0"]}},
                {"code": "ContentsCode", "selection": {"filter": "item", "values": ["Ledige2"]}},
                {"code": "Tid", "selection": {"filter": "all", "values": ["*"]}},
            ],
            "response": {"format": "json-stat2"},
        }
        url = f"{NAV_SSB_URL}/{NAV_SSB_TABLE}"
        response = requests.post(url, json=query, timeout=60)
        response.raise_for_status()

        from .ssb import _parse_jsonstat2

        # _parse_jsonstat2 returns a normalized date/value DataFrame
        return _parse_jsonstat2(response.json())

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        _assert_columns(df, ["date", "value"])
        if len(df) == 0:
            raise ValueError(f"Empty DataFrame for variable '{self.variable_id}'.")
        _assert_no_nulls(df, ["date"])
        if df["value"].dropna().max() > 25:
            raise ValueError(
                f"Variable '{self.variable_id}': suspiciously high unemployment rate "
                f"({df['value'].max():.1f}%). Check units."
            )
        return df


