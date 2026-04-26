"""SSB Statistikkbanken data source (JSON-stat2 API, parsed without external libraries)."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import requests

from .base import DataSource, _assert_columns, _assert_no_nulls

logger = logging.getLogger(__name__)

SSB_API_BASE = "https://data.ssb.no/api/v0/no/table"


class SSBDataSource(DataSource):
    """Fetch time series from SSB Statistikkbanken.

    source_params:
        table_id:  SSB table number (string, e.g. "09190")
        filters:   dict mapping dimension names to lists of values or ["*"]
    """

    def __init__(self, variable_id: str, source_params: dict[str, Any]) -> None:
        super().__init__(variable_id, source_params)
        self.table_id: str = source_params["table_id"]
        self.filters: dict[str, list[str]] = source_params["filters"]

    def _build_query(self) -> dict:
        query = {
            "query": [
                {"code": dim, "selection": {"filter": "all" if vals == ["*"] else "item", "values": vals}}
                for dim, vals in self.filters.items()
            ],
            "response": {"format": "json-stat2"},
        }
        return query

    def fetch(self) -> pd.DataFrame:
        url = f"{SSB_API_BASE}/{self.table_id}"
        query = self._build_query()
        logger.info("Fetching SSB table %s for variable '%s'.", self.table_id, self.variable_id)

        response = requests.post(url, json=query, timeout=60)
        response.raise_for_status()

        df = _parse_jsonstat2(response.json())
        return df

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        _assert_columns(df, ["date", "value"])
        if len(df) == 0:
            raise ValueError(f"Empty DataFrame for variable '{self.variable_id}'.")
        _assert_no_nulls(df, ["date"])
        null_values = df["value"].isna().sum()
        if null_values / len(df) > 0.1:
            raise ValueError(
                f"Variable '{self.variable_id}': more than 10% null values ({null_values}/{len(df)})."
            )
        return df


def _parse_jsonstat2(data: dict) -> pd.DataFrame:
    """Parse a JSON-stat2 dataset response from SSB into a date/value DataFrame.

    Handles datasets with one or more dimensions. The time dimension ('Tid') is
    extracted as the date column; all other dimensions are ignored (assuming the
    query already filtered to a single value per non-time dimension).
    """
    dimensions = data["id"]          # e.g. ["ContentsCode", "Tid"]
    sizes = data["size"]             # e.g. [1, 120]
    values = data["value"]           # flat array, row-major

    # Locate the time dimension
    time_idx = next(
        (i for i, d in enumerate(dimensions) if d.lower() in ("tid", "time")),
        len(dimensions) - 1,         # fall back to last dimension
    )
    time_dim_key = dimensions[time_idx]
    time_categories = data["dimension"][time_dim_key]["category"]

    # Build ordered list of time labels
    if "index" in time_categories:
        # index maps label → position; invert to get position → label
        index_map = time_categories["index"]
        n_time = sizes[time_idx]
        inv = {v: k for k, v in index_map.items()}
        time_labels = [inv[i] for i in range(n_time)]
    else:
        time_labels = list(time_categories.get("label", {}).keys())

    # Step size through the flat value array to extract per-time-period values
    # (works correctly when non-time dimensions each have size=1)
    n_time = len(time_labels)
    stride = 1
    for i in range(time_idx + 1, len(dimensions)):
        stride *= sizes[i]

    # Outer product size before the time dimension
    outer = 1
    for i in range(time_idx):
        outer *= sizes[i]

    rows = []
    flat_idx = 0
    for _ in range(outer):
        for t, label in enumerate(time_labels):
            val = values[flat_idx + t * stride]
            rows.append({"date_raw": label, "value": val})
        flat_idx += n_time * stride

    df = pd.DataFrame(rows)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["date"] = df["date_raw"].apply(_parse_ssb_date)
    df = df.dropna(subset=["date"])
    return df[["date", "value"]].sort_values("date").reset_index(drop=True)


def _parse_ssb_date(s: str) -> pd.Timestamp | None:
    """Parse SSB time codes to pandas Timestamps.

    Handles:
        "2023K1"  → 2023-01-01 (quarterly, first day of quarter)
        "2023M01" → 2023-01-01 (monthly)
        "2023"    → 2023-01-01 (annual)
    """
    s = str(s).strip()
    try:
        if "K" in s:
            year, q = s.split("K")
            month = (int(q) - 1) * 3 + 1
            return pd.Timestamp(year=int(year), month=month, day=1)
        if "M" in s:
            year, month = s.split("M")
            return pd.Timestamp(year=int(year), month=int(month), day=1)
        if len(s) == 4 and s.isdigit():
            return pd.Timestamp(year=int(s), month=1, day=1)
    except (ValueError, TypeError):
        pass
    return None
