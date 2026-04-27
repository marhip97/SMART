"""SSB Statistikkbanken data source (JSON-stat2 API, parsed without external libraries)."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import requests

from .base import DataSource, _assert_columns, _assert_no_nulls

logger = logging.getLogger(__name__)

SSB_API_BASE = "https://data.ssb.no/api/v0/no/table"

# Module-level metadata cache – avoids re-fetching within the same process run
_METADATA_CACHE: dict[str, dict] = {}


def _fetch_metadata(table_id: str) -> dict:
    """GET table metadata from SSB; cached per table_id."""
    if table_id not in _METADATA_CACHE:
        url = f"{SSB_API_BASE}/{table_id}"
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        _METADATA_CACHE[table_id] = resp.json()
    return _METADATA_CACHE[table_id]


class SSBDataSource(DataSource):
    """Fetch time series from SSB Statistikkbanken.

    source_params:
        table_id:  SSB table number (string, e.g. "09190")
        filters:   dict mapping dimension names to lists of values or ["*"]
                   String values are automatically normalised to single-item lists.
    """

    def __init__(self, variable_id: str, source_params: dict[str, Any]) -> None:
        super().__init__(variable_id, source_params)
        self.table_id: str = source_params["table_id"]
        # Normalise: scalar strings → single-item list
        raw = source_params["filters"]
        self.filters: dict[str, list[str]] = {
            dim: ([v] if isinstance(v, str) else list(v))
            for dim, v in raw.items()
        }

    # ── Validation ─────────────────────────────────────────────────────────────

    def _validate_filters(self) -> None:
        """Validate configured dimension names and values against SSB table metadata.

        Raises ValueError with the full list of valid codes when a dimension
        name or value is not found in the table.
        """
        meta = _fetch_metadata(self.table_id)
        valid_dims: dict[str, list[str]] = {
            v["code"]: v.get("values", [])
            for v in meta.get("variables", [])
        }

        for dim, vals in self.filters.items():
            if dim.lower() in ("tid", "time"):
                continue

            if dim not in valid_dims:
                raise ValueError(
                    f"SSB table {self.table_id}: dimension '{dim}' not found.\n"
                    f"  Valid dimensions: {list(valid_dims)}"
                )

            if vals == ["*"]:
                continue

            invalid = [v for v in vals if v not in valid_dims[dim]]
            if invalid:
                raise ValueError(
                    f"SSB table {self.table_id}, dimension '{dim}': "
                    f"value(s) {invalid} not found.\n"
                    f"  Valid values (first 30): {valid_dims[dim][:30]}"
                )

        # Warn about required dimensions that are missing from the filter
        for dim, allowed_vals in valid_dims.items():
            if dim.lower() in ("tid", "time"):
                continue
            if dim not in self.filters and len(allowed_vals) > 1:
                logger.warning(
                    "Table %s: dimension '%s' has %d values and is not filtered – "
                    "query will return all combinations. "
                    "Use python -m src.data.discover_api --table %s to inspect.",
                    self.table_id, dim, len(allowed_vals), self.table_id,
                )

    # ── Query building ──────────────────────────────────────────────────────────

    def _build_query(self) -> dict:
        return {
            "query": [
                {
                    "code": dim,
                    "selection": {
                        "filter": "all" if vals == ["*"] else "item",
                        "values": vals,
                    },
                }
                for dim, vals in self.filters.items()
            ],
            "response": {"format": "json-stat2"},
        }

    # ── Fetch / validate ────────────────────────────────────────────────────────

    def fetch(self) -> pd.DataFrame:
        self._validate_filters()
        url = f"{SSB_API_BASE}/{self.table_id}"
        query = self._build_query()
        logger.info("Fetching SSB table %s for variable '%s'.", self.table_id, self.variable_id)

        response = requests.post(url, json=query, timeout=60)
        response.raise_for_status()

        return _parse_jsonstat2(response.json())

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
    extracted as the date column; all other dimensions are expected to have
    size 1 after filtering (i.e. one selected value each). If a non-time
    dimension has size > 1 (e.g. an unfiltered Leveringssektor), only the
    first combination is used and a warning is logged.
    """
    dimensions = data["id"]          # e.g. ["Makrost", "ContentsCode", "Tid"]
    sizes = data["size"]             # e.g. [1, 1, 120]
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
        index_map = time_categories["index"]
        n_time = sizes[time_idx]
        inv = {v: k for k, v in index_map.items()}
        time_labels = [inv[i] for i in range(n_time)]
    else:
        time_labels = list(time_categories.get("label", {}).keys())

    n_time = len(time_labels)
    stride = 1
    for i in range(time_idx + 1, len(dimensions)):
        stride *= sizes[i]

    outer = 1
    for i in range(time_idx):
        outer *= sizes[i]

    if outer > 1:
        logger.warning(
            "JSON-stat2 response has %d outer combinations for non-time dimensions "
            "%s (sizes %s). Using only the first combination.",
            outer,
            [d for i, d in enumerate(dimensions) if i < time_idx],
            [s for i, s in enumerate(sizes) if i < time_idx],
        )

    # Extract values for the first outer combination only
    rows = []
    for t, label in enumerate(time_labels):
        val = values[t * stride]
        rows.append({"date_raw": label, "value": val})

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
