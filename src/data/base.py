"""Abstract base class for all data sources in SMART."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

RAW_DATA_DIR = Path(__file__).parents[2] / "data" / "raw"
PROCESSED_DATA_DIR = Path(__file__).parents[2] / "data" / "processed"


class DataSource(ABC):
    """Base class every data source must subclass.

    Subclasses implement fetch() and validate(); store() and
    load_latest() are provided here and should not be overridden.
    """

    def __init__(self, variable_id: str, source_params: dict[str, Any]) -> None:
        self.variable_id = variable_id
        self.source_params = source_params

    @abstractmethod
    def fetch(self) -> pd.DataFrame:
        """Fetch raw data from the source.

        Returns:
            DataFrame with at minimum columns ['date', 'value'].
            'date' must be a pandas Timestamp; 'value' must be numeric.
        """

    @abstractmethod
    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate the fetched DataFrame.

        Raises:
            ValueError: if the data fails validation checks.

        Returns:
            The same DataFrame (possibly with minor cleaning applied).
        """

    def store(self, df: pd.DataFrame) -> Path:
        """Store raw data as Parquet with a vintage timestamp.

        Files are written to data/raw/<variable_id>/<YYYY-MM-DD>.parquet.
        Historical files are never overwritten.

        Returns:
            Path to the written file.
        """
        vintage = datetime.utcnow().strftime("%Y-%m-%d")
        out_dir = RAW_DATA_DIR / self.variable_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{vintage}.parquet"

        if out_path.exists():
            logger.info("Vintage %s for %s already stored; skipping.", vintage, self.variable_id)
            return out_path

        df.to_parquet(out_path, index=False)
        logger.info("Stored %d rows for %s → %s", len(df), self.variable_id, out_path)
        return out_path

    def load_latest(self) -> pd.DataFrame | None:
        """Load the most recently stored raw vintage, or None if none exist."""
        raw_dir = RAW_DATA_DIR / self.variable_id
        if not raw_dir.exists():
            return None
        files = sorted(raw_dir.glob("*.parquet"))
        if not files:
            return None
        return pd.read_parquet(files[-1])

    def run(self) -> pd.DataFrame:
        """Convenience method: fetch → validate → store → return DataFrame."""
        df = self.fetch()
        df = self.validate(df)
        self.store(df)
        return df


def _assert_columns(df: pd.DataFrame, required: list[str]) -> None:
    """Raise ValueError if any required columns are missing."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")


def _assert_no_nulls(df: pd.DataFrame, columns: list[str]) -> None:
    """Raise ValueError if any of the given columns contain null values."""
    for col in columns:
        null_count = df[col].isna().sum()
        if null_count > 0:
            raise ValueError(f"Column '{col}' has {null_count} null value(s).")
