"""Unit tests for FREDDataSource using mocked HTTP responses."""

import pandas as pd
import pytest
import responses as resp_lib

from src.data.fred import FREDDataSource

FRED_CSV_OK = "DATE,DCOILBRENTEU\n2023-01-01,80.5\n2023-02-01,82.1\n2023-03-01,.\n2023-04-01,79.9\n"


@resp_lib.activate
def test_fetch_parses_csv_and_handles_missing():
    resp_lib.add(
        resp_lib.GET,
        "https://fred.stlouisfed.org/graph/fredgraph.csv",
        body=FRED_CSV_OK,
        status=200,
        content_type="text/csv",
    )
    source = FREDDataSource("oljepris", {"series_id": "DCOILBRENTEU"})
    df = source.fetch()
    assert list(df.columns) == ["date", "value"]
    assert len(df) == 4
    assert pd.isna(df.loc[df["date"] == pd.Timestamp("2023-03-01"), "value"].iloc[0])
    assert df.loc[df["date"] == pd.Timestamp("2023-01-01"), "value"].iloc[0] == pytest.approx(80.5)


def test_validate_ok():
    source = FREDDataSource("oljepris", {"series_id": "DCOILBRENTEU"})
    df = pd.DataFrame({"date": pd.to_datetime(["2023-01-01", "2023-02-01"]), "value": [80.5, 82.1]})
    result = source.validate(df)
    assert len(result) == 2


def test_validate_empty_raises():
    source = FREDDataSource("oljepris", {"series_id": "DCOILBRENTEU"})
    with pytest.raises(ValueError, match="Empty"):
        source.validate(pd.DataFrame({"date": [], "value": []}))


def test_validate_too_many_nulls_raises():
    source = FREDDataSource("oljepris", {"series_id": "DCOILBRENTEU"})
    df = pd.DataFrame({
        "date": pd.to_datetime(["2023-01-01", "2023-02-01", "2023-03-01"]),
        "value": [None, None, 80.0],
    })
    with pytest.raises(ValueError, match="null"):
        source.validate(df)
