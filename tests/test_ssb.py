"""Unit tests for SSBDataSource using mocked HTTP responses."""

import pandas as pd
import pytest
import responses as resp_lib

from src.data.ssb import SSBDataSource, _parse_ssb_date, _parse_jsonstat2

# ── _parse_ssb_date ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw,expected", [
    ("2023K1",  pd.Timestamp("2023-01-01")),
    ("2023K2",  pd.Timestamp("2023-04-01")),
    ("2023K3",  pd.Timestamp("2023-07-01")),
    ("2023K4",  pd.Timestamp("2023-10-01")),
    ("2023M01", pd.Timestamp("2023-01-01")),
    ("2023M12", pd.Timestamp("2023-12-01")),
    ("2023",    pd.Timestamp("2023-01-01")),
    ("bad",     None),
])
def test_parse_ssb_date(raw, expected):
    assert _parse_ssb_date(raw) == expected


# ── SSBDataSource.validate ─────────────────────────────────────────────────────

def _make_df(dates, values):
    return pd.DataFrame({"date": pd.to_datetime(dates), "value": values})


def test_validate_ok():
    source = SSBDataSource("kpi", {"table_id": "03013", "filters": {}})
    df = _make_df(["2022-01-01", "2022-02-01"], [3.5, 4.1])
    result = source.validate(df)
    assert len(result) == 2


def test_validate_empty_raises():
    source = SSBDataSource("kpi", {"table_id": "03013", "filters": {}})
    with pytest.raises(ValueError, match="Empty"):
        source.validate(pd.DataFrame({"date": [], "value": []}))


def test_validate_too_many_nulls_raises():
    source = SSBDataSource("kpi", {"table_id": "03013", "filters": {}})
    df = _make_df(
        ["2022-01-01", "2022-02-01", "2022-03-01"],
        [None, None, 3.5],  # 67% null → exceeds 10% threshold
    )
    with pytest.raises(ValueError, match="null"):
        source.validate(df)


# ── SSBDataSource.fetch (mocked HTTP) ─────────────────────────────────────────

def _minimal_jsonstat2(time_values, data_values):
    """Build a minimal JSON-stat2 payload with a single dimension 'Tid'."""
    return {
        "class": "dataset",
        "label": "Test",
        "id": ["Tid"],
        "size": [len(time_values)],
        "dimension": {
            "Tid": {
                "label": "tid",
                "category": {
                    "index": {v: i for i, v in enumerate(time_values)},
                    "label": {v: v for v in time_values},
                },
            }
        },
        "value": data_values,
    }


def test_parse_jsonstat2_single_dimension():
    payload = _minimal_jsonstat2(["2022K1", "2022K2"], [1.1, 1.5])
    df = _parse_jsonstat2(payload)
    assert list(df.columns) == ["date", "value"]
    assert len(df) == 2
    assert df["date"].iloc[0] == pd.Timestamp("2022-01-01")
    assert df["value"].iloc[1] == pytest.approx(1.5)


def test_parse_jsonstat2_two_dimensions():
    """Simulate a two-dimension response (ContentsCode × Tid) with one contents value."""
    payload = {
        "id": ["ContentsCode", "Tid"],
        "size": [1, 3],
        "dimension": {
            "ContentsCode": {
                "category": {"index": {"BNPB": 0}, "label": {"BNPB": "BNP"}}
            },
            "Tid": {
                "category": {
                    "index": {"2022K1": 0, "2022K2": 1, "2022K3": 2},
                    "label": {"2022K1": "2022K1", "2022K2": "2022K2", "2022K3": "2022K3"},
                }
            },
        },
        "value": [1.0, 2.0, 3.0],
    }
    df = _parse_jsonstat2(payload)
    assert len(df) == 3
    assert df["value"].tolist() == pytest.approx([1.0, 2.0, 3.0])


@resp_lib.activate
def test_fetch_parses_quarterly():
    payload = _minimal_jsonstat2(
        ["2022K1", "2022K2", "2022K3", "2022K4"],
        [1.1, 1.5, 2.0, 2.3],
    )
    resp_lib.add(
        resp_lib.POST,
        "https://data.ssb.no/api/v0/no/table/09190",
        json=payload,
        status=200,
    )
    source = SSBDataSource(
        "bnp_fastland",
        {"table_id": "09190", "filters": {"ContentsCode": ["BNPB"], "Tid": ["*"]}},
    )
    df = source.fetch()
    assert list(df.columns) == ["date", "value"]
    assert len(df) == 4
    assert df["date"].iloc[0] == pd.Timestamp("2022-01-01")
    assert df["value"].iloc[1] == pytest.approx(1.5)


@resp_lib.activate
def test_fetch_http_error_raises():
    resp_lib.add(
        resp_lib.POST,
        "https://data.ssb.no/api/v0/no/table/99999",
        status=404,
    )
    source = SSBDataSource("bad", {"table_id": "99999", "filters": {}})
    with pytest.raises(Exception):
        source.fetch()
