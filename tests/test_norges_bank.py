"""Unit tests for NorgesBankDataSource using mocked HTTP responses."""

import pytest
import responses as resp_lib

from src.data.norges_bank import NorgesBankDataSource, _parse_sdmx_json
import pandas as pd


def _sdmx_payload(periods: list[str], values: list[float]) -> dict:
    """Build a minimal SDMX-JSON payload."""
    return {
        "data": {
            "dataSets": [
                {
                    "series": {
                        "0:0:0:0": {
                            "observations": {
                                str(i): [v] for i, v in enumerate(values)
                            }
                        }
                    }
                }
            ],
            "structure": {
                "dimensions": {
                    "observation": [
                        {
                            "id": "TIME_PERIOD",
                            "values": [{"id": p} for p in periods],
                        }
                    ]
                }
            },
        }
    }


def test_parse_sdmx_json_basic():
    payload = _sdmx_payload(["2023-01", "2023-02", "2023-03"], [3.0, 3.25, 3.5])
    df = _parse_sdmx_json(payload)
    assert list(df.columns) == ["date", "value"]
    assert len(df) == 3
    assert df["value"].iloc[2] == pytest.approx(3.5)


def test_unknown_series_raises():
    with pytest.raises(ValueError, match="Unknown"):
        NorgesBankDataSource("bad", {"series": "NONEXISTENT"})


def test_validate_ok():
    source = NorgesBankDataSource("styringsrente", {"series": "SIREN"})
    df = pd.DataFrame({
        "date": pd.to_datetime(["2023-01-01", "2023-02-01"]),
        "value": [3.0, 3.25],
    })
    result = source.validate(df)
    assert len(result) == 2


def test_validate_empty_raises():
    source = NorgesBankDataSource("styringsrente", {"series": "SIREN"})
    with pytest.raises(ValueError, match="Empty"):
        source.validate(pd.DataFrame({"date": [], "value": []}))


@resp_lib.activate
def test_fetch_parses_sdmx():
    payload = _sdmx_payload(["2023-01", "2023-02"], [3.0, 3.25])
    resp_lib.add(
        resp_lib.GET,
        "https://data.norges-bank.no/api/data/SHORT_RATES/B.SIGHT_DEP_RATE.NOK.D",
        json=payload,
        status=200,
    )
    source = NorgesBankDataSource("styringsrente", {"series": "SIREN"})
    df = source.fetch()
    assert list(df.columns) == ["date", "value"]
    assert len(df) == 2
