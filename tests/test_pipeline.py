"""Unit tests for the pipeline runner."""

import pandas as pd
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from src.data.pipeline import build_source, load_config, run_pipeline


def test_load_config_returns_list(tmp_path):
    yaml_content = """
variables:
  - id: test_var
    name: Test
    type: target
    unit: pct
    source: SSBDataSource
    frequency: monthly
    transform: yoy_pct
    source_params:
      table_id: "12345"
      filters: {}
"""
    cfg_file = tmp_path / "variables.yaml"
    cfg_file.write_text(yaml_content)
    variables = load_config(cfg_file)
    assert len(variables) == 1
    assert variables[0]["id"] == "test_var"


def test_build_source_unknown_raises():
    with pytest.raises(ValueError, match="Unknown source"):
        build_source({"id": "x", "source": "NoSuchSource", "source_params": {}})


def _mock_source_registry(mock_df=None, side_effect=None):
    """Return a patched _SOURCE_REGISTRY where every class is a MagicMock."""
    instance = MagicMock()
    if side_effect:
        instance.run.side_effect = side_effect
    else:
        instance.run.return_value = mock_df or pd.DataFrame(
            {"date": pd.to_datetime(["2023-01-01"]), "value": [1.0]}
        )
    MockCls = MagicMock(return_value=instance)
    registry = {
        "FREDDataSource": MockCls,
        "SSBDataSource": MockCls,
        "NAVDataSource": MockCls,
        "NorgesBankDataSource": MockCls,
    }
    return registry, instance


def test_run_pipeline_calls_run_on_each_variable(tmp_path):
    yaml_content = """
variables:
  - id: var_a
    name: A
    type: target
    unit: pct
    source: FREDDataSource
    frequency: monthly
    transform: level
    source_params:
      series_id: "FAKE_A"
  - id: var_b
    name: B
    type: target
    unit: pct
    source: FREDDataSource
    frequency: monthly
    transform: level
    source_params:
      series_id: "FAKE_B"
"""
    cfg_file = tmp_path / "variables.yaml"
    cfg_file.write_text(yaml_content)

    registry, instance = _mock_source_registry()
    with patch("src.data.pipeline._SOURCE_REGISTRY", registry):
        results = run_pipeline(config_path=cfg_file)

    assert results["var_a"] == "ok"
    assert results["var_b"] == "ok"
    assert instance.run.call_count == 2


def test_run_pipeline_captures_errors(tmp_path):
    yaml_content = """
variables:
  - id: bad_var
    name: Bad
    type: target
    unit: pct
    source: FREDDataSource
    frequency: monthly
    transform: level
    source_params:
      series_id: "FAKE"
"""
    cfg_file = tmp_path / "variables.yaml"
    cfg_file.write_text(yaml_content)

    registry, _ = _mock_source_registry(side_effect=RuntimeError("network error"))
    with patch("src.data.pipeline._SOURCE_REGISTRY", registry):
        results = run_pipeline(config_path=cfg_file)

    assert "network error" in results["bad_var"]


def test_run_pipeline_filters_by_variable_ids(tmp_path):
    yaml_content = """
variables:
  - id: var_a
    name: A
    type: target
    unit: pct
    source: FREDDataSource
    frequency: monthly
    transform: level
    source_params:
      series_id: "A"
  - id: var_b
    name: B
    type: target
    unit: pct
    source: FREDDataSource
    frequency: monthly
    transform: level
    source_params:
      series_id: "B"
"""
    cfg_file = tmp_path / "variables.yaml"
    cfg_file.write_text(yaml_content)

    registry, _ = _mock_source_registry()
    with patch("src.data.pipeline._SOURCE_REGISTRY", registry):
        results = run_pipeline(config_path=cfg_file, variable_ids=["var_a"])

    assert "var_a" in results
    assert "var_b" not in results
