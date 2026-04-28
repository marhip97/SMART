"""SMART data pipeline – iterates over config/variables.yaml and runs each source."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import yaml

from .base import DataSource
from .fred import FREDDataSource
from .nav import NAVDataSource
from .norges_bank import NorgesBankDataSource
from .ssb import SSBDataSource

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parents[2] / "config" / "variables.yaml"

_SOURCE_REGISTRY: dict[str, type[DataSource]] = {
    "SSBDataSource":        SSBDataSource,
    "NAVDataSource":        NAVDataSource,
    "NorgesBankDataSource": NorgesBankDataSource,
    "FREDDataSource":       FREDDataSource,
}


def load_config(config_path: Path = CONFIG_PATH) -> list[dict[str, Any]]:
    """Load and return the list of variable definitions from variables.yaml."""
    with open(config_path, encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    return config["variables"]


def build_source(variable_cfg: dict[str, Any]) -> DataSource:
    """Instantiate the correct DataSource subclass for a variable config entry."""
    source_cls_name = variable_cfg["source"]
    if source_cls_name not in _SOURCE_REGISTRY:
        raise ValueError(
            f"Unknown source class '{source_cls_name}'. "
            f"Registered sources: {list(_SOURCE_REGISTRY)}"
        )
    cls = _SOURCE_REGISTRY[source_cls_name]
    return cls(
        variable_id=variable_cfg["id"],
        source_params=variable_cfg["source_params"],
    )


def run_pipeline(
    config_path: Path = CONFIG_PATH,
    variable_ids: list[str] | None = None,
) -> dict[str, str]:
    """Run the full data pipeline for all (or a subset of) variables.

    Args:
        config_path:   Path to variables.yaml.
        variable_ids:  If provided, only these variable IDs are processed.

    Returns:
        Dict mapping variable_id → outcome ("ok", "skipped", or error message).
    """
    variables = load_config(config_path)
    if variable_ids:
        variables = [v for v in variables if v["id"] in variable_ids]

    results: dict[str, str] = {}
    for var in variables:
        vid = var["id"]
        try:
            source = build_source(var)
            source.run()
            results[vid] = "ok"
            logger.info("✓ %s", vid)
        except Exception as exc:  # noqa: BLE001
            results[vid] = str(exc)
            logger.error("✗ %s: %s", vid, exc)

    n_ok = sum(1 for v in results.values() if v == "ok")
    n_fail = len(results) - n_ok
    logger.info("Pipeline complete: %d ok, %d failed.", n_ok, n_fail)
    return results


def main() -> None:
    """Entry point for CLI use: `python -m src.data.pipeline`."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    variable_ids = sys.argv[1:] or None
    results = run_pipeline(variable_ids=variable_ids)
    failed = {k: v for k, v in results.items() if v != "ok"}
    if failed:
        logger.error("Failed variables: %s", failed)
    # Exit 1 only if NOTHING succeeded – a partial run should not block the model runner.
    if not any(v == "ok" for v in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
