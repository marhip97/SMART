"""SSB table metadata inspector.

Prints the title, dimension codes, and available values for an SSB table.
Use this to find correct filter values for config/variables.yaml.

Usage:
    python -m src.data.discover_api --table 09190
    python -m src.data.discover_api --table 14702
"""

from __future__ import annotations

import argparse

import requests

SSB_API_BASE = "https://data.ssb.no/api/v0/no/table"


def inspect_table(table_id: str, max_values: int = 20) -> None:
    url = f"{SSB_API_BASE}/{table_id}"
    resp = requests.get(url, timeout=30)
    if resp.status_code != 200:
        print(f"ERROR {resp.status_code}: {resp.text[:200]}")
        return

    meta = resp.json()
    print(f"\nTable {table_id}: {meta.get('title', '(no title)')}")
    print(f"Updated: {meta.get('updated', '?')}")
    print()

    for var in meta.get("variables", []):
        code = var.get("code", var.get("id", "?"))
        text = var.get("text", "")
        vals = var.get("values", [])
        texts = var.get("valueTexts", vals)

        print(f"  Dimension: {code!r}  ← text: {text!r}")
        for val, txt in zip(vals[:max_values], texts[:max_values]):
            print(f"    {val!r}: {txt!r}")
        if len(vals) > max_values:
            print(f"    ... ({len(vals)} values total)")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect SSB table dimensions and valid filter values."
    )
    parser.add_argument("--table", required=True, help="SSB table ID, e.g. 09190")
    parser.add_argument(
        "--max-values", type=int, default=20,
        help="Max number of values to print per dimension (default: 20)",
    )
    args = parser.parse_args()
    inspect_table(args.table, args.max_values)


if __name__ == "__main__":
    main()
