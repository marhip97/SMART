"""Diagnostics – print available dimension codes for SSB tables and Norges Bank dataflows.

Run manually to find the correct filter values for config/variables.yaml:

    python -m scripts.discover_api
"""

from __future__ import annotations

import requests

SSB_API_BASE = "https://data.ssb.no/api/v0/no/table"
NBD_BASE = "https://data.norges-bank.no/api/data"

SSB_TABLES = {
    "bnp_fastland":   "09190",
    "kpi_jae":        "10235",
    "ledighet_nav":   "05111",
    "lonnsvekst":     "11417",
    "boligprisvekst": "07230",
}

# Candidate URLs to probe for each Norges Bank variable.
# Try broad (no key filter) first, then specific keys.
NB_PROBE: dict[str, list[str]] = {
    "styringsrente": [
        f"{NBD_BASE}/SHORT_RATES?format=sdmx-json&startPeriod=2024-01-01",
        f"{NBD_BASE}/SHORT_RATES/B.SIGHT_DEP_RATE.NOK.D?format=sdmx-json&startPeriod=2024-01-01",
        f"{NBD_BASE}/SHORT_RATES/D.RNBO.?format=sdmx-json&startPeriod=2024-01-01",
    ],
    "eurnok": [
        f"{NBD_BASE}/EXR/B.EUR.NOK.SP?format=sdmx-json&startPeriod=2024-01-01",
        f"{NBD_BASE}/EXR/B.EUR.NOK.SP.A?format=sdmx-json&startPeriod=2024-01-01",
        f"{NBD_BASE}/EXR?format=sdmx-json&startPeriod=2024-01-01",
    ],
    "handelspartnervekst": [
        f"{NBD_BASE}/MPM/TPGDP_Q?format=sdmx-json&startPeriod=2020-01-01",
        f"{NBD_BASE}/MPM?format=sdmx-json&startPeriod=2020-01-01",
    ],
    "k2_kredittvekst": [
        f"{NBD_BASE}/CR/K2.H.12M.NOK?format=sdmx-json&startPeriod=2020-01-01",
        f"{NBD_BASE}/CR?format=sdmx-json&startPeriod=2020-01-01",
    ],
}


def check_ssb_tables() -> None:
    print("\n═══ SSB table metadata (dimension codes) ═══")
    for var, table_id in SSB_TABLES.items():
        url = f"{SSB_API_BASE}/{table_id}"
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            meta = r.json()
        except Exception as exc:
            print(f"\n[{var} – table {table_id}]  ERROR: {exc}")
            continue

        print(f"\n[{var} – table {table_id}]  {meta.get('title', '')}")
        for var_meta in meta.get("variables", []):
            dim_id = var_meta.get("code", var_meta.get("id", "?"))
            vals = var_meta.get("values", [])
            texts = var_meta.get("valueTexts", vals)
            pairs = list(zip(vals[:12], texts[:12]))
            suffix = "..." if len(vals) > 12 else ""
            print(f"  {dim_id}: {pairs}{suffix}")


def check_nb_series() -> None:
    print("\n═══ Norges Bank series probes ═══")
    for var, urls in NB_PROBE.items():
        print(f"\n[{var}]")
        for url in urls:
            try:
                r = requests.get(url, timeout=30)
                status = r.status_code
                if status == 200:
                    data = r.json()
                    # Try to extract series keys from SDMX-JSON structure
                    try:
                        series = data["data"]["dataSets"][0]["series"]
                        series_keys = list(series.keys())[:5]
                        dims = data["data"]["structure"]["dimensions"].get("series", [])
                        dim_ids = [d["id"] for d in dims]
                        print(f"  ✓ {status}  {url}")
                        print(f"    dimensions: {dim_ids}")
                        print(f"    series keys (first 5): {series_keys}")
                    except (KeyError, IndexError, TypeError):
                        print(f"  ✓ {status}  {url}  (unexpected structure)")
                    break  # Found a working URL
                else:
                    print(f"  ✗ {status}  {url}")
            except Exception as exc:
                print(f"  ✗ ERROR  {url}  {exc}")


if __name__ == "__main__":
    check_ssb_tables()
    check_nb_series()
    print("\nDone.")
