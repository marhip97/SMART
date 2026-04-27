"""Diagnostics – print available dimension codes for SSB tables and Norges Bank dataflows.

Run manually to find the correct filter values for config/variables.yaml:

    python -m scripts.discover_api
"""

from __future__ import annotations

import requests

SSB_TABLES = {
    "bnp_fastland":   "09190",
    "kpi_jae":        "10235",
    "ledighet_nav":   "05111",
    "lonnsvekst":     "11417",
    "boligprisvekst": "07230",
}

NB_URLS = {
    "styringsrente":      "https://data.norges-bank.no/api/data/IR/B.SIREN.SR.D?format=sdmx-json&startPeriod=2020-01-01",
    "eurnok":             "https://data.norges-bank.no/api/data/EXR/B.EUR.NOK.SP.A?format=sdmx-json&startPeriod=2020-01-01",
    "handelspartnervekst":"https://data.norges-bank.no/api/data/MPM/TPGDP_Q?format=sdmx-json&startPeriod=2020-01-01",
    "k2_kredittvekst":    "https://data.norges-bank.no/api/data/CR/K2.H.12M.NOK?format=sdmx-json&startPeriod=2020-01-01",
}


def check_ssb_tables() -> None:
    print("\n═══ SSB table metadata ═══")
    for var, table_id in SSB_TABLES.items():
        url = f"https://data.ssb.no/api/v0/no/table/{table_id}"
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            meta = r.json()
        except Exception as exc:
            print(f"\n[{var} – table {table_id}]  ERROR: {exc}")
            continue

        print(f"\n[{var} – table {table_id}]  title: {meta.get('title','')}")
        for var_meta in meta.get("variables", []):
            dim_id = var_meta.get("code", var_meta.get("id", "?"))
            vals = var_meta.get("values", [])
            texts = var_meta.get("valueTexts", vals)
            pairs = list(zip(vals[:10], texts[:10]))
            print(f"  {dim_id}: {pairs}{'...' if len(vals) > 10 else ''}")


def check_nb_series() -> None:
    print("\n═══ Norges Bank series (small sample) ═══")
    for var, url in NB_URLS.items():
        try:
            r = requests.get(url, timeout=30)
            status = r.status_code
            print(f"\n[{var}]  {status}  {url}")
            if status == 200:
                data = r.json()
                # Print top-level keys to understand structure
                top = list(data.keys())
                print(f"  top-level keys: {top}")
            else:
                print(f"  body: {r.text[:300]}")
        except Exception as exc:
            print(f"\n[{var}]  ERROR: {exc}")


if __name__ == "__main__":
    check_ssb_tables()
    check_nb_series()
    print("\nDone.")
