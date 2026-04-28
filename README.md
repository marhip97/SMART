# SMART – System for Model Analysis in Real Time

Et automatisert kryssjekkrammeverk for prognoser på sentrale norske makroøkonomiske variabler.

> **Disclaimer:** SMART er et eksperimentelt kryssjekkverktøy. Prognosene er ikke offisielle og skal ikke brukes som grunnlag for investerings- eller policybeslutninger.

**Dashboard:** https://marhip97.github.io/SMART/

---

## Hva er SMART?

SMART kjører flere modeller med ulike forutsetninger parallelt på samme variabler. Avvik mellom modellene er et signal i seg selv – og gir brukeren et bilde av hvor robust en prognose er.

**Dashboardet viser:**
- Historiske observasjoner og ensemble-prognose med usikkerhetsintervall (10.–90. persentil)
- Enkeltmodellers prognoser og historiske walk-forward-prediksjoner
- Treffsikkerhetstabell per modell: RMSE, MAE, R² (out-of-sample) og ensemblevekt
- Modellsprik per prognoseår med varselflagg ved høy uenighet

**Målvariabler (v1):**
- BNP-vekst Fastlands-Norge (SSB)
- KPI og KPI-JAE (SSB)
- AKU-arbeidsledighet (SSB, 15–74 år)
- Styringsrenten (Norges Bank)
- Lønnsvekst (SSB)
- Boligprisvekst (SSB)

**Betingingsvariabler:** Oljepris (FRED), EUR/NOK (Norges Bank), handelspartnervekst (FRED), ECB-rente (FRED), K2-kredittvekst (SSB)

**Modeller (v1):** ARIMA, VAR/BVAR, Dynamic Factor Model, AR-X, ML-baseline

**Prognosehorisont:** 1–3 år (årsvis veksttakt), kvantiler 10/50/90

---

## Repo-struktur

```
config/          Variabel- og modellkonfigurasjon (variables.yaml, models.yaml)
src/data/        Datapipeline (hentere, validering, lagring)
src/models/      Modellimplementasjoner
src/evaluation/  Backtesting og metrikker
src/ensemble/    Kryssjekk og aggregering
src/runner.py    Orkestrerer modellkjøring og produserer prognose-JSON
data/raw/        Rådata med vintage-tidsstempler (parquet, versjonskontrollert)
data/processed/  Prognoseresultater og manifest (JSON, versjonskontrollert)
tests/           Enhetstester og integrasjonstester
docs/            Modellkort og variabelbeskrivelser
dashboard/       GitHub Pages-frontend (Plotly.js, statisk HTML/CSS/JS)
scripts/         Hjelpeskript (f.eks. API-oppdagelse)
```

---

## Kom i gang

```bash
pip install -r requirements-dev.txt
pytest tests/

python -m src.data.pipeline          # Hent alle variabler
python -m src.data.pipeline kpi bnp_fastland  # Hent spesifikke variabler
python -m src.runner                  # Kjør modeller og generer prognose-JSON
```

### Legge til en ny variabel

1. Legg til en ny blokk i `config/variables.yaml` (kopier en eksisterende).
2. Hvis kilden ikke finnes fra før, implementer en ny `DataSource`-subklasse i `src/data/` og registrer den i `_SOURCE_REGISTRY` i `pipeline.py`.
3. Kjør testene: `pytest tests/`

### Forecast-JSON-struktur

Hvert kjørt variabel lagres som `data/processed/forecasts/<variabel>/latest.json` med feltene:

| Felt | Innhold |
|---|---|
| `history` | Siste 20 år med faktiske observasjoner (transformert til modellformat) |
| `ensemble.forecasts` | Ensemble q10/q50/q90 per prognoseår |
| `ensemble.weights` | Modellvekter (inversely proportional til RMSE) |
| `models` | Per-modell q10/q50/q90 per prognoseår |
| `evaluation` | RMSE, MAE, R², antall obs og backtest-historikk per modell |
| `disagreement` | Sprik og standardavvik mellom modellene per prognoseår |

### Oppdatering av prognosedata

GitHub Actions kjører automatisk hver mandag kl. 06:00 UTC:
1. **fetch-data** – henter nye datapunkter fra alle 12 variabler og committer vintager til `data/raw/`
2. **run-models** – kjører alle modeller og ensemble, lagrer prognose-JSON til `data/processed/forecasts/`
3. **deploy_dashboard** – deployer oppdatert dashboard til GitHub Pages

---

## Lisens

MIT License – se LICENSE-fil.

Datakilder: SSB (NLOD 2.0), Norges Bank (CC BY 4.0), FRED/Federal Reserve Bank of St. Louis (ikke-kommersiell bruk).
