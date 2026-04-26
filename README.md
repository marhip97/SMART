# SMART – System for Model Analysis in Real Time

Et automatisert kryssjekkrammeverk for prognoser på sentrale norske makroøkonomiske variabler.

> **Disclaimer:** SMART er et eksperimentelt kryssjekkverktøy. Prognosene er ikke offisielle og skal ikke brukes som grunnlag for investerings- eller policybeslutninger.

---

## Hva er SMART?

SMART kjører flere modeller med ulike forutsetninger parallelt på samme variabler. Avvik mellom modellene er et signal i seg selv – og gir brukeren et bilde av hvor robust en prognose er.

**Målvariabler (v1):**
- BNP-vekst Fastlands-Norge
- KPI og KPI-JAE
- Registrert ledighet (NAV)
- Styringsrenten
- Lønnsvekst
- Boligprisvekst

**Modeller (v1):** ARIMA, VAR/BVAR, Dynamic Factor Model, AR-X, ML-baseline

**Prognosehorisont:** 1–3 år (årsvis veksttakt), kvantiler 10/50/90

---

## Repo-struktur

```
config/          Variabelkonfigurasjon (variables.yaml)
src/data/        Datapipeline (hentere, validering, lagring)
src/models/      Modellimplementasjoner
src/evaluation/  Backtesting og metrikker
src/ensemble/    Kryssjekk og aggregering
data/raw/        Rådata med vintage-tidsstempler (ikke versjonskontrollert)
data/processed/  Prosessert, modell-klar data (ikke versjonskontrollert)
tests/           Enhetstester og integrasjonstester
docs/            Modellkort og variabelbeskrivelser
dashboard/       GitHub Pages-frontend
notebooks/       Utforskning og EDA
```

---

## Kom i gang

```bash
pip install -r requirements-dev.txt
pytest tests/
python -m src.data.pipeline          # Hent alle variabler
python -m src.data.pipeline kpi bnp_fastland  # Hent spesifikke variabler
```

### Legge til en ny variabel

1. Legg til en ny blokk i `config/variables.yaml` (kopier en eksisterende).
2. Hvis kilden ikke finnes fra før, implementer en ny `DataSource`-subklasse i `src/data/` og registrer den i `_SOURCE_REGISTRY` i `pipeline.py`.
3. Kjør testene: `pytest tests/`

---

## Lisens

MIT License – se LICENSE-fil.

Kilde for data: SSB (NLOD 2.0), NAV (NLOD 2.0), Norges Bank (CC BY 4.0), FRED/Federal Reserve Bank of St. Louis.
