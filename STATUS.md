# STATUS – SMART

Levende status- og fremdriftsprotokoll. Oppdateres av prosjektleder (Claude Code) løpende.

**Format:** Nyeste oppføring øverst. Bruk dato (YYYY-MM-DD) og kortfattede punkter.

---

## Gjeldende fase
**M5 – QA, dokumentasjon, lansering** (M4 fullført 2026-04-27)

## M0-beslutninger (vedtatt 2026-04-26)

| # | Tema | Beslutning |
|---|---|---|
| 1 | **Variabler v1** | BNP-vekst Fastlands-Norge, KPI, KPI-JAE, registrert ledighet, styringsrente, lønnsvekst, boligprisvekst (7 variabler) |
| 2 | **Modeller v1** | ARIMA, VAR/BVAR, Dynamic Factor Model, AR-X, ML-baseline (random forest/gradient boosting) |
| 3 | **Prognosehorisont** | 1–3 år (årsvis veksttakt for variabelen), kvantiler 10/50/90 |
| 4 | **Oppdateringsfrekvens** | Ukentlig pipeline-kjøring |
| 5 | **Kodelisens** | MIT |
| 6 | **Betingingsvariabler** | Oljepris (Brent/FRED), EUR/NOK (Norges Bank), handelspartnervekst (Norges Bank/IMF), ECB-rente (FRED), K2-kredittvekst (Norges Bank) |
| 7 | **Fleksibilitet** | All variabeldefinisjon i `config/variables.yaml` – ny variabel legges til med YAML-oppføring, ingen kodeendring |

## Til avklaring (åpne spørsmål til prosjekteier)

*(Ingen åpne avklaringer.)*

## Risikoer som har materialisert seg
*(Ingen registrert ennå.)*

---

## Logg

### 2026-04-28 – .gitignore-fix: pipeline kan nå committe datavintagé
- Fjernet `data/raw/`, `data/processed/` og `*.parquet` fra `.gitignore` –
  disse blokkerte `git add data/raw/` i CI og hindret vintagelagring.
- 12/12 variabler hentes uten feil (bekreftet i forrige sesjon).
- **kpi_jae endelig løsning**: tabell 05327, `Konsumgrp: ["JAE_TOTAL"]`, `ContentsCode: ["Tolvmanedersendring"]`.
- Branch `claude/review-smart-project-yLCxR` klar for merge til main.
- **Neste**: merge til main → CI kjører automatisk → dashboard deployes til GitHub Pages.

### 2026-04-27 – M4 fullført: Dashboard v1
- `dashboard/index.html`, `dashboard/style.css`, `dashboard/app.js`: statisk Plotly-dashboard med
  - Vifte-diagram (q10/q50/q90 ensemble-prognose) per variabel
  - Modell-sammenligningsgraf (individuelle medialinjer)
  - Ensemble-vekttabell med stolpediagram
  - Modellspriktabell med flagg for høy uenighet
  - Responsivt mobiloppsett
- `.github/workflows/deploy_dashboard.yml`: GitHub Pages-deployment via `actions/deploy-pages`
  - Kopierer `dashboard/` og `data/processed/forecasts/` til statisk site-rot
  - Triggeres automatisk etter vellykket Data Pipeline-kjøring og ved push
- **kpi_jae-fix**: `Konsumgrp: ["KPI-JAE"]` → `Konsumgrp: ["KPIJAE"]` (se Til avklaring)
- 122/122 tester grønt
- **Aktiveringsoppgave for prosjekteier**: Gå til GitHub → Settings → Pages → Source: GitHub Actions
- **Neste**: M5 – QA, dokumentasjon, lansering

### 2026-04-27 – Orkestrering: modellkjørings-pipeline
- `src/runner.py`: leser siste datavintagé → kjører alle modeller → ensemble → lagrer JSON til `data/processed/forecasts/<var>/latest.json`
- `data/processed/forecasts/manifest.json` oppdateres hver kjøring
- `.github/workflows/data_pipeline.yml` utvidet med jobb `run-models` som kjøres etter `fetch-data`
- 20 nye tester (113/113 totalt), ruff ren

### 2026-04-27 – M3 fullført: kryssjekk og ensemble
- `src/ensemble/disagreement.py`: `DisagreementReport` + `compute_disagreement()` + `disagreement_to_dataframe()`
- `src/ensemble/forecaster.py`: `EnsembleForecaster` med tre vektingsordninger (equal, performance, trimmed) + `_apply_weight_floor()` (iterativ redistribuering)
- Konservative kvantilgrenser: q10 = 10. persentil av modell-q10s, q90 = 90. persentil av modell-q90s
- 24 nye tester, 93/93 grønt totalt, ruff ren
- **Neste:** M4 – Dashboard v1

### 2026-04-26 – M3 oppstart: kryssjekk og ensemble
- M2 PR merget til main. Branch synkronisert.
- Starter implementering av ensemble-lag: avviksmål, vekting, aggregert prognose.

### 2026-04-26 – M2 tillegg: BVAR med Minnesota-prior
- BVARModel implementert fra scratch (numpy/scipy, ingen nye avhengigheter)
- Minnesota-prior: eget-lag-varians > kryss-lag-varians, shrinkage skalerer med 1/lag
- Posterior analytisk beregnet (betinget på OLS Σ); MC-trekk for kvantiler
- 15 nye tester (inkl. korrekt prior-retning, reproduserbarhet, stramhets-monotonicitet)
- 69/69 tester grønt totalt
- docs/models/bvar.md opprettet med referanser til Litterman (1986)

### 2026-04-26 – M2 fullført: modeller v1
- Abstrakt `BaseModel`-grensesnitt med `fit` / `predict` / `evaluate` og standardisert `ForecastResult` / `EvaluationResult`
- `config/models.yaml`: alle 5 modeller konfigurert
- Implementert: `ARIMAModel`, `VARModel` (m/ AR-fallback for enkelt-serie), `DFMModel`, `ARXModel`, `MLBaselineModel`
- Walk-forward backtesting i `src/models/utils.py` (deles av alle modeller)
- 54/54 tester grønt (inkl. 25 nye modell-tester på syntetisk data), ruff ren
- Modellkort for alle 5 modeller i `docs/models/`
- Viktige designvalg: VAR degraderer til AR for enkelt-variable; DFM bruker statsmodels DynamicFactorMQ; ML-baseline bruker approksimativ kvantilestimering via trefordeling
- **Neste:** M3 – kryssjekk og ensemble (avviksmål, vekting, aggregert prognose)

### 2026-04-26 – M2 oppstart: modellimplementasjon
- M1 PR merget til main. Branch synkronisert.
- Starter implementering av modell-lag: abstrakt grensesnitt, ARIMA, VAR, DFM, AR-X, ML-baseline.

### 2026-04-26 – M1 fullført: datapipeline
- Repo-struktur opprettet (`src/data/`, `config/`, `tests/`, `docs/`, `data/`, `.github/workflows/`)
- `config/variables.yaml`: alle 12 variabler definert (7 målvariabler + 5 betingingsvariabler)
- Abstrakt `DataSource`-baseklasse med `fetch`, `validate`, `store`, `run`
- Implementerte hentere: `SSBDataSource` (JSON-stat2, ingen pyjstat-avh.), `NAVDataSource`, `NorgesBankDataSource` (SDMX-JSON), `FREDDataSource` (CSV)
- `pipeline.py`: itererer over YAML-config – ny variabel krever kun YAML-oppføring
- 29/29 enhetstester grønt (pytest, mocked HTTP via `responses`)
- GitHub Actions: `data_pipeline.yml` (ukentlig mandag 06:00 UTC) + `tests.yml` (CI på push/PR)
- Lisensvilkår verifisert for alle kilder: SSB/NAV (NLOD 2.0), Norges Bank (CC BY 4.0), FRED (ikke-kommersiell)
- `docs/data-sources.md` og 12 variabelbeskrivelser under `docs/variables/`
- `README.md` og `LICENSE` (MIT) opprettet
- Viktig: `pyjstat` ikke tilgjengelig i miljøet – erstattet med innebygd JSON-stat2-parser
- **Neste:** M2 – modellimplementasjon (ARIMA, VAR/BVAR, DFM, AR-X, ML-baseline)

### 2026-04-26 – M0 tilleggsavklaring: betingingsvariabler og fleksibilitet
- Vedtatt å skille mellom målvariabler (7 stk.) og betingingsvariabler (eksogene).
- Betingingsvariabler inn i M1-pipeline fra start: oljepris, EUR/NOK, handelspartnervekst, ECB-rente, K2-kredittvekst.
- Arkitekturprinsipp bekreftet: all variabeldefinisjon i `config/variables.yaml` – ny variabel krever bare YAML-oppføring.
- M0 endelig lukket. Starter M1-implementering.

### 2026-04-26 – M0 fullført, M1 planlegging
- Mottok svar fra prosjekteier på alle fem avklaringspunkter.
- M0-beslutninger dokumentert over.
- Viktigste justeringer fra opprinnelig forslag:
  - Variabler: registrert ledighet (NAV) i stedet for AKU-ledighet (SSB); lagt til lønnsvekst og boligprisvekst; EUR/NOK fjernet.
  - Oppdateringsfrekvens: ukentlig (ikke daglig data-pull + ukentlig modell).
  - Prognoseuttrykk: årsvis veksttakt (%) – ikke nivå.
- M0 lukkes. Starter M1-planlegging.

### 2026-04-26 – Sesjon 2 oppstart
- Leste prosjektplan.md, CLAUDE.md og STATUS.md.
- Fase: M0 – avventer svar på de fem avklaringspunktene fra prosjekteier.
- Plan for sesjonen: Stille de fem M0-spørsmålene, motta svar, oppdatere STATUS.md, og foreslå arbeidsplan for M1.

### 2026-04-26 – Prosjektoppstart
- Prosjektplan (`prosjektplan.md`) opprettet og lastet til repo.
- Operativ instruks for utviklingsagenter (`CLAUDE.md`) opprettet.
- Denne statusprotokollen (`STATUS.md`) opprettet.

---

## Milepælsstatus

| Milepæl | Status | Dato fullført |
|---|---|---|
| M0 – Oppstart og avklaring | 🟢 Fullført | 2026-04-26 |
| M1 – Datapipeline | 🟢 Fullført | 2026-04-26 |
| M2 – Modeller v1 | 🟢 Fullført | 2026-04-26 |
| M3 – Kryssjekk og ensemble | 🟢 Fullført | 2026-04-27 |
| M4 – Dashboard v1 | 🟢 Fullført | 2026-04-27 |
| M5 – QA, dokumentasjon, lansering | 🟡 Pågår | – |
| M6 – Drift og iterasjon | ⚪ Ikke startet | – |

**Statuslegende:** 🟢 Fullført · 🟡 Pågår · 🔴 Blokkert · ⚪ Ikke startet
