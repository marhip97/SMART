# STATUS – SMART

Levende status- og fremdriftsprotokoll. Oppdateres av prosjektleder (Claude Code) løpende.

**Format:** Nyeste oppføring øverst. Bruk dato (YYYY-MM-DD) og kortfattede punkter.

---

## Gjeldende fase
**M5 – QA, dokumentasjon, lansering** (M4 fullført 2026-04-27)
**v1-lansering utsatt** – metodegjennomgang avdekket P1-problemer som gir åpenbart misvisende prognoser. Lansering skjer etter at P1 og P2 er lukket.

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

- **R3 (overfitting/dårlig out-of-sample)** – materialisert. ARIMA eksploderer på korte serier (d=2), AR-X gir RMSE=1633 på boligprisvekst, BVAR-fan er 3–5x for trang. Tiltak igangsatt: se TILTAK.md T1–T4.
- **R6 (teknisk gjeld)** – delvis materialisert. `copy.deepcopy` i walk_forward_eval er risikabelt for statsmodels-objekter. Tiltak: T8.

---

## Logg

### 2026-04-28 – Branch P1 ferdig: T1, T2, T3 implementert

Alle akseptansekriterier i TILTAK.md T1–T3 møtt. Konkret resultat:

| Kriterium | Før | Etter |
|---|---|---|
| Styringsrente 2029 q50 (ARIMA) | 41,9 % | 4,02 % |
| Lønnsvekst 2028 fan q10–q90 | 3,4–8,8 (støy) | 2,1–8,2 (ARIMA) |
| KPI ARIMA RMSE | 9,42 | 1,24 |
| Boligprisvekst AR-X RMSE | 1 633 | 11,6 |
| Lønnsvekst aktive modeller | 6 (alle n_obs=0) | 2 (ARIMA + VAR) |

**T1** (`src/models/arima.py`): max_d hardkodet til 1, kort serie (n<15) bruker ARIMA(1,1,0) med trend="n", forhindrer drift fra konstant ledd ved differensiering. Alle q50 klippet til ±10·σ via ny `clip_forecast()` i `utils.py`.

**T2/T6** (`src/models/arx.py`): AR-X bruker samme `clip_forecast()` på predict() og dermed også via walk_forward_eval. RMSE for boligprisvekst: 1 633 → 11,6.

**T3** (`config/models.yaml`, `src/runner.py`): Hver modell har `min_obs`. Runner sjekker og hopper over modeller med utilstrekkelig data, logger til nytt `model_health`-felt i forecast-JSON. ARIMA min_obs=8 (kjører ARIMA(1,1,0) trygt), BVAR/DFM/AR-X=15, ML=20.

128/128 tester grønt (122 + 6 nye for T1, T2, T3).

**Neste:** Branch P2 (T4, T5, T6) – BVAR innovasjons-usikkerhet, ML-baseline kvantilfix.

### 2026-04-28 – M5 metodegjennomgang: 12 funn, lansering utsatt

Systematisk gjennomgang av modellkode og prognoseresultater avdekket fire kategorier funn. Detaljer og akseptansekriterier i `TILTAK.md`.

**P1 – Stabilitetsproblemer (blokkerer lansering):**
- T1: ARIMA velger d=2 på korte serier → divergerende prognoser (styringsrente 2029 = 41,9 %, lønnsvekst 2028 = 11,2 %)
- T2: AR-X eksploderer i backtest-vindu 2007 for boligprisvekst (prediksjon = 7 672, RMSE = 1 633)
- T3: Runner mangler min-obs-sjekk; lønnsvekst (9 obs) kjører BVAR/DFM/AR-X som gir flate vekter og støy-prognose

**P2 – Feilkalibrerte usikkerhetsintervaller:**
- T4: BVAR simulerer deterministisk (ingen ε), fan 3–5x for trang
- T5: ML-baseline bruker tresprednig som kvantiler, ikke prediktive kvantiler
- T6: AR-X mangler clipping i predict() for fremtidsprognoser

**P3 – Pedagogisk klargjøring:**
- T7: Disagreement-flagg treffer alle kombinasjoner (absolutt terskel 1,0 pp er for lav)
- T8: `copy.deepcopy` i walk_forward_eval → bytt til parameterstyrt kloning
- T9: BVAR-modellkort beskriver feil prior (random walk vs white noise)

**P4 – Lanseringsklargjøring:**
- T10: CHANGELOG.md mangler
- T11: `model_health`-felt i forecast-JSON mangler
- T12: Backtesting-disclaimer på dashboard er utilstrekkelig

**Begrunnelse for utsatt lansering:** P1-funnene (T1–T3) gir synlig misvisende prognoser som kan feiloppfattes av brukere. v1.0 lanseres etter at branch P1 og P2 er merget og dashboardet er manuelt verifisert.

**Til avklaring (T5):** Se eskaleringsnotat i `TILTAK.md` – valget mellom bladnode-kvantiler og `sklearn-quantile-forest`. Anbefaling: bladnode-kvantiler (ingen ny avhengighet).

### 2026-04-28 – M5 pågår: dashboard v2 – historikk, treffsikkerhet, boligprisvekst-fix
- **boligprisvekst-fix**: rådata var indeksnivå (94–152); `_apply_transform()` i runner konverterer nå til YoY % før modellering. Prognoseversier oppdatert til ~2–10 % (korrekt størrelsesorden).
- **Historiske observasjoner**: siste 20 år med faktiske data inkludert i forecast-JSON (`history`-felt) og vist som grå referanselinje i begge grafvisninger.
- **Treffsikkerhetstabell**: RMSE, MAE, R² (out-of-sample) og antall evalueringsperioder per modell. Beste RMSE uthevet grønt, negativ R² rød. Walk-forward backtest-historikk (faktisk vs spådd per år) lagret i JSON.
- **Modell-backtest-linjer**: stiplede linjer i enkeltmodell-grafen viser historiske walk-forward-prognoser.
- **Toggle** øverst i grafen for å bytte mellom ensemble-fan og enkeltmodeller.
- **Nøkkeltall-rad**: siste faktiske obs + tre prognoseår med usikkerhetsintervall vises under variabeltittelen.
- **Variabelbeskrivelse** per variabel.
- **Språk og design**: gjennomgående mer pedagogisk norsk, bedre farger, spinner-animasjon, row-hover.
- Alle 7 forecast-JSON-filer oppdatert med ny struktur (`history`, `evaluation`, `backtest`).
- 122/122 tester grønt.

### 2026-04-28 – PR #10 merget til main: full CI/CD-kjede aktiv
- PR #10 merget. Main inneholder nå hele prosjektet (94 filer, 8 936 linjer).
- Datavintagé (12/12 variabler) og prognoseresultater (7/7 målvariabler) er committet til repo.
- Push til main trigget `deploy_dashboard.yml` → dashboard deployes til GitHub Pages.
- Ukentlig pipeline (mandag 06:00 UTC) er aktiv og vil oppdatere data + prognoser automatisk.
- **Neste M5-oppgaver**: verifiser dashboardets innhold i nettleser, gjennomgå backtesting-resultater, oppdater `docs/variables/kpi_jae.md` med riktig tabell-ID (05327).

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
| M5 – QA, dokumentasjon, lansering | 🟡 Pågår | Startet 2026-04-28 |
| M6 – Drift og iterasjon | ⚪ Ikke startet | – |

**Statuslegende:** 🟢 Fullført · 🟡 Pågår · 🔴 Blokkert · ⚪ Ikke startet
