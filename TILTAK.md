# TILTAK – SMART metodisk kvalitetssikring

Operativ arbeidsliste for M5 fase 2. Prioritert rekkefølge innen hver kategori.
Se `STATUS.md` logg 2026-04-28 for kontekst.

---

## P1 – Stabilitet (blokkerer lansering)

### T1 – ARIMA: tving d ≤ 1 og legg til min-obs-grense

| Felt | Verdi |
|---|---|
| **Fil(er)** | `src/models/arima.py` |
| **Problem** | `_select_d()` velger d=2 på korte serier med svak ADF. Med `trend="c"` og d=2 divergerer prognosene. Eksempel: styringsrente 2029 = 41,9 %, lønnsvekst 2028 = 11,2 % |
| **Akseptansekriterium** | Alle q50-prognoser for alle 7 variabler er innenfor [historisk_min − 2·std, historisk_max + 2·std]. ARIMA-RMSE for KPI ≤ 3,0 (vs nå 9,4). Styringsrente 2029 ≤ 10 % |
| **Test** | `tests/test_models.py`: legg til `test_arima_no_explosive_forecast` – syntetisk AR(1)-serie, bekreft at q90 < historisk_max + 3·std for alle horisonter |
| **Kompleksitet** | S |

**Konkret endring:** Endre `_select_d()` til å returnere `min(d, 1)`. Legg til guard: hvis `len(y) < 15`, bruk `order=(1,1,0)` direkte uten seleksjon.

---

### T2 – AR-X: klipp ekstreme prediksjoner

| Felt | Verdi |
|---|---|
| **Fil(er)** | `src/models/arx.py` |
| **Problem** | I backtesting for boligprisvekst (2007-vindu) returnerer SARIMAX en prediksjon på 7 672. RMSE = 1 633, R² = −180 475. Modellen får likevel 5 % ensemblevekt via floor |
| **Akseptansekriterium** | Ingen enkeltprediksjon (inkl. backtest) avviker mer enn 10·std fra historisk gjennomsnitt. AR-X RMSE for boligprisvekst ≤ 20 (vs nå 1 633) |
| **Test** | `tests/test_models.py`: `test_arx_clip_extreme` – tren AR-X på syntetisk serie, bekreft at `predict()` returnerer verdier innenfor rimelig rekkevidde selv med konstante eksogene |
| **Kompleksitet** | S |

**Konkret endring:** I `ARXModel.predict()` og `walk_forward_eval`-iterasjoner: klipp q10/q50/q90 til `[μ − 10σ, μ + 10σ]` der μ og σ beregnes fra treningsdata.

---

### T3 – Runner: hopp over modeller med for lite data

| Felt | Verdi |
|---|---|
| **Fil(er)** | `src/runner.py`, `src/models/utils.py` |
| **Problem** | Lønnsvekst har 9 årlige observasjoner. BVAR(2), DFM med 2 faktorer og AR-X kjøres likevel. Alle eval-felt er null (`n_obs: 0`), vekter flate (0,2). Fan-charten 2028 spenner 3,4–8,8 % – ren støy |
| **Akseptansekriterium** | En modell hoppes over (og logges til `model_health`) hvis: antall observasjoner etter transform < modellens `min_obs`-parameter. Lønnsvekst med 9 obs skal kun kjøre ARIMA og evt. VAR(1), ikke BVAR/DFM/AR-X |
| **Test** | `tests/test_runner.py`: `test_skip_model_insufficient_data` – kall `run_variable` med 8-obs-serie, bekreft at BVAR og DFM ikke er i resultatet |
| **Kompleksitet** | M |

**Konkret endring:** Legg til `min_obs: N` i `config/models.yaml` per modell (ARIMA: 10, VAR: 10, BVAR: 15, DFM: 15, AR-X: 15, ML: 20). I `run_variable`: sjekk `len(y) >= model_cfg.get("min_obs", 10)` før fit.

---

## P2 – Usikkerhetskalibrering

### T4 – BVAR: legg til innovasjons-usikkerhet i simulering

| Felt | Verdi |
|---|---|
| **Fil(er)** | `src/models/bvar.py` |
| **Problem** | `_simulate_forward()` simulerer deterministisk (ingen ε per steg). BVAR-fan er 3–5x for trang. BNP 2026: BVAR q10/q90 = 1,97/2,87 vs ARIMA 0,31/5,18. Korrekte kvantiler skal reflektere modellusikkerhet |
| **Akseptansekriterium** | BVAR q90−q10 for BNP 2026 ≥ 1,5 pp (vs nå 0,9). Spread øker monotont med horisont |
| **Test** | `tests/test_bvar.py`: `test_bvar_fan_widens_with_horizon` – bekreft at (q90−q10) for år 3 > (q90−q10) for år 1. `test_bvar_innovation_variance_nonzero` – bekreft at std(simulerte baner) > 0 |
| **Kompleksitet** | M |

**Konkret endring:** I `_simulate_forward()`: trekk `ε_t ~ N(0, Σ_posterior)` per tidssteg og legg til i tilstandsvektoren. Σ beregnes allerede i `fit()` – pass den til simuleringsmetoden.

---

### T5 – ML-baseline: korrekt kvantilestimering

| Felt | Verdi |
|---|---|
| **Fil(er)** | `src/models/ml_baseline.py` |
| **Problem** | `np.percentile([tree.predict(x) for tree in rf.estimators_], [10,50,90])` gir spredning i gjennomsnittsprediksjoner, ikke prediktive kvantiler. Underestimerer usikkerhet systematisk |
| **Akseptansekriterium** | ML-baseline q90−q10 for BNP 2026 ≥ 2,0 pp. Kvantilene er kalibrerte: i backtesting skal ~80 % av faktiske obs falle innenfor q10–q90 |
| **Test** | `tests/test_models.py`: `test_ml_baseline_quantile_coverage` – syntetisk serie med kjent varians, bekreft empirisk dekningsgrad ~80 % ± 15 pp |
| **Kompleksitet** | M |

**Konkret endring:** Erstatt tree-ensemble-spredning med empiriske kvantiler over bladnode-verdier (`estimator.apply(X)` → samle alle treningsobservasjoner i terminal-blad → ta percentiler). Ingen ny avhengighet nødvendig. *(Se eskaleringsnotat under.)*

---

### T6 – AR-X: sanity-check også på predict() (ikke bare backtest)

| Felt | Verdi |
|---|---|
| **Fil(er)** | `src/models/arx.py` |
| **Problem** | T2 klipper backtest-prediksjoner. `predict()` for fremtiden er ikke klipt og kan gi like ekstreme verdier hvis eksogene antar uvanlige nivåer |
| **Akseptansekriterium** | `predict()` returnerer aldri q50 utenfor [μ_hist − 5σ, μ_hist + 5σ] der μ og σ er fra treningsdata |
| **Test** | Inkluderes i `test_arx_clip_extreme` fra T2 |
| **Kompleksitet** | S |

---

## P3 – Pedagogisk klargjøring

### T7 – Disagreement: bytt til relativ terskel

| Felt | Verdi |
|---|---|
| **Fil(er)** | `src/ensemble/disagreement.py`, `config/models.yaml` (evt. ny param) |
| **Problem** | `spread > 1.0` flagger absolutt alt. Alle 7 variabler har high_disagreement på alle 3 horisonter → flagget gir null informasjonsverdi |
| **Akseptansekriterium** | Flagget treffer ≤ 50 % av (variabel, horisont)-kombinasjoner i normal kjøring. Terskelen bør være relativ: `spread / abs(ensemble_q50) > terskel` (forslag: 0,5), eller `spread > 2·historisk_std` |
| **Test** | `tests/test_ensemble.py`: `test_disagreement_relative_threshold` – bekreft at serier med lite sprik ikke flagges; bekreft at serier med stort sprik (syntetisk) flagges |
| **Kompleksitet** | S |

**Konkret endring:** Endre `high_disagreement`-logikken til: `spread > max(1.0, 0.5 * abs(ensemble_q50))` – dvs. absolutt gulv på 1,0 pp OG relativ grense på 50 % av prognosen.

---

### T8 – walk_forward_eval: tryggere modell-kloning

| Felt | Verdi |
|---|---|
| **Fil(er)** | `src/models/utils.py` |
| **Problem** | `copy.deepcopy(model)` er risikabelt for statsmodels-objekter med cached tilstand (fitted-result-objekter). Kan gi subtile feil ved parallell kjøring eller mellom Python-versjoner |
| **Akseptansekriterium** | Kloningen bruker `model.__class__(variable_id=model.variable_id, **model._init_params)`. Alle modeller lagrer init-parametre i `self._init_params` dict |
| **Test** | `tests/test_models.py`: `test_walk_forward_clone_independent` – bekreft at to walk-forward-kjøringer med samme input gir identisk resultat (determinisme) |
| **Kompleksitet** | M |

---

### T9 – Modellkort BVAR: oppdater etter T4

| Felt | Verdi |
|---|---|
| **Fil(er)** | `docs/models/bvar.md` |
| **Problem** | Kortet beskriver "shrinkage mot random walk" men koden bruker prior-mean = 0 (white noise). Etter T4 endres også kvantilbeskrivelsen |
| **Akseptansekriterium** | Kortet angir korrekt at prior-mean = 0 (ikke random walk), og beskriver at kvantiler stammer fra Monte Carlo-simulering med innovasjons-usikkerhet |
| **Test** | Ingen kodetest – manuell gjennomgang |
| **Kompleksitet** | S |

---

## P4 – Lanseringsklargjøring

### T10 – CHANGELOG.md

| Felt | Verdi |
|---|---|
| **Fil(er)** | `CHANGELOG.md` (ny fil) |
| **Problem** | Ingen logg over metodiske endringer mellom kjøringer. Nødvendig for reproduserbarhet og brukertillit |
| **Akseptansekriterium** | Filen følger Keep a Changelog-format. Alle P1/P2-fikser dokumenteres under `[Unreleased]` → `[1.0.0]` ved lansering |
| **Test** | Ingen kodetest |
| **Kompleksitet** | S |

---

### T11 – model_health-felt i forecast-JSON

| Felt | Verdi |
|---|---|
| **Fil(er)** | `src/runner.py` |
| **Problem** | Ingen maskinlesbar logg over hvilke modeller som ble ekskludert per variabel og hvorfor. Dashboardet kan ikke advare brukeren om redusert modelldekning |
| **Akseptansekriterium** | `latest.json` inneholder `model_health: {included: [...], excluded: [{model: "...", reason: "..."}]}`. Dashboard viser antall inkluderte modeller i nøkkeltall-raden |
| **Test** | `tests/test_runner.py`: `test_model_health_logged` – bekreft at ekskluderte modeller (p.g.a. T3) dukker opp i `model_health.excluded` |
| **Kompleksitet** | S |

---

### T12 – Ærlig backtesting-disclaimer på dashboard

| Felt | Verdi |
|---|---|
| **Fil(er)** | `dashboard/index.html`, `dashboard/app.js` |
| **Problem** | Dashboard viser R² uten kontekst. Mange variabler har <15 backtest-obs og negativ R² på alle modeller. En naiv bruker kan tro RMSE = 2,6 % for BNP er god – men det er dårligere enn historisk gjennomsnitt |
| **Akseptansekriterium** | Treffsikkerhetstabellen inneholder en forklarende note: "Negativ R² betyr at modellen i backtesting var dårligere enn å spå historisk gjennomsnitt. Dette er vanlig for makroøkonomiske prognoser." Antall backtest-observasjoner vises tydelig |
| **Test** | Ingen kodetest |
| **Kompleksitet** | S |

---

## Eskaleringsnotat – T5 (ML-baseline)

**Spørsmål til prosjekteier:** T5 kan løses på to måter:
- **(A) Bladnode-kvantiler** (ingen ny avhengighet): samle treningsobservasjoner i terminal-blader, ta empiriske percentiler. Mer arbeid, ~50 linjer.
- **(B) `sklearn-quantile-forest`** (ny avhengighet): drop-in erstatning for `RandomForestRegressor` med innebygd kvantilregresjon. Korrekt prinsipielt, ~5 linjer.

**Anbefaling:** Alternativ A. `sklearn-quantile-forest` er ikke aktivt vedlikeholdt (siste release 2022) og legger til en avhengighet for noe vi kan implementere selv.

**Blokkerer:** Ikke noe – kan implementeres parallelt med T4.

---

## Implementeringssekvens

| Branch | Tiltakene | Avhengighet |
|---|---|---|
| `claude/p1-stabilize-forecasts` | T1, T2, T3 | Ingen |
| `claude/p2-honest-uncertainty` | T4, T5, T6 | T2 (clip) bør være ferdig |
| `claude/p3-pedagogical-clarity` | T7, T8, T9 | T4 (BVAR-kort) |
| `claude/p4-launch-readiness` | T10, T11, T12 | T3 (model_health) |

Merge-rekkefølge: P1 → P2 → P3 → P4. Lansering av v1.0 etter at P1 og P2 er merget og dashboardet er manuelt verifisert.
