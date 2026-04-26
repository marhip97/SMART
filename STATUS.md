# STATUS – SMART

Levende status- og fremdriftsprotokoll. Oppdateres av prosjektleder (Claude Code) løpende.

**Format:** Nyeste oppføring øverst. Bruk dato (YYYY-MM-DD) og kortfattede punkter.

---

## Gjeldende fase
**M2 – Modeller v1** (M1 fullført 2026-04-26)

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

*(Ingen åpne punkter.)*

## Risikoer som har materialisert seg
*(Ingen registrert ennå.)*

---

## Logg

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
| M2 – Modeller v1 | ⚪ Ikke startet | – |
| M3 – Kryssjekk og ensemble | ⚪ Ikke startet | – |
| M4 – Dashboard v1 | ⚪ Ikke startet | – |
| M5 – QA, dokumentasjon, lansering | ⚪ Ikke startet | – |
| M6 – Drift og iterasjon | ⚪ Ikke startet | – |

**Statuslegende:** 🟢 Fullført · 🟡 Pågår · 🔴 Blokkert · ⚪ Ikke startet
