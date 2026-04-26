# Prosjektplan – SMART
**System for Model Analysis in Real Time**

*Versjon 1.0 – April 2026*

---

## 1. Bakgrunn og problembeskrivelse

Norsk økonomi påvirkes av et bredt sett av drivkrefter – pengepolitikk, oljepris, valutakurs, internasjonal konjunktur, arbeidsmarked og finanspolitikk. Eksisterende prognosearbeid bygger gjerne på enkeltmodeller med kjente begrensninger: strukturelle modeller fanger sammenhenger godt men reagerer tregt på sjokk; tidsserie-/maskinlæringsmodeller fanger korttidsdynamikk men har svak økonomisk fortolkning; ekspertvurderinger er fleksible men lite reproduserbare.

Behovet er et **kryssjekkrammeverk** der flere modeller med ulike forutsetninger kjører parallelt på samme variabel, slik at avvik mellom modellene blir et signal i seg selv – og slik at sluttbruker enkelt kan se hvor robust en prognose er.

**Problem:** Det finnes i dag ikke et åpent, automatisert, pedagogisk verktøy for løpende kryssjekk av prognoser for sentrale norske makrovariabler på 1–3 års sikt.

## 2. Mål

### 2.1 Hovedmål
Utvikle SMART – et automatisert kryssjekkrammeverk og dashboard som genererer og visualiserer kort- og mellomlangsiktige (1–3 år) prognoser for sentrale norske makroøkonomiske variabler, basert på et sett av komplementære modeller.

### 2.2 Delmål
1. **Datalag:** Etablere en automatisert pipeline som henter, validerer og lagrer makrodata fra åpne kilder (SSB, Norges Bank, Eurostat, FRED m.fl.).
2. **Modell-lag:** Implementere et bibliotek av modeller (f.eks. ARIMA, VAR/BVAR, faktor-/DFM, brokenstick/AR-X, enkle ML-baselines) og evaluere forklaringskraft og out-of-sample-treffsikkerhet.
3. **Kryssjekk-lag:** Sammenstille modellene til ensembles med transparent vekting og avviksmål mellom modeller.
4. **Visualiseringslag:** Pedagogisk dashboard på GitHub Pages, oppdatert automatisk via GitHub Actions ved ny data.
5. **Reproduserbarhet:** Fullt versjonskontrollert kode, dokumenterte modellantakelser, og evalueringslogg.

### 2.3 Suksesskriterier
- Dashboard publisert på GitHub Pages og oppdateres automatisk minst månedlig.
- Minst 5 makrovariabler dekket i versjon 1 (forslag: BNP Fastlands-Norge, KPI, KPI-JAE, arbeidsledighet, styringsrente, Lønnsvekst, boligpriser).
- Minst 3 modeller per variabel, med dokumentert backtesting.
- Modulær arkitektur: ny variabel eller ny modell skal kunne legges til uten å bryte eksisterende kode.

### 2.4 Ikke-mål (avgrensning)
- SMART skal **ikke** levere investeringsråd eller normative politikkanbefalinger.
- SMART skal **ikke** konkurrere med Norges Banks eller SSBs offisielle prognoser, men være et komplementært kryssjekkverktøy.
- Strukturelle DSGE-modeller er utenfor scope i versjon 1 (kan vurderes senere).

## 3. Prosjektorganisering

Prosjektet utvikles av Claude Code som representerer prosjektgruppen gjennom spesialiserte agenter. Brukeren er prosjekteier.

### 3.1 Roller

| Rolle | Aktør | Ansvar |
|---|---|---|
| **Prosjekteier** | Brukeren | Strategisk retning, godkjenne milepæler, beslutte ved tvil/risiko, prioritere scope |
| **Prosjektleder** | Claude Code (hovedagent) | Daglig fremdrift, oppdatere `STATUS.md`, koordinere subagenter, eskalere til prosjekteier |
| **Dataingeniør** | Subagent | Datainnhenting, API-integrasjon, datavalidering, lagringsstruktur |
| **Økonom/modellutvikler** | Subagent | Modellvalg, økonomisk fortolkning, variabelseleksjon, antakelser |
| **Kvant/statistiker** | Subagent | Implementering av modeller, backtesting, evalueringsmetrikker, ensembling |
| **Frontend/dashboard-utvikler** | Subagent | Dashboard på GitHub Pages, visualiseringer, UX |
| **DevOps** | Subagent | GitHub Actions, deployment, automasjon, miljøhåndtering |
| **Kvalitetssikrer* | Subagent | risikostyring, kvalitetssikrer |
| **QA/dokumentasjon** | Subagent | Tester, README, modellkort, brukerveiledning |

### 3.2 Eskaleringsregler
Prosjektleder skal **alltid** løfte følgende til prosjekteier:
- Valg av modellsett og variabler før implementering starter.
- Endring i scope eller tidsplan.
- Brudd eller endringer i datakilders tilgjengelighet/lisensvilkår.
- Antatte verdier eller forutsetninger som har vesentlig innvirkning på resultater.
- Konflikt mellom modeller om "riktig" resultat når det er behov for tolkningsbeslutning.
- Risikoer som materialiserer seg eller nye risikoer som identifiseres.

### 3.3 Statusoppdateringer
Prosjektleder oppdaterer `STATUS.md` ved:
- Hver fullført milepæl.
- Hver arbeidsøkt der substansielle endringer er gjort.
- Når en sak eskaleres til prosjekteier.
- Ved oppstart og avslutning av hver sesjon.

## 4. Tekniske rammer

### 4.1 Teknologivalg (foreslått, til godkjenning)
- **Språk:** Python 3.11+ (modeller, pipeline). JavaScript/HTML/CSS for dashboard.
- **Modellbibliotek:** `statsmodels`, `scikit-learn`, `arch`, evt. `pymc` eller `darts` for utvidelser.
- **Data:** `pandas`, `pyjstat` (SSB), `requests`. Lagring som Parquet/CSV i `/data`.
- **Dashboard:** Statisk side (HTML + Plotly/Chart.js) som bygges av GitHub Actions og publiseres til GitHub Pages.
- **Automasjon:** GitHub Actions (cron + push-trigger).
- **Dokumentasjon:** Markdown i `/docs`. Modellkort per modell.

### 4.2 Repostruktur (foreslått)
```
/
├── README.md
├── CLAUDE.md                  # Instruks til Claude Code-agenter
├── prosjektplan.md            # Dette dokumentet
├── STATUS.md                  # Levende status- og fremdriftsprotokoll
├── data/
│   ├── raw/                   # Rådata fra kilder
│   └── processed/             # Renset, modell-klar data
├── src/
│   ├── data/                  # Datapipeline
│   ├── models/                # Modellimplementasjoner
│   ├── evaluation/            # Backtesting, metrikker
│   └── ensemble/              # Kryssjekk og aggregering
├── notebooks/                 # Utforskning, EDA
├── dashboard/                 # GitHub Pages-frontend
├── tests/
├── docs/
│   ├── models/                # Modellkort
│   └── variables/             # Variabeldefinisjoner
└── .github/workflows/         # CI/CD og automasjon
```

### 4.3 Datakilder (foreløpig liste – verifiseres i fase 1)
- **SSB** (Statistikkbanken API): KPI, BNP, arbeidsmarked, lønnsstatistikk.
- **Norges Bank**: Styringsrente, valutakurser, pengemengde.
- **Eurostat**: Sammenliknbar EU-data.
- **FRED** (St. Louis Fed): Internasjonale variabler (oljepris, US-rente, dollarindeks).
- **NAV**: Registrert ledighet (supplerende til SSB AKU).

Alle kilder må verifiseres mot lisens/bruksvilkår før produksjonssetting.

## 5. Faser og milepæler

| # | Fase | Leveranse | Estimert varighet |
|---|---|---|---|
| **M0** | Oppstart og avklaring | Prosjektplan godkjent. Variabel- og modellutvalg avklart med prosjekteier. | Uke 1 |
| **M1** | Datapipeline | Automatisert henting av minst 5 variabler, validering, lagring. | Uke 2–3 |
| **M2** | Modeller v1 | Minst 3 modeller per variabel implementert og backtestet. | Uke 4–6 |
| **M3** | Kryssjekk og ensemble | Avviksmål, vekting, aggregert prognose. | Uke 7 |
| **M4** | Dashboard v1 | Publisert på GitHub Pages med automatisk oppdatering. | Uke 8–9 |
| **M5** | QA, dokumentasjon, lansering | Tester, modellkort, brukerveiledning. Versjon 1.0. | Uke 10 |
| **M6** | Drift og iterasjon | Løpende vedlikehold, utvidelser etter prioritet fra prosjekteier. | Kontinuerlig |

Tidsestimatet er veiledende. Prosjektleder skal varsle prosjekteier ved avvik fra planen.

## 6. Risikovurdering

| ID | Risiko | Sannsynlighet | Konsekvens | Tiltak |
|---|---|---|---|---|
| R1 | Datakilde endrer API eller blir utilgjengelig | Middels | Høy | Modulær datakildehåndtering, fallback-kilder, varsling i CI |
| R2 | Modeller gir motstridende resultater uten klar tolkning | Høy | Middels | Eksplisitte avviksmål, dokumenterte modellantakelser, eskaler til prosjekteier ved store avvik |
| R3 | Overfitting / dårlig out-of-sample-treff | Middels | Høy | Strikt walk-forward backtesting, holdout-perioder, dokumenterte metrikker |
| R4 | Datakvalitet (revisjoner, sesongjustering, brudd i serier) | Høy | Middels | Versjonering av rådata (vintage), eksplisitt håndtering av revisjoner |
| R5 | Scope creep | Middels | Middels | Tydelige milepæler, kun prosjekteier kan utvide scope |
| R6 | Teknisk gjeld i raskt utviklet kode | Middels | Middels | Krav til tester, modulær arkitektur, kodefagfellevurdering mellom agenter |
| R7 | Misbruk: dashboard tolkes som offisielle prognoser | Lav | Høy | Tydelig disclaimer på dashboard, modellkort, ikke-mål eksplisitt dokumentert |
| R8 | GitHub Actions-kostnader/grenser overskrides | Lav | Lav | Effektiv caching, kjør tunge backtester sjeldnere enn daglig pipeline |
| R9 | Lisensbrudd på datakilder | Lav | Høy | Verifisere bruksvilkår per kilde i fase 1, dokumentere i `docs/data-sources.md` |

Risikoregisteret skal oppdateres minst ved hver milepæl, og når nye risikoer identifiseres.

## 7. Kvalitet og leveransekriterier

For at en milepæl skal regnes som fullført, må følgende være oppfylt:
- Kode er testet (enhetstester der relevant, integrasjonstester for pipeline).
- Dokumentasjon er oppdatert (README, modellkort, eventuelt brukerveiledning).
- `STATUS.md` er oppdatert.
- Prosjekteier er orientert og har bekreftet før neste fase startes på milepæler markert som beslutningspunkt (M0, M2, M4).

## 8. Endringshåndtering

Endringer i scope, tidsplan eller hovedteknologivalg krever godkjenning fra prosjekteier. Prosjektleder dokumenterer endringen i `STATUS.md` med begrunnelse, alternativer vurdert og besluttet retning.

## 9. Avhengigheter og forutsetninger

- Brukeren oppretter GitHub-repo og gir Claude Code tilgang.
- Eventuelle API-nøkler (typisk ikke nødvendig for offentlige kilder) håndteres som GitHub Secrets.
- Brukeren er tilgjengelig for avklaringer på beslutningspunkter.

---

*Dette dokumentet skal leses sammen med `CLAUDE.md` (instruks til utviklingsagenter) og `STATUS.md` (levende fremdriftsprotokoll).*
