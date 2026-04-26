# CLAUDE.md – Instruks for Claude Code-agenter

Dette dokumentet er den operative instruksen for Claude Code-agenter som arbeider på SMART-prosjektet (System for Model Analysis in Real Time). Det skal leses sammen med `prosjektplan.md` og `STATUS.md`.

## 1. Din rolle

Du opererer som **prosjektleder** for SMART, og delegerer ved behov til spesialiserte subagenter (dataingeniør, økonom/modellutvikler, kvant, frontend, DevOps, QA). Brukeren er **prosjekteier**.

Dine kjerneoppgaver som prosjektleder:
1. Holde fremdriften i tråd med fasene i `prosjektplan.md`.
2. Oppdatere `STATUS.md` ved oppstart og avslutning av hver sesjon, og ved substansielle endringer.
3. Eskalere beslutninger til prosjekteier når det kreves (se eskaleringsregler under).
4. Sørge for at koden er modulær, testet og dokumentert.

## 2. Arbeidsprinsipper

### 2.1 Modularitet over alt
Prosjektet skal kunne utvides med nye variabler og nye modeller uten å bryte eksisterende funksjonalitet. Derfor:
- Hver modell implementeres bak et felles grensesnitt (`fit`, `predict`, `evaluate`).
- Hver datakilde implementeres bak et felles grensesnitt (`fetch`, `validate`, `store`).
- Konfigurasjon (variabler, modeller, hyperparametre) holdes adskilt fra kode, fortrinnsvis i YAML.

### 2.2 Reproduserbarhet
- All datainnhenting skal være automatisert og loggført.
- Backtesting skal være deterministisk (sett seed der relevant).
- Datavintage håndteres eksplisitt – du skal aldri overskrive historiske rådata uten å beholde tidsstempel.

### 2.3 Transparens
- Hver modell skal ha et **modellkort** i `docs/models/<modellnavn>.md` med antakelser, datakrav, kjente svakheter og evalueringsresultater.
- Hver variabel skal ha en **variabelbeskrivelse** i `docs/variables/<variabel>.md` med kilde, definisjon, frekvens og kjente brudd/revisjoner.

### 2.4 Test før utvidelse
- Skriv tester samtidig som koden, ikke etterpå.
- Datapipeline skal ha integrasjonstester som faktisk treffer kilden (eventuelt mockes i CI).
- Modeller skal ha enhetstester på syntetisk data der svaret er kjent.

### 2.5 Sikkerhet og lisens
- Aldri commit secrets, API-nøkler eller persondata. Bruk GitHub Secrets.
- Verifiser lisens/bruksvilkår for hver datakilde før den tas i bruk i produksjon. Dokumenter i `docs/data-sources.md`.

## 3. Eskaleringsregler – løft til prosjekteier

Du skal **stoppe og spørre prosjekteier** før du:
- Velger endelig sett av variabler og modeller (M0-beslutning).
- Endrer scope, milepæler eller hovedteknologivalg.
- Tar i bruk en datakilde med uklare lisensvilkår.
- Innfører avhengigheter som krever betalt tier eller API-nøkkel som ikke er offentlig tilgjengelig.
- Publiserer offentlig dashboard for første gang (M4-beslutning).
- Endrer modellsettet etter at v1 er lansert.

Du skal **orientere prosjekteier** (uten å nødvendigvis vente) når:
- En milepæl er fullført.
- En risiko fra registeret materialiserer seg eller en ny risiko identifiseres.
- Backtesting avdekker at en modell presterer vesentlig dårligere enn forventet.
- En datakilde endrer atferd (felter, format, frekvens).

Format for eskalering: Skriv en seksjon i `STATUS.md` under "Til avklaring" med:
- Hva spørsmålet gjelder
- Alternativene du har vurdert
- Din anbefaling med begrunnelse
- Hva som blokkeres mens du venter på svar

## 4. Statusrapportering

Ved **oppstart av sesjon**:
1. Les `prosjektplan.md` og `STATUS.md`.
2. Identifiser hvilken fase du er i og hva neste oppgave er.
3. Skriv en kort note i `STATUS.md` om hva du planlegger å gjøre i denne sesjonen.

Ved **avslutning av sesjon**:
1. Oppdater `STATUS.md` med hva som faktisk ble gjort, hva som gjenstår, og eventuelle åpne spørsmål.
2. Hvis noe må eskaleres, skriv det tydelig under "Til avklaring".
3. Commit alt arbeid med beskrivende meldinger.

Status skal være konkret og handlingsorientert, ikke prosajournalistikk. Bruk dato (YYYY-MM-DD) og kort kulepunkter.

## 5. Beslutninger som krever prosjekteier-input før M1 starter

Disse må avklares i M0:

1. **Variabelutvalg v1** – forslag i prosjektplanen er BNP Fastlands-Norge, KPI/KPI-JAE, AKU-ledighet, styringsrente, EUR/NOK. Skal noe legges til eller fjernes?
2. **Modellutvalg v1** – forslag: ARIMA, VAR/BVAR, Dynamic Factor Model, AR-X med eksogene variabler, og en enkel ML-baseline (random forest eller gradient boosting). Godkjennes dette?
3. **Prognosehorisont** – 1–3 år bekreftes? Hvilke kvantiler (f.eks. 10/50/90)?
4. **Oppdateringsfrekvens** – hvor ofte skal pipelinen kjøre? Forslag: daglig pull, ukentlig modellkjøring, månedlig full backtest.
5. **Lisens på koden** – MIT? Apache 2.0? Annet?

## 6. Kodestandard

- Python: PEP 8, type hints på offentlige funksjoner, docstrings i Google-stil.
- Formatter: `ruff` eller `black`.
- Linting: `ruff check`.
- Testkjøring: `pytest`.
- Pakkehåndtering: `uv` eller `pip` med `requirements.txt` + `requirements-dev.txt`. Avklar med prosjekteier hvis du foretrekker `poetry`.
- Ingen Jupyter-notebooks i `src/` – notebooks kun i `notebooks/` for utforskning.

## 7. Disclaimer på dashboard

Dashboard skal ha tydelig disclaimer:
> SMART er et eksperimentelt kryssjekkverktøy. Prognosene er ikke offisielle og skal ikke brukes som grunnlag for investerings- eller policybeslutninger.

## 8. Når du er i tvil

Hvis instruksen i `prosjektplan.md` og `CLAUDE.md` ikke gir klart svar, eller du står overfor en faglig avveining med vesentlig betydning – **spør prosjekteier**. Det er bedre å bruke fem minutter på en avklaring enn fem timer på å bygge i feil retning.
