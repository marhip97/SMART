# Datakilder – SMART

Dokumenterer lisens, bruksvilkår og tilgangsstatus for alle datakilder brukt i SMART.

Sist oppdatert: 2026-04-26

---

## SSB Statistikkbanken

| Felt | Verdi |
|---|---|
| URL | https://www.ssb.no/omssb/tjenester-og-verktoy/api-og-microsdata |
| Lisens | Norsk lisens for offentlige data (NLOD) 2.0 |
| API-nøkkel | Ikke påkrevet |
| Betingelser | Fri bruk med kildeangivelse. Kommersiell bruk tillatt. |
| Status | ✅ Verifisert og godkjent for produksjonsbruk |

**Variabler hentet herfra:** BNP Fastlands-Norge, KPI, KPI-JAE, lønnsvekst, boligprisvekst, registrert ledighet (via tabell 05111)

---

## NAV Åpne data

| Felt | Verdi |
|---|---|
| URL | https://data.nav.no |
| Lisens | NLOD 2.0 |
| API-nøkkel | Ikke påkrevet |
| Betingelser | Fri bruk med kildeangivelse. |
| Status | ✅ Verifisert. Registrert ledighet hentes via SSB tabell 05111 (NAV er primærkilde, SSB er distributør). |

**Variabler hentet herfra:** Registrert ledighet (andel av arbeidsstyrken)

---

## Norges Bank Data (data.norges-bank.no)

| Felt | Verdi |
|---|---|
| URL | https://data.norges-bank.no |
| Lisens | Norges Banks åpne datalisens (CC BY 4.0-kompatibel) |
| API-nøkkel | Ikke påkrevet |
| Betingelser | Fri bruk med kildeangivelse: "Kilde: Norges Bank". |
| Status | ✅ Verifisert og godkjent for produksjonsbruk |

**Variabler hentet herfra:** Styringsrente (SIREN), EUR/NOK valutakurs, handelspartnervekst (TPGDP), K2 kredittvekst husholdninger

---

## FRED – Federal Reserve Bank of St. Louis

| Felt | Verdi |
|---|---|
| URL | https://fred.stlouisfed.org |
| Lisens | Offentlig tilgjengelig; individuelle seriers lisens varierer. Se https://fred.stlouisfed.org/legal/ |
| API-nøkkel | Ikke påkrevet for CSV-nedlasting via fredgraph.csv-endepunktet |
| Betingelser | Tillatt for forskning og ikke-kommersielle formål. SMART er et eksperimentelt kryssjekkverktøy – ikke kommersielt. Kildeangivelse påkrevet: "Source: FRED, Federal Reserve Bank of St. Louis". |
| Status | ✅ Verifisert for ikke-kommersiell bruk. Dersom SMART får kommersiell anvendelse må lisens revurderes. |

**Variabler hentet herfra:** Oljepris Brent (DCOILBRENTEU), ECB innskuddsrente (ECBDFR)

---

## Risikoer knyttet til datakilder

| Risiko | Kilde | Tiltak |
|---|---|---|
| API-endringer / utilgjengelighet | Alle | Modulær DataSource-klasse muliggjør enkel bytte. Varsling via GitHub Actions ved feil. |
| Sesongjustering endres | SSB | Rådata lagres med vintage-tidsstempel; aldri overskriving. |
| Revisjoner i bakover | SSB, Norges Bank | Vintagelagring håndterer dette; modellene kan bruke vintage-kontrollert data. |
| FRED-tilgang krever API-nøkkel i fremtiden | FRED | Bruk offisielt FRED API med API-nøkkel som GitHub Secret dersom CSV-endepunkt stenges. |
