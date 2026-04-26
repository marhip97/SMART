# Modellkort: AR-X

| Felt | Verdi |
|---|---|
| **ID** | arx |
| **Klasse** | `ARXModel` (`src/models/arx.py`) |
| **Type** | Autoregressiv med eksogene variabler |
| **Gjelder** | Alle målvariabler |
| **Bruker eksogene variabler** | Ja |

## Beskrivelse

AR-X er en utvidelse av AR-modellen der laggede verdier av kondisjoneringsvariabler (oljepris, EUR/NOK, handelspartnervekst, ECB-rente, K2) inkluderes som ekstra prediktorer. Laggorden p velges via AIC. Eksogene variabler inngår med `exog_lags` lags (default: 1).

## Antakelser

- Målvariabelen er lineært avhengig av egne laggede verdier og laggede betingingsvariabler.
- Betingingsvariabler er eksogene (ikke simultant bestemt med målvariabelen).
- For prognoser: betingingsvariabler antas å holde seg på siste observerte verdi (naiv antakelse). Kan erstattes med scenariobaserte baner i M3.

## Datakrav

- Betingingsvariabler må ha tilstrekkelig overlapp med målvariabelens tidsperiode.
- Minimum ~15 observasjoner etter lagg-trimming.

## Kvantilberegning

Konfidensintervaller fra statsmodels SARIMAX `get_forecast`. 80 % KI som 10/90-kvantiler.

## Kjente svakheter

- Naiv antagelse om konstante betingingsvariabler i prognosehorisonten svekker 2-3 års prognoser.
- Eksogene variabler er antatt eksogene – simultanitetsproblemer ignoreres.
- Lineær spesifikasjon fanger ikke ikke-lineære reaksjoner (f.eks. asymmetrisk rentepåvirkning).

## Evalueringsresultater

*(Fylles inn etter første backtest-kjøring på historiske data.)*
