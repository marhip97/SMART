# Modellkort: VAR

| Felt | Verdi |
|---|---|
| **ID** | var |
| **Klasse** | `VARModel` (`src/models/var.py`) |
| **Type** | Multivariat tidsserie |
| **Gjelder** | Alle 7 målvariabler (joint) |
| **Bruker eksogene variabler** | Nei (alle variabler er endogene) |

## Beskrivelse

Vector Autoregression (VAR) modellerer alle målvariabler simultant, der hver variabel forklares av laggede verdier av alle variablene i systemet. Laggorden p velges via AIC. Når kun én serie er tilgjengelig, degraderes modellen til et univariate AR(p) (fallback).

## Antakelser

- Alle variabler i systemet er svakt stasjonære (eller differensiert til stasjonaritet).
- Krysseffekter mellom variabler er lineære og stabile over tid.
- Ingen strukturelle brudd i samvariasjonen mellom variablene.

## Datakrav

- Minimum 2 variabler (ellers fallback til AR).
- Minimum `max_lags + 2` observasjoner.
- Felles tidsperiode for alle variabler i panelet.

## Kvantilberegning

Prognoseintervaller beregnes via statsmodels `forecast_interval` (basert på analytisk MSE). 80 % KI tolkes som 10/90-kvantiler.

## Kjente svakheter

- Krever stasjonæritet – norske makrovariabler kan ha enhetrot.
- Antall parametre vokser kvadratisk med antall variabler (overtilpasning ved mange variabler).
- AIC-basert laggvalg kan feile ved singulær kovariansmatrise (håndteres med fallback til lag=1).
- Fanger ikke ikke-lineære sammenhenger.

## Evalueringsresultater

*(Fylles inn etter første backtest-kjøring på historiske data.)*
