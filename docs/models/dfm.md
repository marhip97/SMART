# Modellkort: DFM

| Felt | Verdi |
|---|---|
| **ID** | dfm |
| **Klasse** | `DFMModel` (`src/models/dfm.py`) |
| **Type** | Dynamisk faktormodell |
| **Gjelder** | Alle målvariabler |
| **Bruker eksogene variabler** | Ja (som del av panelet) |

## Beskrivelse

Dynamic Factor Model (DFM) ekstraher et lite antall latente faktorer som fanger fellesdynamikken på tvers av mange observerte serier. Implementert via statsmodels `DynamicFactorMQ`, som støtter blandede frekvenser. Kondisjoneringsvariabler inngår i panelet og bidrar til faktorestimering.

## Antakelser

- Et lite antall latente faktorer driver konjunktursvingningene.
- Faktordynamikken er AR(factor_order).
- Idiosynkratiske feil følger AR(error_order) per variabel.
- Observerte serier er lineære avbildninger av faktorene.

## Datakrav

- Minimum ~20 observasjoner per serie.
- Alle serier må ha felles frekvens (eller håndteres av `DynamicFactorMQ`s blandingsfrekvens-funksjonalitet).

## Kvantilberegning

Konfidensintervaller fra statsmodels `get_forecast`. 80 % KI (α=0.2) som 10/90-kvantiler.

## Kjente svakheter

- Konvergens kan feile ved lite data eller mange faktorer relativt til observasjoner.
- Faktorene er latente og ikke direkte tolkbare uten rotasjon.
- Sensitiv til antall faktorer `n_factors` (default: 2).

## Evalueringsresultater

*(Fylles inn etter første backtest-kjøring på historiske data.)*
