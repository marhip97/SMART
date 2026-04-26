# Modellkort: ML Baseline (Random Forest)

| Felt | Verdi |
|---|---|
| **ID** | ml_baseline |
| **Klasse** | `MLBaselineModel` (`src/models/ml_baseline.py`) |
| **Type** | Ensemblebasert maskinlæring |
| **Gjelder** | Alle målvariabler |
| **Bruker eksogene variabler** | Ja |

## Beskrivelse

Random Forest-regresjon brukt som ikke-lineær baseline. Features er laggede verdier av målvariabelen (lags 1, 2, 4) og laggede betingingsvariabler. En separat modell trenes per prognosehorisont (direkte multi-step-strategi). Kvantiler estimeres fra fordelingen av prediksjoner på tvers av de individuelle trærne i skogen (approksimativ kvantilregresjon).

## Antakelser

- Fremtidig utvikling kan forutsies fra laggede verdier av relevante variabler.
- Sammenhengene kan være ikke-lineære.
- Trenes på årsdata (månedlige/kvartalsvise serier resamples til årlig).

## Datakrav

- Minimum ~15 årslige observasjoner etter feature-konstruksjon.
- Betingingsvariabler resamples til årlig frekvens.

## Kvantilberegning

Hvert tre i skogen gir en prediksjon. 10., 50. og 90. persentil av denne fordelingen brukes som q10, q50 og q90. Dette er en approksimativ metode (kvantilregresjonsskog er mer presis, men krever ekstra avhengigheter).

## Kjente svakheter

- Ekstrapolerer dårlig utenfor treningsdataenes verdirom (Random Forest predikerer aldri utenfor max/min av treningsdata).
- Overfitter lett med lite data – begrenses via `max_depth`.
- Feature engineering er manuell og begrenset; mer avanserte features kan forbedre ytelsen.
- Ikke-tolkbar (black-box), men feature importance gis i metadata.

## Evalueringsresultater

*(Fylles inn etter første backtest-kjøring på historiske data.)*
