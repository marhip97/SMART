# Modellkort: ARIMA

| Felt | Verdi |
|---|---|
| **ID** | arima |
| **Klasse** | `ARIMAModel` (`src/models/arima.py`) |
| **Type** | Univariat tidsserie |
| **Gjelder** | Alle målvariabler |
| **Bruker eksogene variabler** | Nei |

## Beskrivelse

ARIMA(p,d,q) er en klassisk univariat tidsseriemodell som modellerer en variabel utelukkende basert på sine egne historiske verdier og feiltermer. Orden (p,d,q) velges automatisk via AIC-minimering over et grid av kombinasjoner. Differensieringsorden d velges via ADF-test for stasjonaritet.

## Antakelser

- Fremtidig utvikling kan forutsies fra egne historiske verdier og residualer.
- Strukturen i serien er stasjonær etter d differensieringer.
- Feil er normalfordelte og uavhengige (hvit støy etter riktig spesifikasjon).

## Datakrav

- Minimum ~15 observasjoner for meningsfull ordensvalg.
- Ingen hull i tidsserien (NaN-verdier droppes).

## Kvantilberegning

Konfidensintervaller fra statsmodels SARIMAX `get_forecast` brukes direkte. 80 % KI (α=0.2) tolkes som approksimative 10/90-kvantiler under normalitetsantakelse.

## Kjente svakheter

- Fanger ikke krysseffekter mellom variabler (univariat).
- Dårlig på strukturelle brudd og regime-skifter.
- Normalitetsantakelsen for kvantilene er sjelden perfekt oppfylt.

## Evalueringsresultater

*(Fylles inn etter første backtest-kjøring på historiske data.)*
