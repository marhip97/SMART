# Modellkort: BVAR

| Felt | Verdi |
|---|---|
| **ID** | bvar |
| **Klasse** | `BVARModel` (`src/models/bvar.py`) |
| **Type** | Bayesiansk vektorautoregresjon |
| **Gjelder** | Alle 7 målvariabler (joint) |
| **Bruker eksogene variabler** | Nei (alle variabler er endogene) |

## Beskrivelse

BVAR(p) er en bayesiansk utvidelse av standard VAR der Minnesota-prioren regulariserer koeffisientmatrisene. Prioren shrinks koeffisientene mot null med styrke som avtar med lagrekkefølge og er sterkere for kryss-variable enn for egne lags. Posterior-fordelingen over koeffisientene beregnes analytisk (betinget på en OLS-estimert feilkovariansmatrise Σ). Kvantiler genereres via Monte Carlo-trekk fra posterior-normalfordelingen.

## Hyperparametre

| Parameter | Beskrivelse | Default |
|---|---|---|
| `n_lags` | VAR-lagrekkefølge | 2 |
| `lambda1` | Overordnet stramhet. Lavere = sterkere shrinkage. | 0.2 |
| `lambda2` | Kryss-variabel stramhet relativt til egne lags. | 0.5 |
| `n_draws` | Monte Carlo-trekk for kvantilestimering | 500 |

## Minnesota-prior spesifikasjon

Prior-varians for koeffisient på variabel *j* ved lag *l* i ligning *i*:

- Eget lag (*i = j*): `(λ₁/l)²`
- Krysslag (*i ≠ j*): `(λ₁λ₂/l)² × (σᵢ²/σⱼ²)`
- Konstant: diffus (stor varians)

Egne lags er løsere regulert (større varians) – modellen kan lære mye fra egne historiske verdier. Krysslags er strammere regulert – lite informasjon antas å ligge i andres historikk med standard-verdier.

## Antakelser

- Prior-strukturen (Minnesota) er rimelig for makroøkonomiske tidsserier.
- Σ er konstant over tid (homoskedasitet).
- Posterior er normalfordelt betinget på OLS Σ (ikke full Gibbs over Σ).
- Variabler er stasjonære eller nær-stasjonære.

## Datakrav

- Minimum 2 målvariabler anbefales (ellers degenererer BVAR til bayesiansk AR).
- Minimum `n_lags + 2` observasjoner.
- Felles tidsperiode for alle variabler i panelet.

## Kvantilberegning

`n_draws` trekk av koeffisientmatrisen B fra posterior N(b_post, V_post). For hvert trekk simuleres deterministisk h-steg-fremover. 10., 50. og 90. persentil av de resulterende prognosene gir fan-chart-kvantilene. Deterministisk simulering (uten å legge til feilledd) undervurderer usikkerheten noe; dette er en bevisst forenkling for å holde `n_draws` lavt og kjøretiden akseptabel.

## Fordeler fremfor standard VAR

- **Regularisering:** Minnesota-prioren forhindrer overtilpasning, spesielt ved 7 variabler og relativt korte norske makrotidsserier.
- **Ekte bayesianske kvantiler:** Kvantilene reflekterer parameter-usikkerhet, ikke bare forecast-error-varians under normalitet.
- **Bedre out-of-sample-ytelse** dokumentert i Litterman (1986) og Doan, Litterman & Sims (1984) for makroøkonomiske prognoser.

## Kjente svakheter

- Full Gibbs-sampling over Σ er ikke implementert; OLS-estimert Σ kan undervurdere usikkerhet ved korte serier.
- Ikke-stasjonæritet (enhetrot) håndteres ikke eksplisitt.
- Kjøretid skalerer med `n_draws × n_vars²`; ved mange variabler bør `n_draws` reduseres.

## Referanser

- Litterman, R. (1986). *Forecasting with Bayesian vector autoregressions.* Review of Economics and Statistics.
- Doan, T., Litterman, R., & Sims, C. (1984). *Forecasting and conditional projection using realistic prior distributions.* Econometric Reviews.

## Evalueringsresultater

*(Fylles inn etter første backtest-kjøring på historiske data.)*
