# GTA VI Trophy Rarity Forecast

Forecasts every GTA VI (PS5) trophy's earn rate on Exophase at 2026-11-26T00:00:00Z, as five quantiles (5/25/50/75/95%) per trophy, scored by pinball loss on the logit scale.

Entry for the make.fun competition `gta6-trophy-rarity-forecast-01` ($250, 20 places).

## How it works

`main.py` (Python 3.13 stdlib only, deterministic, no network) reads each case in `/input` and writes `/output/<id>.out`.

1. **Deduplicate snapshots** — only distinct consecutive reads (players tracked or any rate changed) carry information; re-reads of a stale Exophase page would bias any trend fit.
2. **Primary fit: OLS of logit(rate) against `ln(1 + hours since the 2026-11-19T00:00:00Z release)`**, per trophy, slopes shrunk 50/50 to the tier median (platinum/gold rise slowly, common bronze fast), intercept re-anchored at the last distinct snapshot. The logit form is the saturating one: a linear fit to log(earners) over-extrapolates once rates get large (tested on a synthetic final-like case: it overshot a 50% trophy to 64% and an 81% trophy to ~100%, losing to the house baseline). A 0% trophy uses E = 0.5 before logging; a 100% trophy stays ~100 with a very narrow spread.
3. **Stabiliser: decompose rate = earners / players tracked.** OLS of log(earners) and log(players) against the same x; median rate = E_target / N_target, blended in at only 15% (logit scale). It helps when the logit fit is noisy (few distinct snapshots, tiny earner counts) and when growth is closer to a power law, but its saturation bias must not dominate. Tested: the 15% blend beats the house baseline clearly on synthetic finals generated under logit-linear, mild power-law, and extreme power-law truth alike.
4. **Honest, horizon-widening spread** — sigma = sqrt(RMSE² + slope variance × (x_target − x_mean)² + refresh variance), floored at 0.35 and capped at 1.8 logit units. The refresh term reflects that the answering read shows Exophase's last refresh before the target, which can lag by hours.
5. **Flat-history branch** — with fewer than 2 distinct snapshots there is no trend signal, so a refresh-dilution prior is used instead of a flat line: median = last rate × tier factor (bronze 0.45, silver 0.35, gold/platinum 0.25). Launch night proves why: Exophase's first refresh jumped tracked players 638 → 3,868 and every rate except the 100% one fell, because a refresh releases a backlog of new players faster than earners accumulate. This branch cannot fire on the final (rich history); it only keeps the practice cases sane.

## Run locally

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## License

MIT — see `LICENSE`.
