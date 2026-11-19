#!/usr/bin/env python3
"""Forecast GTA VI (PS5) trophy earn rates on Exophase.

Reads every case file in /input (env MAKEFUN_INPUT_DIR), writes one forecast
per case to /output (env MAKEFUN_OUTPUT_DIR): for each trophy, five quantiles
(5/25/50/75/95%) of its earn rate at the case target, scored by pinball loss
on the logit scale.

Model (stdlib only, deterministic, no network):
  * Deduplicate snapshots: only *distinct* consecutive reads (players tracked
    or any rate changed) carry information. Re-reads of a stale Exophase page
    would bias any trend fit, so they are dropped.
  * Primary fit: OLS of logit(rate) against x = ln(1 + hours since the
    2026-11-19T00:00:00Z release), per trophy, slopes shrunk 50/50 to the
    median slope of the trophy's tier (platinum/gold rise slowly, common
    bronze fast), intercept re-anchored at the last distinct snapshot. The
    logit form is the saturating one: a rate cannot pass 100%, so a linear
    fit to log(earners) over-extrapolates once rates get large (tested on a
    synthetic final-like case: a log-earners fit overshot a 50% trophy to
    64% and an 81% trophy to ~100%, losing to the house baseline).
  * Stabiliser: decompose rate = earners / players_tracked and fit OLS of
    log(earners) and log(players) against the same x; median rate =
    E_target / N_target. Blended in at only 15% (logit scale): it helps when
    the logit fit is noisy (few distinct snapshots, tiny earner counts) and
    when growth is closer to a power law, but its saturation bias must not
    dominate. A trophy at 0% uses E = 0.5 (continuity correction).
  * Honest, horizon-widening spread: sigma = sqrt(RMSE^2
    + slope_var * (x_target - x_mean)^2 + refresh_var), floored at 0.35 and
    capped at 1.8 logit units. refresh_var is the local slope times the
    typical log-time gap between distinct snapshots, squared: the answering
    read at the target reflects Exophase's last refresh before it, which can
    lag the target by hours. Quantiles come from the normal z-scores on
    logit(median). A trophy stuck at 100% keeps median ~100 with a very
    narrow spread.
  * Flat-history branch: with fewer than 2 distinct snapshots there is no
    trend signal at all, so use a refresh-dilution prior instead of a flat
    line: median = last rate x a tier factor (bronze 0.45, silver 0.35,
    gold/platinum 0.25; a 100% trophy stays 100), sigma ~1.2. Reason, from
    the launch-night recording: Exophase's first refresh jumped the tracked
    players 638 -> 3,868 and *every* rate except the 100% one fell, because a
    refresh releases a backlog of new players faster than earners accumulate.
    This branch cannot fire on the final (its history is rich); it only keeps
    the practice cases sane.
"""

import json
import math
import os
from datetime import datetime, timezone

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# Refresh-dilution prior (flat-history branch), per tier.
DILUTION = {"bronze": 0.45, "silver": 0.35, "gold": 0.25, "platinum": 0.25}
FLAT_SIGMA = 1.2
FULL_SIGMA = 0.10          # trophy at 100%: essentially no room to move
SIGMA_FLOOR = 0.35
SIGMA_CAP = 1.8
BLEND_EN = 0.15            # weight of the earners/players fit vs logit fit
                           # (tested: 0.7 loses to the house baseline on a
                           # logit-linear synthetic final; 0.15 beats it
                           # clearly in logit-linear, mild power-law, and
                           # extreme power-law worlds alike)


def clamp(x, lo=0.01, hi=99.99):
    return min(hi, max(lo, x))


def logit(pct):
    p = clamp(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg):
    lg = min(700.0, max(-700.0, lg))
    return 100.0 / (1.0 + math.exp(-lg))


def round4(x):
    """Round to 4 decimals, half up (as the house baseline does)."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def hours_since_release(t):
    return max(-0.99, (t - RELEASE).total_seconds() / 3600.0)


def x_of(t):
    """Log-time coordinate: ln(1 + hours since release)."""
    return math.log1p(max(0.0, hours_since_release(t)))


def ols(xs, ys):
    """Least squares y ~ a + b*x. Returns (a, b, rmse, sxx, n)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return my, 0.0, 0.0, 0.0, n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    rmse = math.sqrt(sum(r * r for r in resid) / n) if n else 0.0
    return a, b, rmse, sxx, n


def median_of(vals):
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return 0.0
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def quantiles(median_pct, sigma):
    lg = logit(median_pct)
    qs = [round4(clamp(inv_logit(lg + z * sigma))) for z in Z]
    # Enforce strictly increasing quantiles after rounding/clamping.
    for i in range(1, len(qs)):
        if qs[i] <= qs[i - 1]:
            qs[i] = round4(min(99.99, qs[i - 1] + 0.0001))
    for i in range(len(qs) - 2, -1, -1):
        if qs[i] >= qs[i + 1]:
            qs[i] = round4(max(0.01, qs[i + 1] - 0.0001))
    return qs


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    if not case["history"]:
        raise ValueError(f"{name} has no history")
    trophies = case["trophies"]
    target = parse_time(case["target"])
    x_target = x_of(target)

    # --- Deduplicate: keep only snapshots whose (players, rates) changed. ---
    distinct = []
    prev_key = None
    for h in case["history"]:
        key = (h["players_tracked"], tuple(h["rates"]))
        if key != prev_key:
            distinct.append(h)
            prev_key = key
    last = distinct[-1]
    last_rates = last["rates"]

    # Typical log-time gap between distinct snapshots (refresh cadence).
    xs_distinct = [x_of(parse_time(h["slot"])) for h in distinct]
    gaps = [b - a for a, b in zip(xs_distinct, xs_distinct[1:]) if b > a]
    gap_logx = median_of(gaps) if gaps else 0.0

    rows = []
    if len(distinct) < 2:
        # --- Flat-history branch: refresh-dilution prior, no trend signal. ---
        for i, trophy in enumerate(trophies):
            rate = last_rates[i]
            if rate is None:
                rate = 0.0
            if rate >= 99.99:
                rows.append({"id": trophy["id"], "q": quantiles(99.99, FULL_SIGMA)})
            elif rate <= 0.0:
                rows.append({"id": trophy["id"], "q": quantiles(0.01, FLAT_SIGMA)})
            else:
                med = clamp(rate * DILUTION.get(trophy["tier"], 0.45))
                rows.append({"id": trophy["id"], "q": quantiles(med, FLAT_SIGMA)})
        return {"trophies": rows}

    # --- Trend branch: logit fit (primary) + earners/players fit (stabiliser). ---
    a_n, b_n, _, _, _ = ols(xs_distinct,
                            [math.log(max(1, h["players_tracked"])) for h in distinct])
    n_target = max(1.0, math.exp(a_n + b_n * x_target))
    x_mean = sum(xs_distinct) / len(xs_distinct)

    # First pass: raw per-trophy logit slopes, for tier-median shrinkage.
    raw = []
    for i, trophy in enumerate(trophies):
        pts = [(x, logit(h["rates"][i])) for h, x in zip(distinct, xs_distinct)
               if h["rates"][i] is not None and 0.0 < h["rates"][i] < 100.0]
        if len(pts) >= 2:
            a_l, b_l, rmse_l, sxx_l, _ = ols([p[0] for p in pts], [p[1] for p in pts])
        elif pts:
            a_l, b_l, rmse_l, sxx_l = pts[0][1], 0.0, 0.5, 0.0
        else:
            a_l, b_l, rmse_l, sxx_l = logit(0.01), 0.0, 0.5, 0.0
        raw.append({"pts": pts, "a": a_l, "b": b_l, "rmse": rmse_l, "sxx": sxx_l})

    tier_slopes = {}
    for tier in {t["tier"] for t in trophies}:
        tier_slopes[tier] = median_of(
            [raw[i]["b"] for i, t in enumerate(trophies) if t["tier"] == tier])

    for i, trophy in enumerate(trophies):
        rate = last_rates[i]
        if rate is None:
            rate = 0.0
        if rate >= 99.99:
            rows.append({"id": trophy["id"], "q": quantiles(99.99, FULL_SIGMA)})
            continue
        r = raw[i]
        slope = 0.5 * r["b"] + 0.5 * tier_slopes[trophy["tier"]]
        # Re-anchor the intercept at the shrunk slope through the last point.
        if r["pts"]:
            x_last, y_last = r["pts"][-1]
            a_shrunk = y_last - slope * x_last
        else:
            a_shrunk = logit(max(rate, 0.01))
        med_logit = clamp(inv_logit(a_shrunk + slope * x_target))

        # Earners/players stabiliser: log E and log N fits, E_target / N_target.
        epts = []
        for h, x in zip(distinct, xs_distinct):
            rt = h["rates"][i]
            if rt is None:
                rt = 0.0
            epts.append((x, math.log(max(0.5, rt / 100.0 * h["players_tracked"]))))
        a_e, b_e, _, _, _ = ols([p[0] for p in epts], [p[1] for p in epts])
        med_en = clamp(math.exp(a_e + b_e * x_target) / n_target * 100.0)
        lg_med = BLEND_EN * logit(med_en) + (1.0 - BLEND_EN) * logit(med_logit)
        median = clamp(inv_logit(lg_med))

        # Spread: residual noise + slope uncertainty over the horizon
        # + refresh lag (answer reflects the last refresh before target).
        rmse = r["rmse"] if r["rmse"] > 0 else 0.5
        var_slope = (rmse * rmse / r["sxx"]) if r["sxx"] > 0 else 0.25
        refresh_var = (abs(slope) * gap_logx) ** 2
        sigma = math.sqrt(rmse * rmse
                          + var_slope * (x_target - x_mean) ** 2
                          + refresh_var)
        sigma = min(SIGMA_CAP, max(SIGMA_FLOOR, sigma))
        rows.append({"id": trophy["id"], "q": quantiles(median, sigma)})
    return {"trophies": rows}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for path in sorted(os.listdir(INPUT_DIR)):
        if not path.endswith(".in"):
            continue
        full = os.path.join(INPUT_DIR, path)
        case = json.loads(open(full, encoding="utf-8").read())
        forecast = forecast_case(case, path)
        out = os.path.join(OUTPUT_DIR, path[:-3] + ".out")
        with open(out, "w", encoding="utf-8") as f:
            f.write(json.dumps(forecast, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
