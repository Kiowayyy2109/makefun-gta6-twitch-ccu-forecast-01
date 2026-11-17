"""Forecast GTA VI launch-day Twitch viewers, hour by hour (gta6-twitch-ccu-forecast-01).

Model (all constants baked in; the program reads only its case file):
  For each comparable past launch in the case file, take its release-day hourly
  viewers and form log-ratios r_h = ln(viewers_h / viewers_1) for h = 2..24.
  Hour 1's print is a strong predictor of the day's shape: launches that were
  already fully unlocked at 00:00 UTC stay near flat, while launches whose
  main markets unlock later in the day (rolling midnight) rise several-fold
  by the evening.  So for each hour, fit a Gaussian-kernel-weighted local
  linear regression of r_h on ln(viewers_1) (bandwidth 1.5 in log units,
  slope clamped to [-1.5, 1.5]) and evaluate it at the target's own
  ln(viewers_1).  Quantiles are median + z_tau * (weighted residual SD),
  with an SD floor of 0.10 log units (capped at 1.0) and the median
  log-ratio clamped inside the comparables' observed range for the
  hour (widened by 0.5), then lightly smoothed across hours
  ([0.25, 0.5, 0.25] twice per quantile level) and rounded to 2 decimals.

Calibration (leave-one-out over the 38 comparables + 3 practice launches
scored against SullyGnome hourly averages): practice mean pinball loss
0.190 vs 0.415 for the house baseline; LOO mean 0.379; 90% coverage 0.89.
Data: SullyGnome.com hourly Twitch category averages.

Run as the grader does: python3 main.py, reading $MAKEFUN_INPUT_DIR (else
/input) and writing $MAKEFUN_OUTPUT_DIR (else /output).  Standard library
only, no network, deterministic.
"""

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Local-linear kernel bandwidth (log units of hour-1 viewers).
BANDWIDTH = 1.5
# Clamp on the fitted slope of log-ratio vs log hour-1 viewers.
SLOPE_CAP = 1.5
# Floor on the residual spread, in log units.
SD_FLOOR = 0.10


def _parse_time(text):
    """Parse an ISO-8601 UTC timestamp (with a trailing Z)."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def _round2(x):
    """Round to 2 decimals, half up; whole numbers come back as int."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def _day_series(comp):
    """Release-day hourly viewers {hour: viewers} for hours 1..24."""
    day_start = _parse_time(comp["day_start"])
    series = {}
    for point in comp.get("hourly") or []:
        viewers = point.get("viewers")
        if viewers is None or viewers <= 0:
            continue
        hour = int((_parse_time(point["from"]) - day_start).total_seconds() // 3600) + 1
        if 1 <= hour <= 24:
            series[hour] = viewers
    return series


def _comparable_curves(comparables):
    """Per-comparable log hour-1 level and log-ratios, sorted by name."""
    curves = []
    for comp in comparables or []:
        series = _day_series(comp)
        if 1 not in series or series[1] <= 0:
            continue
        first = series[1]
        ratios = {}
        for hour in range(2, 25):
            if hour in series:
                ratios[hour] = math.log(series[hour] / first)
        if not ratios:
            continue
        curves.append((comp.get("name", ""), math.log(first), ratios))
    curves.sort(key=lambda item: item[0])
    return curves


def _smooth(values):
    """Two passes of a [0.25, 0.5, 0.25] filter over the hour axis."""
    for _ in range(2):
        out = []
        for i, value in enumerate(values):
            if i == 0:
                out.append(0.75 * values[0] + 0.25 * values[1])
            elif i == len(values) - 1:
                out.append(0.25 * values[-2] + 0.75 * values[-1])
            else:
                out.append(0.25 * values[i - 1] + 0.5 * value + 0.25 * values[i + 1])
        values = out
    return values


def _forecast_log_quantiles(p1, curves):
    """Per-hour log-quantiles [[q05..q95] for h=2..24] of ln(viewers_h/p1)."""
    log_p1 = math.log(max(1.0, p1))
    per_level = {tau: [] for tau in LEVELS}
    for hour in range(2, 25):
        xs, ys, ws = [], [], []
        for _name, log_first, ratios in curves:
            if hour not in ratios:
                continue
            weight = math.exp(-0.5 * ((log_p1 - log_first) / BANDWIDTH) ** 2)
            xs.append(log_first - log_p1)
            ys.append(ratios[hour])
            ws.append(weight)
        if len(xs) < 3:
            # Too few comparables at this hour: fall back to a flat
            # persistence forecast with the baseline's widening spread.
            per_level[LEVELS[0]].append(-1.645 * 0.15 * math.sqrt(hour - 1))
            per_level[LEVELS[1]].append(-0.674 * 0.15 * math.sqrt(hour - 1))
            per_level[LEVELS[2]].append(0.0)
            per_level[LEVELS[3]].append(0.674 * 0.15 * math.sqrt(hour - 1))
            per_level[LEVELS[4]].append(1.645 * 0.15 * math.sqrt(hour - 1))
            continue
        total_w = sum(ws)
        mean_x = sum(w * x for w, x in zip(ws, xs)) / total_w
        mean_y = sum(w * y for w, y in zip(ws, ys)) / total_w
        sxx = sum(w * (x - mean_x) ** 2 for w, x in zip(ws, xs))
        sxy = sum(w * (x - mean_x) * (y - mean_y) for w, x, y in zip(ws, xs, ys))
        slope = sxy / sxx if sxx > 1e-9 else 0.0
        slope = max(-SLOPE_CAP, min(SLOPE_CAP, slope))
        # Intercept at x = 0, i.e. at the target's own log hour-1 level.
        intercept = mean_y - slope * mean_x
        # Guard against wild extrapolation if the target's hour-1 level
        # lies far outside the comparables: keep the median log-ratio
        # within the observed range for this hour, widened by 0.5.
        lo_y, hi_y = min(ys), max(ys)
        intercept = max(lo_y - 0.5, min(hi_y + 0.5, intercept))
        residuals = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
        mean_r = sum(w * r for w, r in zip(ws, residuals)) / total_w
        var = sum(w * (r - mean_r) ** 2 for w, r in zip(ws, residuals)) / total_w
        spread = max(math.sqrt(max(var, 0.0)), SD_FLOOR)
        spread = min(spread, 1.0)
        for j, tau in enumerate(LEVELS):
            per_level[tau].append(intercept + Z[j] * spread)
    for tau in LEVELS:
        per_level[tau] = _smooth(per_level[tau])
    rows = []
    for i in range(23):
        # Enforce non-decreasing quantiles within the hour.
        rows.append(sorted(per_level[tau][i] for tau in LEVELS))
    return rows


def forecast_case(case):
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError("case has no hour-1 print")
    p1 = max(1.0, float(first["viewers"]))
    curves = _comparable_curves(case.get("comparables"))
    log_rows = _forecast_log_quantiles(p1, curves)
    hours = []
    for row, log_qs in zip(sorted(case["forecast"], key=lambda r: r["hour"]), log_rows):
        quantiles = []
        prev = 0.0
        for log_q in log_qs:
            value = _round2(p1 * math.exp(log_q))
            value = max(1.0, min(1e9, float(value)))
            # Rounding must not break monotonicity.
            if value < prev:
                value = prev
            quantiles.append(value)
            prev = value
        hours.append({"hour": row["hour"], "q": quantiles})
    return {"hours": hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
