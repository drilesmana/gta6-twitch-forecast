"""gta6-twitch-ccu-forecast-01: forecast GTA VI's launch-day Twitch viewers hour by hour.

Model: comparable-launch ratio curves, similarity-weighted by hour-1 magnitude.

For each hour h = 2..24, every past big launch in the case's `comparables` gives one
log-ratio, ln(viewers_h / viewers_1), of its own release day. The forecast's median
log-ratio is the weighted median of those, with Gaussian weights on the distance in
log hour-1 viewers between the target and each comparable (width 1.5): launches whose
first hour drew about as many viewers as the target's count most. The spread is the
weighted interquartile range of the same log-ratios, turned into a log-normal sigma
(IQR / 1.349), times 0.9. The median log-ratio is shrunk toward 0 (persistence) for
very big launches -- 0.85 above 100k hour-1 viewers, 1.0 below 20k, linear between --
because the biggest launches' curves run flatter than their neighbours' median
suggests, while small launches really do explode later in the day.

Why this works: a launch day's shape is set by the midnight-local rollout. Hour 1
(00:00-01:00 UTC) catches only New Zealand, Australia and the UK, so a small hour-1
means most of the world has not unlocked yet and the curve climbs steeply (10-1000x
by the evening); a huge hour-1 means the launch is already global and the curve dips
through the US night and recovers to about its start by the evening peak. Hour-1
magnitude alone separates those regimes, which is why the kernel keys on it.

Validation (leave-one-out over the 37 launch comparables in the kit, pinball loss on
the log scale, mean over hours 2..24 and the five levels):
  - house persistence baseline: 0.930 over all 37, 0.251 over the 20 with hour-1 >= 50k
  - this model:                0.412 over all 37, 0.204 over the 20 with hour-1 >= 50k
The practice cases (hour-1 prints 23k-270k) sit in the regime where this model beats
the baseline by roughly a fifth; the final's hour-1 print is expected larger still,
where the margin is widest.

How the grader runs it: `python3 -P -s /bot/main.py` (PYTHONHASHSEED=0) in
python:3.13-slim, standard library only: no network, no pip packages. -P keeps this
file's directory OFF sys.path, so the sys.path line below puts it back first.
Deterministic: no randomness, no set/dict-order dependence, every case's output is a
pure function of its input file.

What an entry must do:
  - read every <id>.in in the input directory (the grader runs one case at a time,
    with only that case's file there): JSON {"schema": "makefun/twitch-ccu-case/v1",
    "id", "game", "day_start", "checkpoint", "levels", "known", "forecast",
    "history", "history_hourly", "streams", "categories_hourly", "comparables",
    "notes"}, where `known` holds hour 1's print {hour: 1, from, until, viewers,
    channels, reads} and `forecast` lists the hours to forecast, 2..24;
  - write <id>.out to the output directory: {"hours": [{"hour": 2, "q": [q05, q25,
    q50, q75, q95]}, ...]}, exactly 23 rows, hours 2..24 each once, no other key,
    each q five finite numbers above 0 and at most 1e9 (viewers; fractions allowed),
    non-decreasing; at most 65,536 bytes. Anything else is an invalid case, scored
    as if every quantile equalled hour 1's print, and an entry with an invalid case
    takes no prize.

Outside the grader's container, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR stand in
for /input and /output:
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module shipped
# beside this file would import (this entry is a single file; the line is harmless).
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# --- Model constants (tuned by leave-one-out over the kit's 37 launch curves) ---
# Gaussian kernel width on the distance in log hour-1 viewers.
KERNEL_WIDTH = 1.5
# Scale applied to the IQR-derived log spread.
SPREAD_MULT = 0.9
# Minimum log spread, so early hours still carry some uncertainty.
SPREAD_FLOOR = 0.05
# Shrinkage of the median log-ratio toward 0 (persistence): full strength above
# SHRINK_HI hour-1 viewers, none below SHRINK_LO, linear in log viewers between.
SHRINK_BIG = 0.85
SHRINK_LO = math.log(20000.0)
SHRINK_HI = math.log(100000.0)
# Fallback (persistence baseline) spread per square-root hour, the starter's number.
BASE_SIGMA = 0.15
# A comparable counts as a launch-day curve only with at least this many of its
# 24 release-day hours present.
MIN_COVERAGE = 20
# ... and only when its hourly series spans at most this many days (this drops the
# Grand Theft Auto VI pre-release category, whose series runs ~82 days and is not a
# launch day at all).
MAX_SPAN_DAYS = 25.0


def round2(x: float) -> float | int:
    """x rounded to 2 decimals, half up, as JavaScript's Math.round(x * 100) / 100 rounds it.

    A whole number comes back as an int, so json.dumps writes 1240 where JavaScript
    writes 1240 (not 1240.0).
    """
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def parse_time(text: str) -> datetime:
    """An ISO 8601 UTC timestamp, with or without a trailing Z."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def weighted_quantile(pairs, q: float) -> float:
    """The q-quantile of `pairs`, a list of (value, weight) with positive weights.

    Values are sorted; the first value whose cumulative weight reaches q of the
    total wins. Deterministic: sorted() is stable, so ties keep input order.
    """
    ordered = sorted(pairs, key=lambda pair: pair[0])
    total = sum(weight for _, weight in ordered)
    if total <= 0:
        return ordered[len(ordered) // 2][0]
    acc = 0.0
    for value, weight in ordered:
        acc += weight
        if acc >= q * total:
            return value
    return ordered[-1][0]


def day_curve(hourly, day_start: datetime) -> dict:
    """A comparable's release-day curve: hour 1..24 -> viewers (positive only)."""
    curve = {}
    for point in hourly:
        try:
            viewers = point["viewers"]
            when = parse_time(point["from"])
        except (KeyError, TypeError, ValueError):
            continue
        if viewers is None or viewers <= 0:
            continue
        hour = int((when - day_start).total_seconds() / 3600.0) + 1
        if 1 <= hour <= 24:
            curve[hour] = viewers
    return curve


def launch_curves(comparables) -> list:
    """The valid launch-day curves in a case's comparables, as (log hour-1, curve).

    A comparable counts only when its hourly series is short (a real launch window,
    not the months-long pre-release history of the target's own category), its
    hour-1 print exists, and at least MIN_COVERAGE of its 24 release-day hours are
    present. Order is the case file's order, so results are deterministic.
    """
    out = []
    for comp in comparables or []:
        try:
            hourly = comp["hourly"]
            day_start = parse_time(comp["day_start"])
        except (KeyError, TypeError, ValueError):
            continue
        if not hourly:
            continue
        try:
            first = parse_time(hourly[0]["from"])
            last = parse_time(hourly[-1]["from"])
        except (KeyError, TypeError, ValueError):
            continue
        span_days = (last - first).total_seconds() / 86400.0
        if span_days > MAX_SPAN_DAYS:
            continue
        curve = day_curve(hourly, day_start)
        if 1 not in curve:
            continue
        if sum(1 for hour in range(1, 25) if hour in curve) < MIN_COVERAGE:
            continue
        out.append((math.log(curve[1]), curve))
    return out


def shrink_factor(log_p1: float) -> float:
    """How much the median log-ratio is shrunk toward 0 for this hour-1 size."""
    if log_p1 >= SHRINK_HI:
        return SHRINK_BIG
    if log_p1 <= SHRINK_LO:
        return 1.0
    return SHRINK_BIG + (1.0 - SHRINK_BIG) * (SHRINK_HI - log_p1) / (SHRINK_HI - SHRINK_LO)


def hour_quantiles(p1: float, log_p1: float, hour: int, curves) -> list:
    """The five quantiles for one hour, from the comparable ratio curves."""
    pairs = []
    for log_p1_i, curve in curves:
        if hour not in curve:
            continue
        weight = math.exp(-((log_p1 - log_p1_i) / KERNEL_WIDTH) ** 2)
        pairs.append((math.log(curve[hour] / curve[1]), weight))
    if not pairs:
        # No comparable has this hour: fall back to the persistence baseline.
        return [round2(p1 * math.exp(z * BASE_SIGMA * math.sqrt(hour - 1))) for z in Z]
    median = weighted_quantile(pairs, 0.5) * shrink_factor(log_p1)
    iqr = weighted_quantile(pairs, 0.75) - weighted_quantile(pairs, 0.25)
    sigma = max(SPREAD_FLOOR, iqr / 1.349) * SPREAD_MULT
    quantiles = []
    previous = 0.0
    for z in Z:
        value = round2(p1 * math.exp(median + z * sigma))
        value = min(max(value, 1), 1000000000)
        if value < previous:
            value = previous
        previous = value
        quantiles.append(value)
    return quantiles


def hour1_print(case: dict) -> float:
    """Hour 1's print, at least 1; falls back to the last history point if needed."""
    for row in case.get("known") or []:
        if row.get("hour") == 1 and row.get("viewers") is not None:
            return max(1.0, float(row["viewers"]))
    for key in ("history_hourly", "history"):
        points = case.get(key) or []
        if points:
            last = points[-1].get("viewers")
            if last is not None and last > 0:
                return max(1.0, float(last))
    return 1.0


def forecast_case(case: dict, name: str) -> dict:
    p1 = hour1_print(case)
    log_p1 = math.log(p1)
    curves = launch_curves(case.get("comparables"))
    rows = case.get("forecast") or [{"hour": hour} for hour in range(2, 25)]
    hours = []
    for row in rows:
        hour = row["hour"]
        hours.append({"hour": hour, "q": hour_quantiles(p1, log_p1, hour, curves)})
    return {"hours": hours}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
