"""gta6-twitch-ccu-forecast-01, version 2: forecast GTA VI's launch-day Twitch
viewers hour by hour, from the first hour's print.

Model: similarity-weighted comparable-launch ratio curves.

For each hour h = 2..24, every past big launch in the case's `comparables`
contributes one log-ratio, ln(viewers_h / viewers_1), of its own release day.
The forecast's median log-ratio is the weighted median of those, with weights
from two Gaussian kernels:

  - how close the launch's hour-1 print is to the target's, in log viewers
    (width 1.5): launches whose first hour drew about as many viewers count most;
  - how close the launch's pre-release buzz growth is to the target's (width
    5.0). The growth is ln(viewers 24h before release) - ln(viewers 7d before
    release), read off the launch's hourly series (the target's own growth comes
    from the case's `history_hourly`). Launches whose audience was building at
    the same pace in the last week count most: a launch whose buzz was still
    exploding into release day keeps climbing through the day, while one whose
    buzz had already plateaued before release flattens out early.

The median log-ratio is shrunk toward 0 (persistence) for very big launches --
0.85 above 100k hour-1 viewers, 1.0 below 20k, linear in log viewers between --
because the biggest launches' curves run flatter than their neighbours' median
suggests. The spread is the weighted interquartile range of the same
log-ratios, turned into a log-normal sigma (IQR / 1.349), times 0.9, with a
0.05 floor. Hours with no comparable data fall back to the persistence
baseline (the starter's spread, 0.15 * sqrt(h - 1)).

Why two kernels: hour-1 magnitude alone separates the midnight-rollout
regime (small hour-1, most of the world not unlocked yet, curve climbs
steeply) from the already-global regime (huge hour-1, curve dips through the
US night and recovers by the evening peak), but among launches of similar
size the curves still spread widely, and the last week's buzz growth explains
a real part of that spread (correlation about -0.4 with the evening ratios:
fast pre-launch growth means the first hour was already inflated, so the day
grows less). Weighting on both cuts the error.

Validation (leave-one-out over the kit's 37 launch curves, pinball loss on
the log scale, mean over hours 2..24 and the five levels):
  - house persistence baseline: 0.930 over all 37, 0.251 over the 20 with
    hour-1 >= 50k, 0.349 over the 21 with hour-1 in 20k..300k
  - version 1 (hour-1 kernel only): 0.412 / 0.204 / 0.288
  - this model:                     0.391 / 0.184 / 0.258
The final's hour-1 print is expected larger still, where the margin over the
baseline is widest; the version-1 entry scored 0.247 on the practice phase.

How the grader runs it: `python3 -P -s /bot/main.py` (PYTHONHASHSEED=0) in
python:3.13-slim, standard library only: no network, no pip packages. -P keeps
this file's directory OFF sys.path, so the sys.path line below puts it back
first. Deterministic: no randomness, no set/dict-order dependence, every
case's output is a pure function of its input file.

What an entry must do:
  - read every <id>.in in the input directory (the grader runs one case at a
    time, with only that case's file there): JSON {"schema":
    "makefun/twitch-ccu-case/v1", "id", "game", "day_start", "checkpoint",
    "levels", "known", "forecast", "history", "history_hourly", "streams",
    "categories_hourly", "comparables", "notes"}, where `known` holds hour 1's
    print {hour: 1, from, until, viewers, channels, reads} and `forecast`
    lists the hours to forecast, 2..24;
  - write <id>.out to the output directory: {"hours": [{"hour": 2, "q":
    [q05, q25, q50, q75, q95]}, ...]}, exactly 23 rows, hours 2..24 each once,
    no other key, each q five finite numbers above 0 and at most 1e9
    (viewers; fractions allowed), non-decreasing; at most 65,536 bytes.
    Anything else is an invalid case, scored as if every quantile equalled
    hour 1's print, and an entry with an invalid case takes no prize.

Outside the grader's container, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR stand
in for /input and /output:
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
# Gaussian kernel width on the distance in pre-launch growth (log units).
GROWTH_WIDTH = 5.0
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
# Pre-launch growth is measured between these two offsets (hours before day_start).
GROWTH_RECENT_HOURS = 24.0
GROWTH_EARLIER_HOURS = 24.0 * 7.0
# A series point counts for a growth offset only when it is within this many hours
# of it (the series are hourly, so the nearest point is normally a few hours off).
GROWTH_TOLERANCE_HOURS = 72.0


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


def _nearest_log_viewers(points, target_offset: float) -> float | None:
    """Log viewers of the series point nearest to `target_offset` hours before
    day_start, or None when no point is close enough or positive.

    `points` is a list of (offset_hours, viewers) with negative offsets before
    day_start.
    """
    best = None
    for offset, viewers in points:
        if viewers is None or viewers <= 0:
            continue
        if best is None or abs(offset - target_offset) < abs(best[0] - target_offset):
            best = (offset, viewers)
    if best is None or abs(best[0] - target_offset) > GROWTH_TOLERANCE_HOURS:
        return None
    return math.log(best[1])


def prelaunch_growth(hourly, day_start: datetime) -> float | None:
    """ln(viewers 24h before release) - ln(viewers 7d before release), or None.

    `hourly` is a series of {from, viewers} points (a comparable's `hourly`, or a
    case's `history_hourly` / `history`). The growth measures how fast the
    audience was building in the last week before release.
    """
    points = []
    for point in hourly or []:
        try:
            viewers = point["viewers"]
            when = parse_time(point.get("from") or point.get("at"))
        except (KeyError, TypeError, ValueError):
            continue
        offset = (when - day_start).total_seconds() / 3600.0
        if offset < 0:
            points.append((offset, viewers))
    recent = _nearest_log_viewers(points, -GROWTH_RECENT_HOURS)
    earlier = _nearest_log_viewers(points, -GROWTH_EARLIER_HOURS)
    if recent is None or earlier is None:
        return None
    return recent - earlier


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
    """The valid launch-day curves in a case's comparables.

    Each is a dict with the log hour-1 viewers, the pre-launch growth, and the
    release-day curve (hour -> viewers). A comparable counts only when its
    hourly series is short (a real launch window, not the months-long
    pre-release history of the target's own category), its hour-1 print
    exists, and at least MIN_COVERAGE of its 24 release-day hours are present.
    Order is the case file's order, so results are deterministic.
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
        out.append(
            {
                "logp1": math.log(curve[1]),
                "growth": prelaunch_growth(hourly, day_start),
                "curve": curve,
            }
        )
    return out


def shrink_factor(log_p1: float) -> float:
    """How much the median log-ratio is shrunk toward 0 for this hour-1 size."""
    if log_p1 >= SHRINK_HI:
        return SHRINK_BIG
    if log_p1 <= SHRINK_LO:
        return 1.0
    return SHRINK_BIG + (1.0 - SHRINK_BIG) * (SHRINK_HI - log_p1) / (SHRINK_HI - SHRINK_LO)


def hour_quantiles(p1: float, log_p1: float, growth: float | None, hour: int, curves) -> list:
    """The five quantiles for one hour, from the comparable ratio curves."""
    pairs = []
    for entry in curves:
        curve = entry["curve"]
        if hour not in curve:
            continue
        weight = math.exp(-((log_p1 - entry["logp1"]) / KERNEL_WIDTH) ** 2)
        other_growth = entry["growth"]
        if growth is not None and other_growth is not None:
            weight *= math.exp(-((growth - other_growth) / GROWTH_WIDTH) ** 2)
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


def target_growth(case: dict) -> float | None:
    """The target's pre-launch growth, from its history (None when not measurable)."""
    try:
        day_start = parse_time(case["day_start"])
    except (KeyError, TypeError, ValueError):
        return None
    for key in ("history_hourly", "history"):
        growth = prelaunch_growth(case.get(key) or [], day_start)
        if growth is not None:
            return growth
    return None


def forecast_case(case: dict, name: str) -> dict:
    p1 = hour1_print(case)
    log_p1 = math.log(p1)
    growth = target_growth(case)
    curves = launch_curves(case.get("comparables"))
    rows = case.get("forecast") or [{"hour": hour} for hour in range(2, 25)]
    hours = []
    for row in rows:
        hour = row["hour"]
        hours.append({"hour": hour, "q": hour_quantiles(p1, log_p1, growth, hour, curves)})
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
