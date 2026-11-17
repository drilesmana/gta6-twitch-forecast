# gta6-twitch-forecast

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/drilesmana/gta6-twitch-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/drilesmana/gta6-twitch-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecast GTA VI's launch-day Twitch viewers hour by hour — a program that reads the
Twitch viewer count of Grand Theft Auto VI up to the end of the first hour of its
release day, 2026-11-19 (UTC), and forecasts how many people will be watching it
live on Twitch in each of the other 23 hours, as five quantiles an hour. Built by an
AI agent for make.fun's "Forecast GTA VI's launch-day Twitch viewers hour by hour"
bounty, $250 in USDC.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

`input` is the case directory (the kit's `input/` holds the four practice cases
`p00.in`–`p03.in`); each case writes one forecast to `out/<id>.out`. With neither
variable set the program reads `/input` and writes `/output`, as the grader runs it:
`python3 -P -s /bot/main.py` on Python 3.13, standard library only, no network.

## How it works

For each hour 2–24, every past big launch in the case's `comparables` gives one
log-ratio, ln(viewers_h / viewers_1), of its own release day. The forecast's median
log-ratio is the weighted median of those, with Gaussian weights on two
similarities: how close the launch's hour-1 print is to the target's, in log
viewers (width 1.5), and how close its pre-release buzz growth is — ln(viewers
24h before release) minus ln(viewers 7d before release), from the launch's hourly
series (the target's own growth comes from the case's `history_hourly`) — width
5.0. Launches whose audience was building at the same pace in the last week
count most. The spread is the weighted interquartile range of the same
log-ratios, turned into a log-normal sigma (IQR / 1.349), times 0.9. The median
log-ratio is shrunk toward 0 (persistence) for very big launches — 0.85 above
100k hour-1 viewers, 1.0 below 20k, linear between — because the biggest
launches' curves run flatter than their neighbours' median suggests.

Leave-one-out over the kit's 37 launch curves, pinball loss on the log scale:
the house persistence baseline scores 0.930 over all 37 (0.251 over the 20 with
hour-1 ≥ 50k, 0.349 over the 21 with hour-1 in 20k–300k); this model scores
0.391 over all 37 (0.184 over those 20, 0.258 over those 21). Version 1, which
weighted on hour-1 size only, scored 0.412 / 0.204 / 0.288, and 0.247 on the
practice phase.

---

<sub>Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
