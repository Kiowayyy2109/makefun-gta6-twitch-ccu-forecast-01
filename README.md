# makefun-gta6-twitch-ccu-forecast-01

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Kiowayyy2109/makefun-gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Kiowayyy2109/makefun-gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecast GTA VI's launch-day Twitch viewers hour by hour. `main.py` (Python 3.13,
standard library only) reads one case file and, for each hour 2–24 of release
day 2026-11-19 UTC, writes five quantiles (5%, 25%, 50%, 75%, 95%) of the
live viewers in the "Grand Theft Auto VI" Twitch category.

## How it works

For every comparable past launch in the case file, the model forms log-ratios
`ln(viewers_h / viewers_1)` for hours 2–24. Hour 1's print is a strong
predictor of the day's shape: launches already fully unlocked at 00:00 UTC
stay near flat, while rolling-midnight launches whose main markets unlock
later rise several-fold by the evening. So for each hour it fits a
Gaussian-kernel-weighted local linear regression of the log-ratio on
`ln(viewers_1)` (bandwidth 1.5, slope clamped to ±1.5) and evaluates it at
the target's own hour-1 level. Quantiles are median ± z·(weighted residual
SD), floored at 0.10 log units, then lightly smoothed across hours.

Calibration: leave-one-out over the comparables plus practice launches scored
against SullyGnome hourly averages — practice mean pinball loss **0.190**
vs **0.415** for the house baseline; 90% coverage 0.89. Data:
SullyGnome.com hourly Twitch category averages.

## Run it

```sh
python3 /bot/main.py
```

The grader runs `python3 /bot/main.py` with the case file in `/input` and
expects the forecast in `/output`. Locally, set `MAKEFUN_INPUT_DIR` and
`MAKEFUN_OUTPUT_DIR` to point at your case and output directories. The
program is deterministic, needs no network, and runs in ~0.1 s per case.

Entered on [make.fun](https://make.fun/?ref=8b9bbsfs&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards
