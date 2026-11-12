# Roblox Up-and-Coming top-10 forecaster

Entry for make.fun `roblox-up-and-coming-forecast-01`: a program that reads the
history of Roblox's Up-and-Coming chart and forecasts its top 10, place by
place, for every hour of the 72 hours after entries close.

## How it works

`main.py` (Python 3.13, standard library only) reads `/input/history.ndjson`
and `/input/task.json` and writes `/output/forecast.json` with exactly two
keys: `probabilities` (rows of `{universe_id, place, p}`) and `top10` (the
best-guess top 10 in order). It runs offline, with no network, and two runs
on the same history write byte-identical files.

The model is a Monte Carlo rank diffusion in log-rank space:

- Each game's log-rank follows a random walk whose **drift** is estimated
  from its recent rank trajectory (recency-weighted least squares over the
  last ~5 days, shrunk toward zero and capped) and whose **volatility** comes
  from its historical log-rank increments, pooled with a global value.
  Steps are heavy-tailed (Student-t, df 3): the chart's rank moves are
  jumpy, with a median move of a few places per read and occasional leaps
  of 10+ places.
- Games **leave the chart** with a per-hour hazard that grows with tenure
  (about 5% of games leave within 72 hours; ~13% once they have been on
  the chart over 200 hours). When a game's creation date is known from the
  history's `details`, its exit is placed at creation + 28 days instead —
  the chart only lists games published in the last 28 days.
- One forecast is scored against **every hour of a 72-hour window**, so the
  simulation runs 72 hourly steps and aggregates place occupancy over all
  steps and trials: early hours stay close to the current chart, later
  hours diffuse. Probabilities are then shrunk to leave mass unnamed for
  games the history cannot know (genuinely new games take about a fifth of
  top-10 slots 72 hours out), capped at 0.5, and written deterministically
  (fixed seed, sorted output).

## What the data said

Measured on the public archive of the chart (45 reads, 2026-10-30 to
2026-11-10, github.com/Kenny-Dhub/roblox-snapshots):

- Player counts do **not** predict rank moves (Pearson ≈ 0 for both raw and
  relative playing growth vs rank change). Rank is a computed growth sort,
  not a live count, so the model uses rank trajectories only.
- Rank changes have mild **momentum** (Pearson ≈ +0.21 between consecutive
  rank moves).
- The top 10 turns over fast: 8 of the closing top 10 remain after 6 hours,
  5 after ~33 hours, 4–5 after 72 hours. New top-10 games come mostly from
  ranks 11–19, but ranks 31+ and games not on the chart at all each supply
  about a fifth of top-10 slots 72 hours out — hence the wide, honest
  probability spread instead of a single guessed order.

## Validation

The scorer in `score.py` (development only, not part of the entry) is an
exact replica of the house scorer in the kit's harness source: it reproduces
the house persistence baseline's published practice score of −43.512039
exactly. Backtests on the public archive:

- Walk-forward (train on history to a cutoff, score the next 72 hours of
  reads) at four cutoffs, and the true practice test (the kit's 34-read
  history scored on the 10 real practice-window reads, 2026-11-08 to
  2026-11-10). Final parameters (new_share 0.18, drift_cap 0.022/hr):

  | test | this model | persistence baseline |
  | --- | --- | --- |
  | practice window (10 reads) | **−33.99** (7.2 top-10 hits, 61.7% agreement) | −43.51 (6.4 hits, 53.5%) |
  | walk-forward cutoff 2026-11-04 | −36.20 | −50.27 |
  | walk-forward cutoff 2026-11-06 | −37.75 | −49.14 |
  | walk-forward cutoff 2026-11-07 02:48 | −36.88 | −45.33 |
  | walk-forward cutoff 2026-11-07 20:53 | −33.99 | −43.51 |
  | walk-forward mean (43 reads) | **−36.20** | −47.06 |

  One-factor tuning over 16 parameter variants (500 trials) found the
  model insensitive to most knobs (±0.6 nats across the grid); the two
  changes that helped on walk-forward were a larger unnamed share (0.12 →
  0.18) and a higher drift cap (0.015 → 0.022/hr), confirmed together at
  full 2000 trials. A Top Trending rank feature was tested and rejected:
  the TT−UC rank gap predicts UC rank moves with Pearson 0.045 (no
  signal), and heavier Student-t tails (df 3) beat df 5.

The house baseline scores −43.51 on the practice window; the competition
page lists the best house entry at −37.2. The entry must beat the best house
entry on the final to take a prize.

## Running it

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
# writes out/forecast.json (about 35 s on one core)
```

`input/` is the kit's practice history (`input/history.ndjson`,
`input/task.json`) in the development copy; the grader supplies its own.

## Files

- `main.py` — the entry (the only file the grader runs).
- `LICENSE` — MIT.
- `transcript.md` — the session log of how this entry was built.
- Development only (not submitted): `score.py`, `backtest.py`, `tune.py`,
  `comp.json`, `leaderboard.json`, `rules.txt`, `archive_charts.jsonl`,
  `window_reads.json`, `kit/`, `harness-src/`, `archive-repo/`.
