#!/usr/bin/env python3
"""Roblox Up-and-Coming top-10 forecaster (make.fun roblox-up-and-coming-forecast-01).

Reads /input/history.ndjson (hourly-ish snapshots of Roblox's Up-and-Coming chart)
and /input/task.json, writes /output/forecast.json: for each of the top 10 places,
a probability for each game named being at that place, plus the best-guess top 10.

Method: Monte Carlo rank diffusion in log-rank space.
  * Each game's log-rank follows a random walk with a drift estimated from its
    recent rank trajectory (weighted least squares, shrunk toward zero) and a
    volatility estimated from its historical log-rank increments (pooled with a
    global value), with heavy-tailed Student-t steps.
  * Games leave the chart (age out at 28 days from creation, or otherwise
    disappear) with a per-hour hazard that grows with tenure; when a game's
    creation date is known from the history's `details`, its exit hour is
    computed from it (created + 28 days), otherwise the hazard is driven by
    time since the game was first seen on the chart.
  * One forecast is scored against every hour of a 72-hour window, so the
    simulation runs 72 hourly steps and aggregates place occupancy over all
    steps and trials: early hours stay close to the current chart, later
    hours diffuse.  Probabilities are then shrunk to leave mass unnamed for
    games the history cannot know (genuinely new games), capped, and written
    deterministically (fixed seed, sorted output).

Standard library only.  Two runs on the same history write byte-identical files.
"""
import json
import math
import os
import random
import sys
from datetime import datetime, timezone

PARAMS = {
    # Monte Carlo
    "trials": 2000,
    "horizon_hours": 72,
    "seed": 20261113,
    # Drift (log-rank per hour) from a weighted fit over the last FIT_HOURS.
    "fit_hours": 120.0,
    "drift_shrink": 2.0,        # pseudo-count: mu *= n_eff / (n_eff + drift_shrink)
    "drift_cap": 0.022,         # |mu| at most this many log-rank per hour
    "drift_decay": 48.0,        # recency weight half-life-ish (exp decay hours)
    # Volatility (log-rank per sqrt(hour)).
    "sigma_floor": 0.02,
    "sigma_cap": 0.12,
    "sigma_pool": 5.0,          # pseudo-count pooling a game's sigma to the global one
    "t_df": 3.0,                # Student-t degrees of freedom for steps
    # Mean reversion toward the game's recent median log-rank (per hour).
    "reversion": 0.0,
    # Exit hazard per hour: base + tenure term; when `created` is known the
    # exit is placed at created + 28 days with a small jitter instead.
    "hazard_base": 0.0007,
    "hazard_tenure": 0.0013,    # extra hazard once tenure passes HAZARD_TENURE_H
    "hazard_tenure_h": 200.0,
    "exit_jitter_h": 6.0,
    # Calibration of the written probabilities.
    "new_share": 0.18,          # mass left unnamed at each place (new games)
    "p_cap": 0.50,              # no single named probability above this
    "p_min": 0.0015,            # drop named rows below this
}

AGE_LIMIT_HOURS = 28.0 * 24.0  # a game leaves the chart 28 days after creation


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def load_history(input_dir):
    """Return (reads, created) where reads is a list of (time, {uid: rank})
    oldest first, and created maps uid -> creation datetime (when known)."""
    path = os.path.join(input_dir, "history.ndjson")
    reads = []
    created = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            at = parse_time(rec["at"])
            chart = {}
            for g in rec.get("up_and_coming") or []:
                try:
                    chart[int(g["universe_id"])] = int(g["rank"])
                except (KeyError, TypeError, ValueError):
                    continue
            if chart:
                reads.append((at, chart))
            details = rec.get("details")
            if isinstance(details, dict):
                for key, d in details.items():
                    if not isinstance(d, dict):
                        continue
                    c = d.get("created")
                    if c:
                        try:
                            uid = int(key)
                        except (TypeError, ValueError):
                            continue
                        if uid not in created:
                            try:
                                created[uid] = parse_time(c)
                            except ValueError:
                                pass
    reads.sort(key=lambda r: r[0])
    return reads, created


def build_series(reads):
    """Per-game time series: uid -> list of (hours_since_last_read, log_rank)."""
    if not reads:
        return {}, None
    t_last = reads[-1][0]
    series = {}
    for at, chart in reads:
        h = (at - t_last).total_seconds() / 3600.0  # <= 0
        for uid, rank in chart.items():
            series.setdefault(uid, []).append((h, math.log(max(1, rank))))
    return series, t_last


def fit_game(obs, params, global_sigma):
    """Fit drift mu (log-rank/hour) and volatility sigma for one game.

    obs: list of (hours, log_rank), oldest first, hours <= 0.
    Returns (mu, sigma, tenure_hours, last_log_rank)."""
    n = len(obs)
    if n == 0:
        return 0.0, global_sigma, 0.0, math.log(50.0)
    last_lr = obs[-1][1]
    tenure = -obs[0][0]
    if n < 2:
        return 0.0, global_sigma, tenure, last_lr
    fit_h = params["fit_hours"]
    decay = params["drift_decay"]
    pts = [(h, lr) for (h, lr) in obs if h >= -fit_h]
    if len(pts) < 2:
        pts = obs[-2:]
    # Weighted least squares of log-rank on hours (weights decay with age).
    sw = sx = sy = sxx = sxy = 0.0
    for h, lr in pts:
        w = math.exp(h / decay)  # h <= 0: recent points weigh ~1
        sw += w
        sx += w * h
        sy += w * lr
        sxx += w * h * h
        sxy += w * h * lr
    denom = sw * sxx - sx * sx
    if abs(denom) < 1e-12:
        mu = 0.0
    else:
        mu = (sw * sxy - sx * sy) / denom
    n_eff = min(len(pts), 8.0)
    mu *= n_eff / (n_eff + params["drift_shrink"])
    mu = max(-params["drift_cap"], min(params["drift_cap"], mu))
    # Volatility from log-rank increments, per sqrt(hour).
    sq = []
    for (h0, lr0), (h1, lr1) in zip(obs, obs[1:]):
        dt = h1 - h0
        if dt > 0:
            sq.append(((lr1 - lr0) / math.sqrt(dt)) ** 2)
    if sq:
        var_g = sum(sq) / len(sq)
        m = params["sigma_pool"]
        var = (len(sq) * var_g + m * global_sigma * global_sigma) / (len(sq) + m)
        sigma = math.sqrt(var)
    else:
        sigma = global_sigma
    sigma = max(params["sigma_floor"], min(params["sigma_cap"], sigma))
    return mu, sigma, tenure, last_lr


def global_sigma(series):
    """Pooled per-hour log-rank increment volatility over all games."""
    sq = []
    for obs in series.values():
        for (h0, lr0), (h1, lr1) in zip(obs, obs[1:]):
            dt = h1 - h0
            if dt > 0:
                sq.append(((lr1 - lr0) / math.sqrt(dt)) ** 2)
    if not sq:
        return 0.05
    return max(0.02, min(0.12, math.sqrt(sum(sq) / len(sq))))


def student_t(rng, df):
    """One Student-t draw via normal / sqrt(chi2/df)."""
    z = rng.gauss(0.0, 1.0)
    chi2 = 2.0 * rng.gammavariate(df / 2.0, 1.0)
    return z / math.sqrt(max(chi2, 1e-9) / df)


def simulate(games, params, t_last, created):
    """Monte Carlo over the horizon.  games: list of dicts with mu, sigma,
    tenure, x0 (log-rank), exit_hour (float or None), uid.
    Returns (counts, rank_sums): counts[place][uid] = trial-hours at that
    place; rank_sums[uid] = sum of simulated ranks (for the top-10 order)."""
    rng = random.Random(params["seed"])
    trials = params["trials"]
    horizon = params["horizon_hours"]
    n = len(games)
    counts = [dict() for _ in range(10)]
    rank_sums = [0.0] * n
    rank_counts = [0] * n
    df = params["t_df"]
    rev = params["reversion"]
    for _ in range(trials):
        x = [g["x0"] for g in games]
        alive = [True] * n
        exit_at = []
        for g in games:
            if g["exit_hour"] is not None:
                e = g["exit_hour"] + rng.gauss(0.0, params["exit_jitter_h"])
                exit_at.append(max(0.0, e))
            else:
                exit_at.append(None)
        for step in range(1, horizon + 1):
            for i, g in enumerate(games):
                if not alive[i]:
                    continue
                if exit_at[i] is not None and step > exit_at[i]:
                    alive[i] = False
                    continue
                if exit_at[i] is None and rng.random() < g["hazard"]:
                    alive[i] = False
                    continue
                dx = g["mu"] + g["sigma"] * student_t(rng, df)
                if rev:
                    dx += -rev * (x[i] - g["x_med"])
                x[i] += dx
                if x[i] < math.log(0.5):
                    x[i] = math.log(0.5)
                elif x[i] > math.log(250.0):
                    x[i] = math.log(250.0)
            order = sorted((i for i in range(n) if alive[i]), key=lambda i: x[i])
            for place, i in enumerate(order[:10]):
                uid = games[i]["uid"]
                d = counts[place]
                d[uid] = d.get(uid, 0) + 1
            # record each alive game's rank for the top-10 ordering
            for r, i in enumerate(order):
                rank_sums[i] += r + 1
                rank_counts[i] += 1
    total_steps = trials * horizon
    return counts, rank_sums, rank_counts, total_steps


def forecast(input_dir, params=None):
    params = dict(PARAMS if params is None else params)
    with open(os.path.join(input_dir, "task.json"), "r", encoding="utf-8") as f:
        task = json.load(f)
    places = int(task.get("places", 10))
    reads, created = load_history(input_dir)
    if not reads:
        return {"probabilities": [], "top10": []}
    series, t_last = build_series(reads)
    last_chart = reads[-1][1]
    gsig = global_sigma(series)
    games = []
    for uid, rank in sorted(last_chart.items()):
        obs = series.get(uid, [])
        mu, sigma, tenure, last_lr = fit_game(obs, params, gsig)
        hazard = params["hazard_base"]
        if tenure > params["hazard_tenure_h"]:
            hazard += params["hazard_tenure"]
        exit_hour = None
        c = created.get(uid)
        if c is not None:
            age_h = (t_last - c).total_seconds() / 3600.0
            exit_hour = AGE_LIMIT_HOURS - age_h
            if exit_hour < 0:
                exit_hour = 0.0
        x_med = statistics_median([lr for _, lr in obs]) if obs else last_lr
        games.append({
            "uid": uid, "mu": mu, "sigma": sigma, "tenure": tenure,
            "x0": last_lr, "x_med": x_med, "hazard": hazard,
            "exit_hour": exit_hour,
        })
    counts, rank_sums, rank_counts, total_steps = simulate(games, params, t_last, created)
    # Convert counts to probabilities, shrunk for unknown new games.
    shrink = 1.0 - params["new_share"]
    probs = []
    for place in range(places):
        d = counts[place]
        for uid, cnt in d.items():
            p = cnt / total_steps * shrink
            if p < params["p_min"]:
                continue
            p = min(p, params["p_cap"])
            probs.append({"universe_id": uid, "place": place + 1, "p": round(p, 6)})
    # Renormalise per place if capping pushed a sum over 1 (it cannot exceed
    # shrink, but rounding may nudge it; enforce the cap explicitly).
    sums = {}
    for row in probs:
        sums[row["place"]] = sums.get(row["place"], 0.0) + row["p"]
    for row in probs:
        s = sums[row["place"]]
        if s > 1.0:
            row["p"] = round(row["p"] / s, 6)
    probs.sort(key=lambda r: (r["place"], -r["p"], r["universe_id"]))
    # Top 10 by mean simulated rank.
    mean_rank = {}
    for i, g in enumerate(games):
        if rank_counts[i]:
            mean_rank[g["uid"]] = rank_sums[i] / rank_counts[i]
    order = sorted(mean_rank, key=lambda u: (mean_rank[u], u))
    top10 = order[:places]
    # If fewer than `places` games ever ranked, pad with the last chart's order.
    if len(top10) < places:
        for uid in sorted(last_chart, key=lambda u: last_chart[u]):
            if uid not in top10:
                top10.append(uid)
            if len(top10) == places:
                break
    return {"probabilities": probs, "top10": top10}


def statistics_median(vals):
    vals = sorted(vals)
    m = len(vals)
    if m == 0:
        return 0.0
    if m % 2:
        return vals[m // 2]
    return 0.5 * (vals[m // 2 - 1] + vals[m // 2])


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    with open(os.path.join(input_dir, "task.json"), "r", encoding="utf-8") as f:
        task = json.load(f)
    out_name = task.get("output", "forecast.json")
    result = forecast(input_dir)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, out_name), "w", encoding="utf-8") as f:
        f.write(json.dumps(result) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
