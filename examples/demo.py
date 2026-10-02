"""Run the full audit on synthetic random-walk data and save the charts in docs/.

    python examples/demo.py
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from audit.data import random_walk_ohlc
from audit.signals import swing_low_entry
from audit.engine import run_trades
from audit.baseline import random_baseline
from audit.stats import t_stat, percentile_vs_random, block_bootstrap
from audit.costs import round_trip_cost_R

DOCS = os.path.join(os.path.dirname(__file__), "..", "docs")
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
                     "ytick.color": MUTED, "font.size": 11})


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID)
    ax.set_axisbelow(True)


def main():
    os.makedirs(DOCS, exist_ok=True)
    df = random_walk_ohlc(n_bars=30000, seed=7)
    spread = 0.0002                                       # 2 bps, a liquid CFD/index

    res = {}
    for name, bug in (("Look-ahead bug", True), ("Causal (correct)", False)):
        sig = swing_low_entry(df, k=3, lookahead_bug=bug)
        t = run_trades(df, sig)
        t["cost_R"] = [round_trip_cost_R(e, r, spread) for e, r in zip(t.entry, t.risk)]
        t["R_net"] = t.R - t.cost_R
        res[name] = t
    n_sig = int(len(res["Causal (correct)"]))
    rand = random_baseline(df, n_sig, runs=200)

    print(f"Data: driftless random walk, {len(df):,} hourly bars (no real edge exists)")
    print(f"Random-entry baseline: mean {np.nanmean(rand):+.3f}R gross")
    for name, t in res.items():
        print(f"{name:18s} trades {len(t):5d} | gross {t.R.mean():+.3f}R | net {t.R_net.mean():+.3f}R | "
              f"win {(t.R > 0).mean()*100:4.1f}% | t vs random {t_stat(t.R.values, np.nanmean(rand)):+.2f} | "
              f"beats {percentile_vs_random(t.R.mean(), rand):.0f}% of random runs")

    # chart 1: buggy vs causal vs random distribution
    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=120)
    ax.hist(rand, bins=12, color=BLUE, edgecolor="white", label="200 random-entry backtests")
    for (name, t), col, ls in zip(res.items(), (ORANGE, INK), ("-", "--")):
        ax.axvline(t.R.mean(), color=col, lw=2.5, ls=ls)
        right = col == ORANGE
        ax.text(t.R.mean() + (-0.02 if right else 0.02), ax.get_ylim()[1] * (0.92 if right else 0.78), f"{name}: {t.R.mean():+.2f}R",
                color=col, fontsize=11, weight="bold", ha="right" if right else "left")
    ax.set_xlabel("Average R per trade (gross)"); ax.set_ylabel("Count")
    ax.set_title("Same strategy on random-walk data: the 'edge' exists only with the bug", loc="left", fontsize=12)
    style(ax); fig.tight_layout(); fig.savefig(os.path.join(DOCS, "lookahead_vs_random.png")); plt.close(fig)

    # chart 2: cost sensitivity for the causal version
    t = res["Causal (correct)"]
    spreads = np.array([0, 1, 2, 4, 6, 8, 10]) / 1e4
    net = [np.mean(t.R - [round_trip_cost_R(e, r, s) for e, r in zip(t.entry, t.risk)]) for s in spreads]
    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=120)
    ax.plot(spreads * 1e4, net, color=BLUE, lw=2, marker="o", ms=8)
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xlabel("Round-trip spread (basis points)"); ax.set_ylabel("Average R per trade, net")
    ax.set_title("Cost sensitivity: how fast spread eats a strategy", loc="left", fontsize=12)
    style(ax); fig.tight_layout(); fig.savefig(os.path.join(DOCS, "cost_sensitivity.png")); plt.close(fig)

    # chart 3: Monte Carlo of the causal trade sequence at 1% risk per trade
    r = (t.R_net.values * 0.01)
    paths = block_bootstrap(r, horizon=min(500, len(r)), n_paths=2000, block=10)
    x = np.arange(1, paths.shape[1] + 1)
    q = np.percentile(paths, [5, 25, 50, 75, 95], axis=0)
    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=120)
    ax.fill_between(x, q[0], q[4], color=BLUE, alpha=0.18, lw=0, label="5-95% of paths")
    ax.fill_between(x, q[1], q[3], color=BLUE, alpha=0.40, lw=0, label="25-75%")
    ax.plot(x, q[2], color=ORANGE, lw=2, label="Median")
    ax.axhline(1, color=MUTED, lw=1)
    ax.set_xlabel("Trade number"); ax.set_ylabel("Equity (start = 1)")
    ax.set_title("Monte Carlo (block bootstrap): with no real edge, costs leave you going nowhere", loc="left", fontsize=12)
    ax.legend(frameon=False, loc="upper left")
    style(ax); fig.tight_layout(); fig.savefig(os.path.join(DOCS, "monte_carlo.png")); plt.close(fig)
    print("Charts saved to docs/")


if __name__ == "__main__":
    main()
