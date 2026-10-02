import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from audit.data import random_walk_ohlc
from audit.signals import zone_long_signal, swing_lows, swing_low_entry
from audit.engine import run_trades
from audit.stats import t_stat, block_bootstrap
from audit.costs import slippage


def test_swing_low_needs_both_sides():
    low = np.array([5, 4, 3, 1, 3, 4, 5], float)
    assert swing_lows(low, k=3)[3] and swing_lows(low, k=3).sum() == 1


def test_causal_signal_never_uses_future_bars():
    df = random_walk_ohlc(3000, seed=1)
    full = zone_long_signal(df, k=3, lookahead_bug=False)
    cut = zone_long_signal(df.iloc[:2000], k=3, lookahead_bug=False)
    assert (full[:2000] == cut).all()          # adding future data must not change past signals


def test_buggy_signal_depends_on_future_bars():
    df = random_walk_ohlc(3000, seed=1)
    sig = swing_low_entry(df, 3, lookahead_bug=True)
    p = int(np.where(sig)[0][5])
    df2 = df.copy()
    df2.iloc[p + 2, df2.columns.get_loc("low")] = df.low.iloc[p] * 0.99   # change only a FUTURE bar
    assert not swing_low_entry(df2, 3, lookahead_bug=True)[p]            # past signal flips -> look-ahead


def test_engine_enters_next_open():
    df = random_walk_ohlc(500, seed=2)
    sig = np.zeros(len(df), bool); sig[100] = True
    t = run_trades(df, sig)
    assert t.bar.iloc[0] == 101 and t.entry.iloc[0] == df.open.iloc[101]


def test_stats_helpers():
    assert abs(t_stat(np.ones(10) + np.r_[0.1, -0.1] .repeat(5))) > 2
    p = block_bootstrap(np.full(60, 0.01), horizon=24, n_paths=10)
    assert p.shape == (10, 24) and np.allclose(p[:, -1], 1.01 ** 24)
    assert slippage(0.0002, 0.02, 1e5, 1e8) > 0.0001


def test_swing_entry_bug_vs_causal():
    df = random_walk_ohlc(3000, seed=3)
    assert (swing_low_entry(df, 3, False)[:2000] == swing_low_entry(df.iloc[:2000], 3, False)).all()
