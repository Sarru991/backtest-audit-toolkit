import numpy as np
import pandas as pd
from .engine import run_trades


def random_entries(df: pd.DataFrame, n_signals: int, seed: int = 0) -> np.ndarray:
    """Random signal mask with the same number of signals as the strategy."""
    rng = np.random.default_rng(seed)
    sig = np.zeros(len(df), dtype=bool)
    sig[rng.choice(np.arange(30, len(df) - 50), size=n_signals, replace=False)] = True
    return sig


def random_baseline(df: pd.DataFrame, n_signals: int, runs: int = 200, **trade_kw) -> np.ndarray:
    """Mean R of `runs` random-entry backtests using identical trade rules."""
    out = []
    for s in range(runs):
        t = run_trades(df, random_entries(df, n_signals, seed=s), **trade_kw)
        out.append(t.R.mean() if len(t) else np.nan)
    return np.array(out)
