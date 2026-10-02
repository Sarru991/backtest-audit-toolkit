"""Swing-point 'discount zone' signal in two versions.

A swing low at bar p is a low lower than the k bars on each side. It can only
be *known* at bar p + k (when the right-hand bars have closed). The buggy
version marks the swing at bar p itself: that silently uses future bars.
"""
import numpy as np
import pandas as pd


def swing_lows(low: np.ndarray, k: int = 3) -> np.ndarray:
    n = len(low)
    out = np.zeros(n, dtype=bool)
    for p in range(k, n - k):
        w = low[p - k:p + k + 1]
        out[p] = low[p] == w.min() and (w == low[p]).sum() == 1
    return out


def zone_long_signal(df: pd.DataFrame, k: int = 3, lookahead_bug: bool = False) -> np.ndarray:
    """Long signal: price is back near the most recent swing low ('demand').

    lookahead_bug=True  -> swing becomes usable at bar p (WRONG, peeks k bars ahead)
    lookahead_bug=False -> swing becomes usable at bar p + k (causal)
    """
    low, close = df["low"].values, df["close"].values
    sw = swing_lows(low, k)
    n = len(low)
    sig = np.zeros(n, dtype=bool)
    level = np.nan
    delay = 0 if lookahead_bug else k
    known = np.zeros(n, dtype=bool)
    pos = np.where(sw)[0] + delay
    known[pos[pos < n]] = True
    swing_idx = -1
    for i in range(n):
        if known[i]:
            swing_idx = i - delay
            level = low[swing_idx]
        if not np.isnan(level) and i > swing_idx and abs(close[i] - level) / level < 0.0015:
            sig[i] = True
    return sig


def swing_low_entry(df: pd.DataFrame, k: int = 3, lookahead_bug: bool = False) -> np.ndarray:
    """'Buy the confirmed swing low' signal.

    lookahead_bug=True  -> signal fires at the swing bar p itself (WRONG: needs bars p+1..p+k)
    lookahead_bug=False -> signal fires at bar p + k, the first bar where the swing is known
    """
    sw = swing_lows(df["low"].values, k)
    n = len(sw)
    sig = np.zeros(n, dtype=bool)
    pos = np.where(sw)[0] + (0 if lookahead_bug else k)
    sig[pos[pos < n]] = True
    return sig
