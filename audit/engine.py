"""Minimal, honest trade engine.

Rules that keep it honest:
* signal at the close of bar i -> entry at the OPEN of bar i+1
* stop checked before target inside each bar (worst case)
* if stop and target are both inside the entry bar, count the stop
"""
import numpy as np
import pandas as pd


def run_trades(df: pd.DataFrame, signal: np.ndarray, stop_atr: float = 1.0,
               target_R: float = 2.0, max_hold: int = 48, atr_len: int = 20) -> pd.DataFrame:
    o, h, l, c = (df[x].values for x in ("open", "high", "low", "close"))
    tr = np.maximum(h - l, np.maximum(abs(h - np.r_[c[0], c[:-1]]), abs(l - np.r_[c[0], c[:-1]])))
    atr = pd.Series(tr).rolling(atr_len).mean().values
    n, rows, i = len(c), [], atr_len
    while i < n - 2:
        if not signal[i] or np.isnan(atr[i]):
            i += 1
            continue
        e = i + 1
        entry = o[e]
        risk = stop_atr * atr[i]
        stop, target = entry - risk, entry + target_R * risk
        R, j = None, e
        for j in range(e, min(e + max_hold, n)):
            if l[j] <= stop:
                R = (min(o[j], stop) - entry) / risk if j > e else -1.0
                break
            if h[j] >= target and j > e:
                R = target_R
                break
        if R is None:
            R = (c[j] - entry) / risk
        rows.append((df.index[e], e, entry, risk, R, j - e + 1))
        i = j + 1
    return pd.DataFrame(rows, columns=["time", "bar", "entry", "risk", "R", "bars_held"])
