import numpy as np
import pandas as pd


def random_walk_ohlc(n_bars: int = 20000, seed: int = 7, start: float = 100.0,
                     vol: float = 0.002, freq: str = "h") -> pd.DataFrame:
    """Hourly OHLC bars from a driftless random walk.

    Because the series has no edge by construction, any strategy that 'wins'
    on it is either luck or a bug. That makes it the perfect test bench.
    """
    rng = np.random.default_rng(seed)
    steps = 12                                   # intrabar path points
    r = rng.normal(0.0, vol / np.sqrt(steps), size=(n_bars, steps))
    path = start * np.exp(np.cumsum(r.ravel())).reshape(n_bars, steps)
    o = np.r_[start, path[:-1, -1]]
    h = np.maximum(o, path.max(axis=1))
    l = np.minimum(o, path.min(axis=1))
    c = path[:, -1]
    idx = pd.date_range("2020-01-01", periods=n_bars, freq=freq)
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c}, index=idx)
