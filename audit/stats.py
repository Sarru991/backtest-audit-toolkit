import numpy as np


def t_stat(x: np.ndarray, baseline_mean: float = 0.0) -> float:
    """t = mean(x - baseline) / (std / sqrt(n)). Need |t| >= 2 before trusting an edge."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return np.nan
    return (x.mean() - baseline_mean) / (x.std(ddof=1) / np.sqrt(len(x)))


def percentile_vs_random(strategy_mean: float, random_means: np.ndarray) -> float:
    """Share of random runs the strategy beats (0-100). Permutation-style p-value = 1 - this/100."""
    r = random_means[~np.isnan(random_means)]
    return float((r < strategy_mean).mean() * 100)


def block_bootstrap(returns: np.ndarray, horizon: int, n_paths: int = 2000,
                    block: int = 12, seed: int = 0) -> np.ndarray:
    """Equity paths from resampled blocks of consecutive returns (keeps autocorrelation)."""
    rng = np.random.default_rng(seed)
    r = np.asarray(returns, float)
    paths = np.empty((n_paths, horizon))
    for p in range(n_paths):
        idx = []
        while len(idx) < horizon:
            s = rng.integers(0, len(r) - block)
            idx.extend(range(s, s + block))
        paths[p] = np.cumprod(1 + r[idx[:horizon]])
    return paths
