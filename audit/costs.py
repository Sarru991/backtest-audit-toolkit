import numpy as np


def slippage(spread: float, sigma: float, order_value: float, adv: float, k: float = 0.5) -> float:
    """Expected one-way slippage as a fraction of price.

    slip = 1/2 * spread + k * sigma * sqrt(order_value / ADV)

    spread      : quoted spread as a fraction of price (e.g. 0.0002)
    sigma       : daily return volatility (e.g. 0.02)
    order_value : size of the order in currency
    adv         : average daily traded value in currency
    k           : impact coefficient (0.3-1.0 is typical in the literature)
    """
    return 0.5 * spread + k * sigma * np.sqrt(max(order_value, 0.0) / max(adv, 1e-9))


def round_trip_cost_R(entry: float, stop_distance: float, spread: float,
                      commission_frac: float = 0.0) -> float:
    """Cost of one round trip expressed in R (multiples of the stop distance)."""
    return (spread * entry + 2 * commission_frac * entry) / stop_distance
