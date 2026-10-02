"""backtest-audit-toolkit: check whether a trading backtest is real.

Modules
-------
data      : synthetic random-walk prices (no edge by construction) for honest demos
signals   : swing-point zone signal, in a buggy (look-ahead) and a causal version
engine    : next-bar-open fills, stop/target management, R-multiples
costs     : spread + commission + square-root market-impact slippage model
baseline  : random-entry control with matched side, stop and holding rules
stats     : t-test vs baseline, block bootstrap Monte Carlo
"""
__version__ = "0.1.0"
