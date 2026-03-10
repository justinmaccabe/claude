# portfolio/allocator.py
# Combines base optimizer weights with signal overlays (trend, carry, regime).
# Produces final target weights and a trade list vs current holdings.
#
# Flow:
#   1. Run max-Sharpe optimizer → base weights
#   2. Apply regime tilt (VIX-based)
#   3. Apply trend overlay (scale down bearish trend assets)
#   4. Apply carry overlay (scale up high-carry assets)
#   5. Clip to per-asset [min, max] bounds and renormalise
#   6. Diff vs current holdings → trade list

import numpy as np
import pandas as pd
import logging
from dataclasses import dataclass, field

from config import TICKERS, PORTFOLIO, CASH_RATE_REAL, MARKET_VOL_TARGET

log = logging.getLogger(__name__)


# ─── Max-Sharpe Optimizer (same logic as the React prototype) ─────────────────

def max_sharpe_weights(
    expected_returns: dict[str, float],
    vols: dict[str, float],
    corr_matrix: pd.DataFrame,
) -> pd.Series:
    """
    Gradient-ascent max-Sharpe optimisation with per-asset bounds.
    Returns a pd.Series of weights indexed by ticker.
    """
    tickers = list(expected_returns.keys())
    n = len(tickers)
    rets = np.array([expected_returns[t] for t in tickers])
    vs   = np.array([vols[t] for t in tickers])
    corr = corr_matrix.loc[tickers, tickers].values

    # Build covariance matrix
    cov = np.outer(vs, vs) * corr / 10000

    min_w = np.array([TICKERS[t]["min_w"] for t in tickers])
    max_w = np.array([TICKERS[t]["max_w"] for t in tickers])

    w = np.full(n, 1.0 / n)
    lr = 0.03

    for _ in range(PORTFOLIO["optimizer_iterations"]):
        mu    = w @ rets
        Mw    = cov @ w
        sigma = np.sqrt(max(w @ Mw, 1e-10))
        grad  = (rets * sigma * 100 - (mu - CASH_RATE_REAL) * (Mw / sigma)) / (sigma ** 2 * 10000)
        w     = np.clip(w + lr * grad, min_w, max_w)
        w    /= w.sum()

    return pd.Series(w, index=tickers)


def lever_to_target_vol(weights: pd.Series, vols: dict, corr: pd.DataFrame) -> tuple[pd.Series, float]:
    """Scale weights so portfolio vol matches MARKET_VOL_TARGET."""
    tickers = weights.index.tolist()
    vs  = np.array([vols[t] for t in tickers])
    c   = corr.loc[tickers, tickers].values
    cov = np.outer(vs, vs) * c / 10000
    w   = weights.values
    port_vol = np.sqrt(max(w @ cov @ w, 1e-10)) * 100
    leverage = MARKET_VOL_TARGET / port_vol
    return weights * leverage, leverage


# ─── Signal overlay ───────────────────────────────────────────────────────────

@dataclass
class SignalOverlay:
    trend_signals:  pd.Series = field(default_factory=pd.Series)   # ticker → signal (+1/0/-1)
    carry_signals:  pd.Series = field(default_factory=pd.Series)   # ticker → signal (+1/0/-1)
    regime_tilts:   dict      = field(default_factory=dict)         # ticker → weight adjustment


def apply_overlays(base_weights: pd.Series, overlay: SignalOverlay) -> pd.Series:
    """
    Adjusts base weights using signal overlays.
    Each signal can shift a weight by ±SIGNAL_OVERLAY_MAX.
    """
    max_shift = PORTFOLIO["signal_overlay_max"]
    w = base_weights.copy()

    # Trend: bearish signal → reduce weight by up to max_shift
    for ticker, sig in overlay.trend_signals.items():
        if ticker in w.index and sig != 0:
            shift = sig * max_shift * 0.5   # half weight to trend
            w[ticker] = max(w[ticker] + w[ticker] * shift, TICKERS[ticker]["min_w"])

    # Carry: similar
    for ticker, sig in overlay.carry_signals.items():
        if ticker in w.index and sig != 0:
            shift = sig * max_shift * 0.3   # smaller carry weight
            w[ticker] = max(w[ticker] + w[ticker] * shift, TICKERS[ticker]["min_w"])

    # Regime tilts: direct additive adjustment
    for ticker, tilt in overlay.regime_tilts.items():
        if ticker in w.index:
            w[ticker] = np.clip(
                w[ticker] + tilt,
                TICKERS[ticker]["min_w"],
                TICKERS[ticker]["max_w"],
            )

    # Renormalise to sum to 1
    w = w.clip(lower=pd.Series({t: TICKERS[t]["min_w"] for t in w.index}))
    w /= w.sum()
    return w


# ─── Trade list ───────────────────────────────────────────────────────────────

def generate_trades(
    target_weights: pd.Series,
    current_weights: pd.Series,
    portfolio_value: float,
) -> pd.DataFrame:
    """
    Compare target vs current weights.
    Returns DataFrame of trades to execute (only where drift > threshold).
    """
    threshold = PORTFOLIO["rebalance_threshold"]
    all_tickers = target_weights.index.union(current_weights.index)

    rows = []
    for ticker in all_tickers:
        target  = target_weights.get(ticker, 0.0)
        current = current_weights.get(ticker, 0.0)
        drift   = target - current

        rows.append({
            "ticker":         ticker,
            "current_weight": round(current, 4),
            "target_weight":  round(target, 4),
            "drift":          round(drift, 4),
            "trade_required": abs(drift) >= threshold,
            "trade_value":    round(drift * portfolio_value, 2),
            "action":         "BUY" if drift > threshold else "SELL" if drift < -threshold else "HOLD",
        })

    df = pd.DataFrame(rows).set_index("ticker")
    return df.sort_values("drift", ascending=False)


# ─── Main allocator ───────────────────────────────────────────────────────────

class Allocator:
    def __init__(
        self,
        expected_returns: dict[str, float],
        vols: dict[str, float],
        corr_matrix: pd.DataFrame,
    ):
        self.expected_returns = expected_returns
        self.vols = vols
        self.corr = corr_matrix
        self._base_weights = None

    def base_weights(self) -> pd.Series:
        """Max-Sharpe optimizer weights (cached)."""
        if self._base_weights is None:
            self._base_weights = max_sharpe_weights(
                self.expected_returns, self.vols, self.corr
            )
        return self._base_weights

    def target_weights(self, overlay: SignalOverlay | None = None, levered: bool = True) -> pd.Series:
        """Final target weights with optional signal overlay and leverage."""
        w = self.base_weights().copy()

        if overlay:
            w = apply_overlays(w, overlay)

        if levered:
            w, lev = lever_to_target_vol(w, self.vols, self.corr)
            log.info(f"Leverage factor: {lev:.2f}x")

        return w

    def trade_list(
        self,
        current_weights: pd.Series,
        portfolio_value: float,
        overlay: SignalOverlay | None = None,
    ) -> pd.DataFrame:
        target = self.target_weights(overlay)
        return generate_trades(target, current_weights, portfolio_value)

    def portfolio_stats(self, weights: pd.Series) -> dict:
        """Compute portfolio vol, return, Sharpe for a given weight vector."""
        tickers = weights.index.tolist()
        w   = weights.values
        vs  = np.array([self.vols[t] for t in tickers])
        c   = self.corr.loc[tickers, tickers].values
        cov = np.outer(vs, vs) * c / 10000
        rets = np.array([self.expected_returns[t] for t in tickers])

        port_vol = np.sqrt(max(w @ cov @ w, 1e-10)) * 100
        port_ret = w @ rets
        sharpe   = (port_ret - CASH_RATE_REAL) / port_vol

        return {
            "expected_return": round(port_ret, 3),
            "volatility":      round(port_vol, 3),
            "sharpe_ratio":    round(sharpe, 3),
            "gross_exposure":  round(weights.abs().sum(), 3),
        }
