# jobs/weekly.py
# Weekly signal run.
#
# Usage:
#   python3 -m jobs.weekly
#   python3 -m jobs.weekly --portfolio-value 100000
#   python3 -m jobs.weekly --dry-run

import argparse
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from config import TICKERS, DATA
from data.store import Store
from data.fetcher import fetch_prices
from signals.trend import TrendSignal
from signals.carry import CarrySignal
from signals.regime import RegimeSignal
from portfolio.allocator import Allocator, SignalOverlay

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── AQR 2026 CMA estimates — update each January ─────────────────────────────
EXPECTED_RETURNS = {
    "SPY":  3.5,
    "EFA":  6.0,
    "EEM":  7.5,
    "IEF":  2.0,
    "SCHP": 2.2,
    "HYG":  4.5,
    "QMOM": 7.5,
    "QUAL": 5.5,
    "AVUV": 7.0,
    "KMLM": 5.0,
    "DBMF": 4.8,
}
VOLS = {
    "SPY":  16.5,
    "EFA":  17.0,
    "EEM":  22.0,
    "IEF":   7.0,
    "SCHP":  5.5,
    "HYG":   9.5,
    "QMOM": 18.5,
    "QUAL": 15.5,
    "AVUV": 22.0,
    "KMLM": 12.0,
    "DBMF": 11.5,
}

# fmt: off
#              SPY    EFA    EEM    IEF   SCHP    HYG   QMOM   QUAL   AVUV   KMLM   DBMF
CORR_DATA = np.array([
    #SPY
    [ 1.00,  0.88,  0.80, -0.15, -0.05,  0.62,  0.75,  0.78,  0.82,  0.05,  0.08],
    #EFA
    [ 0.88,  1.00,  0.82, -0.10, -0.03,  0.58,  0.65,  0.72,  0.72,  0.05,  0.08],
    #EEM
    [ 0.80,  0.82,  1.00, -0.08, -0.01,  0.60,  0.60,  0.68,  0.68,  0.04,  0.06],
    #IEF
    [-0.15, -0.10, -0.08,  1.00,  0.70,  0.10, -0.10, -0.08, -0.12,  0.20,  0.18],
    #SCHP
    [-0.05, -0.03, -0.01,  0.70,  1.00,  0.15, -0.03, -0.02, -0.04,  0.15,  0.12],
    #HYG
    [ 0.62,  0.58,  0.60,  0.10,  0.15,  1.00,  0.55,  0.55,  0.55,  0.08,  0.10],
    #QMOM
    [ 0.75,  0.65,  0.60, -0.10, -0.03,  0.55,  1.00,  0.65,  0.72,  0.05,  0.08],
    #QUAL
    [ 0.78,  0.72,  0.68, -0.08, -0.02,  0.55,  0.65,  1.00,  0.70,  0.05,  0.08],
    #AVUV
    [ 0.82,  0.72,  0.68, -0.12, -0.04,  0.55,  0.72,  0.70,  1.00,  0.04,  0.06],
    #KMLM
    [ 0.05,  0.05,  0.04,  0.20,  0.15,  0.08,  0.05,  0.05,  0.04,  1.00,  0.75],
    #DBMF
    [ 0.08,  0.08,  0.06,  0.18,  0.12,  0.10,  0.08,  0.08,  0.06,  0.75,  1.00],
])
# fmt: on

TICKERS_LIST = list(EXPECTED_RETURNS.keys())
CORR_MATRIX  = pd.DataFrame(CORR_DATA, index=TICKERS_LIST, columns=TICKERS_LIST)


def run(portfolio_value: float = 100_000, dry_run: bool = False) -> dict:
    store = Store(DATA["db_path"])

    # 1. Fetch prices
    if not dry_run:
        fetch_prices(store, period="3mo")
    else:
        log.info("Dry run — skipping price fetch, using cached data")

    prices = store.read_prices(tickers=TICKERS_LIST + ["VGRO"])
    if prices.empty:
        log.error("No price data in DB. Run: python3 -m data.fetcher")
        return {}

    # 2. Signals
    trend  = TrendSignal(prices)
    carry  = CarrySignal(prices, store=store)
    regime = RegimeSignal(store=store, prices=prices)

    trend_df   = trend.signal()
    carry_df   = carry.signal()
    regime_det = regime.detect()

    print("\n" + trend.summary())
    print("\n" + carry.summary())
    print("\n" + regime.summary())

    # 3. Build overlay
    trend_signals = pd.Series(dtype=float)
    if not trend_df.empty:
        trend_signals = trend_df["signal"].reindex(TICKERS_LIST).fillna(0)

    carry_signals = pd.Series(dtype=float)
    if not carry_df.empty:
        carry_map = {"HYG  (credit spread)": "HYG"}
        cs = {}
        for label, ticker in carry_map.items():
            if label in carry_df.index:
                cs[ticker] = carry_df.loc[label, "signal"]
        carry_signals = pd.Series(cs)

    overlay = SignalOverlay(
        trend_signals=trend_signals,
        carry_signals=carry_signals,
        regime_tilts=regime.tilts(),
    )

    # 4. Compute target weights
    allocator = Allocator(EXPECTED_RETURNS, VOLS, CORR_MATRIX)
    target_w  = allocator.target_weights(overlay=overlay, levered=True)
    stats     = allocator.portfolio_stats(target_w)

    print(f"\n── Portfolio Stats ──────────────────────────────")
    for k, v in stats.items():
        print(f"  {k:<22} {v}")

    # 5. Trade list
    current_w = pd.Series({t: 0.0 for t in TICKERS_LIST})
    trades    = allocator.trade_list(current_w, portfolio_value, overlay=overlay)

    print(f"\n── Trade List (portfolio value ${portfolio_value:,.0f}) ────────")
    trades_to_do = trades[trades["trade_required"]]
    if trades_to_do.empty:
        print("  No trades required — all positions within threshold.")
    else:
        print(trades_to_do[["action", "current_weight", "target_weight", "drift", "trade_value"]].to_string())

    run_record = {
        "timestamp":      datetime.now(timezone.utc).isoformat(),
        "regime":         regime_det.value,
        "stats":          stats,
        "target_weights": target_w.round(4).to_dict(),
        "trades":         trades_to_do.to_dict(orient="index"),
    }

    return run_record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Weekly portfolio signal run")
    parser.add_argument("--portfolio-value", type=float, default=100_000)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    result = run(portfolio_value=args.portfolio_value, dry_run=args.dry_run)
    print(f"\n── Run complete: {result.get('timestamp', '')}")
