# signals/trend.py
# 12-1 time-series momentum signal.
#
# Logic:
#   - For each asset, compute total return over past 12 months, skip last month
#   - Positive = bullish (hold/overweight), negative = bearish (underweight/skip)
#   - Score is normalised cross-sectionally so you can rank assets
#
# Reference: Moskowitz, Ooi, Pedersen (2012) "Time Series Momentum"

import pandas as pd
import numpy as np
import logging

from config import SIGNAL, TICKERS

log = logging.getLogger(__name__)


class TrendSignal:
    def __init__(self, prices: pd.DataFrame):
        """
        prices: wide DataFrame, date (index) x ticker (columns), adjusted close.
        """
        self.prices = prices
        self.lookback = SIGNAL["trend"]["lookback_months"]
        self.skip     = SIGNAL["trend"]["skip_months"]
        self.assets   = [t for t in SIGNAL["trend"]["assets"] if t in prices.columns]

    def monthly_returns(self) -> pd.DataFrame:
        """Resample daily prices to month-end, compute monthly returns."""
        monthly = self.prices[self.assets].resample("ME").last()
        return monthly.pct_change()

    def momentum_score(self, as_of: pd.Timestamp | None = None) -> pd.Series:
        """
        12-1 momentum: cumulative return from t-12 to t-1.
        Returns a Series of raw momentum scores per asset.
        """
        monthly = self.monthly_returns()

        if as_of:
            monthly = monthly[monthly.index <= as_of]

        # Need at least lookback + skip months of data
        min_rows = self.lookback + self.skip + 1
        if len(monthly) < min_rows:
            log.warning(f"Insufficient history for trend signal ({len(monthly)} months, need {min_rows})")
            return pd.Series(dtype=float)

        # Cumulative return: months t-12 to t-1 (skip most recent month)
        window = monthly.iloc[-(self.lookback + self.skip): -self.skip]
        cum_ret = (1 + window).prod() - 1

        return cum_ret.rename("momentum_12_1")

    def signal(self, as_of: pd.Timestamp | None = None) -> pd.DataFrame:
        """
        Returns a DataFrame with columns:
          momentum_12_1  — raw 12-1 return
          signal         — +1 (bullish), -1 (bearish), 0 (neutral / insufficient data)
          z_score        — cross-sectional z-score (for relative ranking)
          percentile     — rank percentile across assets (0-100)
        """
        scores = self.momentum_score(as_of)
        if scores.empty:
            return pd.DataFrame()

        df = scores.to_frame()
        df["signal"]     = np.sign(df["momentum_12_1"]).astype(int)
        df["z_score"]    = (df["momentum_12_1"] - df["momentum_12_1"].mean()) / (df["momentum_12_1"].std() + 1e-9)
        df["percentile"] = df["momentum_12_1"].rank(pct=True) * 100

        return df.round(4)

    def signal_history(self) -> pd.DataFrame:
        """
        Compute signal for every month in history.
        Returns DataFrame: date (index) x (ticker, metric) MultiIndex columns.
        Useful for backtesting and dashboard charting.
        """
        monthly = self.monthly_returns()
        results = {}

        for i in range(self.lookback + self.skip, len(monthly)):
            as_of = monthly.index[i]
            sig   = self.signal(as_of=as_of)
            if not sig.empty:
                results[as_of] = sig["signal"]

        if not results:
            return pd.DataFrame()

        return pd.DataFrame(results).T

    def summary(self) -> str:
        """Human-readable signal summary for dashboard / alerts."""
        sig = self.signal()
        if sig.empty:
            return "Trend signals unavailable — insufficient history."

        lines = ["── Trend Signal (12-1 Momentum) ──────────────────"]
        for ticker, row in sig.iterrows():
            direction = "▲ BULLISH" if row["signal"] > 0 else "▼ BEARISH"
            lines.append(
                f"  {ticker:<6} {direction:<12}  "
                f"ret={row['momentum_12_1']:+.1%}  "
                f"z={row['z_score']:+.2f}  "
                f"pct={row['percentile']:.0f}"
            )
        return "\n".join(lines)
