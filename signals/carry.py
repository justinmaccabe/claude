# signals/carry.py
# Carry signals across FX (DBV), commodities (PDBC), and credit (HYG).
#
# Since we're using ETFs rather than futures, we proxy carry as:
#   FX carry (DBV):         12m total return vs cash (SGOV) — ETF already IS the carry trade
#   Commodity carry (PDBC): 12m total return vs spot commodity index (DJP as proxy)
#                           positive gap = PDBC capturing roll yield above spot
#   Credit carry (HYG):     OAS spread vs its 3yr average (from FRED if available)
#                           wide spread = high carry, above avg = attractive
#
# These are "is carry currently being rewarded?" signals, not raw carry levels.

import pandas as pd
import numpy as np
import logging

from config import SIGNAL, DATA

log = logging.getLogger(__name__)

CASH_PROXY    = "SGOV"   # iShares 0-3M Treasury — cash rate proxy
COMMODITY_SPOT = "DJP"   # iPath Bloomberg Commodity — no roll optimisation (spot proxy)


class CarrySignal:
    def __init__(self, prices: pd.DataFrame, store=None):
        """
        prices: wide DataFrame of adjusted closes (must include DBV, PDBC, HYG + proxies)
        store:  optional Store instance to pull FRED macro (HYG OAS spread)
        """
        self.prices = prices
        self.store  = store
        self.lookback = SIGNAL["carry"]["lookback_months"]

    # ── FX Carry ──────────────────────────────────────────────────────────────

    def fx_carry(self, as_of: pd.Timestamp | None = None) -> dict:
        """
        DBV 12m return minus cash proxy 12m return.
        Positive = carry trade being rewarded.
        """
        if "DBV" not in self.prices or CASH_PROXY not in self.prices:
            return {"value": None, "signal": 0, "note": "DBV or SGOV not in prices"}

        px = self.prices[["DBV", CASH_PROXY]]
        if as_of:
            px = px[px.index <= as_of]

        monthly = px.resample("ME").last().pct_change()
        if len(monthly) < self.lookback + 2:
            return {"value": None, "signal": 0, "note": "Insufficient history"}

        window     = monthly.iloc[-self.lookback:]
        dbv_ret    = (1 + window["DBV"]).prod() - 1
        cash_ret   = (1 + window[CASH_PROXY]).prod() - 1
        carry_ret  = dbv_ret - cash_ret

        # Historical distribution for z-score
        rolling = (
            (1 + monthly["DBV"]).rolling(self.lookback).apply(lambda x: x.prod()) - 1
            - (1 + monthly[CASH_PROXY]).rolling(self.lookback).apply(lambda x: x.prod()) - 1
        )
        z = (carry_ret - rolling.mean()) / (rolling.std() + 1e-9)

        return {
            "value":   round(carry_ret, 4),
            "z_score": round(z, 2),
            "signal":  int(np.sign(carry_ret)),
            "note":    f"DBV {dbv_ret:+.1%} vs cash {cash_ret:+.1%}"
        }

    # ── Commodity Roll Yield ──────────────────────────────────────────────────

    def commodity_carry(self, as_of: pd.Timestamp | None = None) -> dict:
        """
        PDBC 12m return minus DJP (spot proxy) 12m return.
        Positive gap = PDBC capturing positive roll yield (backwardation).
        """
        cols = [c for c in ["PDBC", COMMODITY_SPOT] if c in self.prices.columns]
        if len(cols) < 2:
            return {"value": None, "signal": 0, "note": "PDBC or DJP not in prices — add to fetcher"}

        px = self.prices[["PDBC", COMMODITY_SPOT]]
        if as_of:
            px = px[px.index <= as_of]

        monthly = px.resample("ME").last().pct_change()
        if len(monthly) < self.lookback + 2:
            return {"value": None, "signal": 0, "note": "Insufficient history"}

        window    = monthly.iloc[-self.lookback:]
        pdbc_ret  = (1 + window["PDBC"]).prod() - 1
        spot_ret  = (1 + window[COMMODITY_SPOT]).prod() - 1
        roll_yield = pdbc_ret - spot_ret

        return {
            "value":   round(roll_yield, 4),
            "signal":  int(np.sign(roll_yield)),
            "note":    f"PDBC {pdbc_ret:+.1%} vs DJP spot {spot_ret:+.1%}"
        }

    # ── Credit Carry (HYG OAS) ────────────────────────────────────────────────

    def credit_carry(self, as_of: pd.Timestamp | None = None) -> dict:
        """
        HYG OAS spread vs its 3yr rolling average.
        Uses FRED data if available; falls back to ETF price momentum proxy.
        """
        # Try FRED first
        if self.store and "hyg_oas" in self.store.available_macro():
            oas = self.store.read_macro("hyg_oas")
            if as_of:
                oas = oas[oas.index <= as_of]
            if len(oas) > 60:
                current   = oas.iloc[-1]
                avg_3yr   = oas.iloc[-756:].mean()   # ~3 trading years
                z         = (current - avg_3yr) / (oas.iloc[-756:].std() + 1e-9)
                # Wide spread = higher carry = bullish signal for HYG
                signal    = 1 if current > avg_3yr * 0.9 else -1
                return {
                    "value":   round(current, 2),
                    "avg_3yr": round(avg_3yr, 2),
                    "z_score": round(z, 2),
                    "signal":  signal,
                    "note":    f"OAS {current:.0f}bps vs 3yr avg {avg_3yr:.0f}bps"
                }

        # Fallback: use HYG price momentum as proxy
        if "HYG" not in self.prices.columns:
            return {"value": None, "signal": 0, "note": "HYG not in prices"}

        px = self.prices[["HYG"]]
        if as_of:
            px = px[px.index <= as_of]

        monthly = px.resample("ME").last().pct_change()
        if len(monthly) < self.lookback + 2:
            return {"value": None, "signal": 0, "note": "Insufficient history"}

        ret    = (1 + monthly["HYG"].iloc[-self.lookback:]).prod() - 1
        signal = int(np.sign(ret))

        return {
            "value":  round(ret, 4),
            "signal": signal,
            "note":   f"HYG {self.lookback}m return proxy (add FRED key for OAS)"
        }

    # ── Combined ──────────────────────────────────────────────────────────────

    def signal(self, as_of: pd.Timestamp | None = None) -> pd.DataFrame:
        """
        Returns a DataFrame summarising all carry signals.
        """
        rows = {
            "DBV  (FX carry)":         self.fx_carry(as_of),
            "PDBC (commodity roll)":   self.commodity_carry(as_of),
            "HYG  (credit spread)":    self.credit_carry(as_of),
        }
        records = []
        for name, r in rows.items():
            records.append({
                "asset":  name,
                "value":  r.get("value"),
                "signal": r.get("signal", 0),
                "z_score":r.get("z_score"),
                "note":   r.get("note", ""),
            })
        return pd.DataFrame(records).set_index("asset")

    def summary(self, as_of: pd.Timestamp | None = None) -> str:
        sig = self.signal(as_of)
        lines = ["── Carry Signal ──────────────────────────────────"]
        for name, row in sig.iterrows():
            direction = "▲ POSITIVE" if row["signal"] > 0 else "▼ NEGATIVE" if row["signal"] < 0 else "── NEUTRAL"
            z_str = f"  z={row['z_score']:+.2f}" if pd.notna(row.get("z_score")) else ""
            lines.append(f"  {name:<28} {direction}  {row['note']}{z_str}")
        return "\n".join(lines)
