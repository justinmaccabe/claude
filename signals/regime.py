# signals/regime.py
# VIX-based market regime detection — 4 states.
#
# CALM     VIX < 15   — risk-on tilt
# NEUTRAL  VIX 15-20  — base weights unchanged
# ELEVATED VIX 20-28  — mild defensive tilt
# STRESS   VIX > 28   — full defensive tilt

import pandas as pd
import logging
from enum import Enum
from config import SIGNAL

log = logging.getLogger(__name__)

VIX_CALM    = 15
VIX_NEUTRAL = 20
VIX_STRESS  = 28

class Regime(str, Enum):
    CALM     = "CALM"
    NEUTRAL  = "NEUTRAL"
    ELEVATED = "ELEVATED"
    STRESS   = "STRESS"

REGIME_TILTS = {
    Regime.CALM:     {"QMOM":+0.15,"EEM":+0.10,"HYG":+0.08,"IEF":-0.08,"SCHP":-0.05,"QUAL":-0.05},
    Regime.NEUTRAL:  {},
    Regime.ELEVATED: {"QUAL":+0.08,"SCHP":+0.06,"IEF":+0.05,"QMOM":-0.06,"EEM":-0.07,"HYG":-0.05},
    Regime.STRESS:   {"QUAL":+0.12,"SCHP":+0.10,"IEF":+0.08,"QMOM":-0.10,"EEM":-0.10,"HYG":-0.08},
}

class RegimeSignal:
    def __init__(self, store=None, prices=None):
        self.store  = store
        self.prices = prices

    def _vix_series(self, as_of=None):
        if not (self.store and "vix" in self.store.available_macro()):
            return None
        vix = self.store.read_macro("vix")
        if as_of:
            vix = vix[vix.index <= as_of]
        return vix if not vix.empty else None

    def current_vix(self, as_of=None):
        vix = self._vix_series(as_of)
        return float(vix.iloc[-1]) if vix is not None else None

    def vix_percentile(self, as_of=None):
        vix = self._vix_series(as_of)
        if vix is None or len(vix) < 252:
            return None
        return float((vix.iloc[-252:] <= vix.iloc[-1]).mean() * 100)

    def vix_history(self, days=252, as_of=None):
        vix = self._vix_series(as_of)
        return vix.iloc[-days:] if vix is not None else None

    def detect(self, as_of=None):
        vix = self._vix_series(as_of)
        if vix is None or len(vix) < 20:
            log.info("VIX unavailable — defaulting to NEUTRAL")
            return Regime.NEUTRAL
        v = float(vix.rolling(20).mean().iloc[-1])
        if v < VIX_CALM:      return Regime.CALM
        elif v < VIX_NEUTRAL: return Regime.NEUTRAL
        elif v < VIX_STRESS:  return Regime.ELEVATED
        else:                  return Regime.STRESS

    def tilts(self, as_of=None):
        return REGIME_TILTS[self.detect(as_of)].copy()

    def summary(self, as_of=None):
        regime  = self.detect(as_of)
        vix     = self.current_vix(as_of)
        vix_pct = self.vix_percentile(as_of)
        icons   = {"CALM":"🟢","NEUTRAL":"🔵","ELEVATED":"🟠","STRESS":"🔴"}
        vix_str = f"VIX={vix:.1f}" if vix else "VIX=unavailable"
        pct_str = f"  ({vix_pct:.0f}th pct)" if vix_pct else ""
        tilts   = self.tilts(as_of)
        lines   = [f"── Regime ──────────────", f"  {icons.get(regime.value,'')} {regime.value}  {vix_str}{pct_str}"]
        if tilts:
            lines.append("  Tilts:")
            for t, v in tilts.items():
                lines.append(f"    {t:<6} {v:+.0%}")
        else:
            lines.append("  No tilt (NEUTRAL)")
        return "\n".join(lines)
