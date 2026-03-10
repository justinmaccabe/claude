# config.py — central configuration for the portfolio system
# Edit this file to change tickers, signal parameters, or thresholds.

# ─── Universe ─────────────────────────────────────────────────────────────────

TICKERS = {
    # Core beta
    "SPY":  {"sleeve": "core",   "label": "US Large Cap",         "min_w": 0.03, "max_w": 0.20},
    "EFA":  {"sleeve": "core",   "label": "Intl Dev Equities",    "min_w": 0.05, "max_w": 0.25},
    "EEM":  {"sleeve": "core",   "label": "EM Equities",          "min_w": 0.05, "max_w": 0.22},
    "IEF":  {"sleeve": "core",   "label": "US Treasuries 7-10Y",  "min_w": 0.03, "max_w": 0.18},
    "SCHP": {"sleeve": "core",   "label": "US TIPS",              "min_w": 0.03, "max_w": 0.18},
    "HYG":  {"sleeve": "core",   "label": "US High Yield",        "min_w": 0.03, "max_w": 0.18},
    # Multifactor
    "QMOM": {"sleeve": "factor", "label": "US Momentum",          "min_w": 0.03, "max_w": 0.15},
    "QUAL": {"sleeve": "factor", "label": "US Quality",           "min_w": 0.03, "max_w": 0.15},
    "AVUV": {"sleeve": "factor", "label": "Small Cap Value+Prof", "min_w": 0.03, "max_w": 0.15},
    "KMLM": {"sleeve": "factor", "label": "Trend (Mount Lucas)",  "min_w": 0.03, "max_w": 0.12},
    "DBMF": {"sleeve": "factor", "label": "Managed Futures",      "min_w": 0.02, "max_w": 0.10},
}

BENCHMARK = "VGRO"          # Vanguard growth ETF (Canadian equiv of 2065 TDF)
BENCHMARK_LABEL = "VGRO (Vanguard All-in-One Growth)"
CASH_RATE_REAL = 1.8        # AQR 2026 real cash rate estimate
MARKET_VOL_TARGET = 16.5    # Target annualised vol % (US equity market vol)

# ─── Signal parameters ────────────────────────────────────────────────────────

SIGNAL = {
    "trend": {
        "lookback_months": 12,      # 12-1 momentum window
        "skip_months": 1,           # skip last month (avoid reversal)
        "assets": ["SPY", "EFA", "EEM", "QMOM", "KMLM", "DBMF"],
    },
    "carry": {
        "lookback_months": 12,
        "assets": ["HYG"],          # credit carry only — HYG OAS vs history
    },
    "value_spread": {
        "assets": ["SPY", "EFA", "EEM"],
    },
    "regime": {
        "low_vix":  18,
        "high_vix": 28,
    },
}

# ─── Portfolio engine ─────────────────────────────────────────────────────────

PORTFOLIO = {
    "rebalance_threshold": 0.02,
    "signal_overlay_max":  0.20,
    "optimizer_iterations": 5000,
}

# ─── Sleeve bounds (total allocation per sleeve) ─────────────────────────────

SLEEVE_BOUNDS = {
    "core":   (0.50, 0.65),
    "factor": (0.35, 0.50),
}

# ─── Data ─────────────────────────────────────────────────────────────────────

DATA = {
    "db_path":        "data/portfolio.db",
    "price_history":  "3y",
    "fred_series": {
        "hyg_oas":    "BAMLH0A0HYM2",  # ICE BofA HY OAS spread
        "vix":        "VIXCLS",
        "breakeven":  "T10YIE",
        "real_yield": "DFII10",
    },
}

# ─── Alerts ──────────────────────────────────────────────────────────────────

ALERTS = {
    "rebalance_drift":  0.05,
    "drawdown_pct":    -0.10,
    "signal_flip":      True,
}
