# Portfolio System

AQR 2026 CMA-based portfolio with signal-driven weekly overlay.
Three sleeves: Core Beta / Multifactor / Carry. Max Sharpe, levered to market vol.

## Setup

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Get a free FRED API key for macro data (VIX, HYG OAS, TIPS yield)
#    https://fred.stlouisfed.org/docs/api/api_key.html
export FRED_API_KEY=your_key_here

# 4. Initialise database (downloads 3yr of price history)
python -m data.fetcher

# 5. Run weekly signal job
python -m jobs.weekly --portfolio-value 100000

# 6. Launch dashboard
streamlit run monitor/dashboard.py
```

## Weekly workflow

```bash
# Every week (add to cron or run manually)
python -m jobs.weekly --portfolio-value 100000
```

To automate on Mac (runs every Monday 7am):
```
crontab -e
0 7 * * 1 cd /path/to/portfolio-system && source venv/bin/activate && python -m jobs.weekly
```

## Structure

```
config.py               — tickers, signal params, sleeve bounds
data/
  fetcher.py            — yfinance price download + FRED macro
  store.py              — SQLite read/write
signals/
  trend.py              — 12-1 momentum signal
  carry.py              — FX / commodity / credit carry
  regime.py             — VIX regime detection
portfolio/
  allocator.py          — max-Sharpe optimizer + signal overlay + trade list
monitor/
  dashboard.py          — Streamlit dashboard
jobs/
  weekly.py             — weekly signal run (prices + signals + trades)
```

## Tickers

| Ticker | Sleeve     | Description                    |
|--------|------------|-------------------------------|
| SPY    | Core       | US Large Cap                  |
| EFA    | Core       | Intl Developed Equities       |
| EEM    | Core       | EM Equities                   |
| IEF    | Core       | US Treasuries 7-10Y           |
| SCHP   | Core       | US TIPS                       |
| HYG    | Core       | US High Yield Credit          |
| QMOM   | Factor     | US Momentum                   |
| QUAL   | Factor     | US Quality                    |
| QMIX   | Factor     | AQR Multi-Style               |
| KMLM   | Factor     | Trend Following (Mount Lucas) |
| DBMF   | Factor     | Managed Futures (SG index)    |
| DBV    | Carry      | FX Carry G10                  |
| PDBC   | Carry      | Commodity Roll Yield          |

## Signal logic

| Signal  | Assets                          | Frequency | Method                        |
|---------|---------------------------------|-----------|-------------------------------|
| Trend   | SPY,EFA,EEM,QMOM,KMLM,DBMF     | Weekly    | 12-1 month momentum, z-score  |
| Carry   | DBV, PDBC, HYG                  | Monthly   | ETF vs benchmark return diff  |
| Regime  | All                             | Weekly    | VIX 20d smoothed threshold    |

## Updating AQR CMAs

Each January, AQR publishes new capital market assumptions. Update `EXPECTED_RETURNS`
in `jobs/weekly.py` with the new estimates. The optimizer will re-run with updated
return inputs on the next weekly job.
```

## Next steps (not yet built)

- [ ] Alpaca paper trading connection (`broker/alpaca.py`)
- [ ] Monthly job: carry + value spread signals
- [ ] Quarterly job: full re-optimisation with rolling covariance estimate
- [ ] Backtesting engine
- [ ] Email / Telegram alerts
