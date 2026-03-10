# data/fetcher.py
import argparse
import logging
from datetime import datetime

import pandas as pd
import yfinance as yf

from data.store import Store
from config import TICKERS, DATA

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

BENCHMARK_TICKERS = ["SPY", "VTI"]  # always fetch these


def fetch_prices(store: Store, period: str = DATA["price_history"]) -> pd.DataFrame:
    symbols = list(TICKERS.keys()) + BENCHMARK_TICKERS
    symbols = list(dict.fromkeys(symbols))  # dedupe preserve order
    log.info(f"Fetching prices for {len(symbols)} symbols (period={period})")

    raw = yf.download(tickers=symbols, period=period, auto_adjust=True, progress=False)

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]].rename(columns={"Close": symbols[0]})

    prices = prices.dropna(how="all")
    log.info(f"Downloaded {len(prices)} rows, {prices.shape[1]} symbols")
    store.write_prices(prices)
    return prices


def fetch_macro(store: Store) -> dict:
    try:
        import os
        from fredapi import Fred

        api_key = os.environ.get("FRED_API_KEY")
        if not api_key:
            log.warning("FRED_API_KEY not set — skipping macro fetch.")
            return {}

        fred = Fred(api_key=api_key)
        results = {}

        series_map = {**DATA["fred_series"], "t2y": "DGS2", "t10y": "DGS10", "t30y": "DGS30", "usdcad": "DEXCAUS", "ca10y": "IRLTLT01CAM156N"}

        for name, series_id in series_map.items():
            try:
                s = fred.get_series(series_id, observation_start="2018-01-01")
                s.name = name
                store.write_macro(name, s)
                results[name] = s
                log.info(f"  FRED {series_id} ({name}): {len(s)} observations")
            except Exception as e:
                log.warning(f"  Could not fetch {series_id}: {e}")

        return results

    except ImportError:
        log.warning("fredapi not installed. Run: pip install fredapi")
        return {}


def fetch_all(period: str = DATA["price_history"]) -> None:
    store = Store(DATA["db_path"])
    fetch_prices(store, period)
    fetch_macro(store)
    log.info("Fetch complete.")


def fetch_news(tickers: list = None) -> dict:
    if tickers is None:
        from config import TICKERS
        tickers = list(TICKERS.keys())
    results = {}
    for ticker in tickers:
        try:
            t = yf.Ticker(ticker)
            news = t.news or []
            parsed = []
            for item in news[:5]:
                content = item.get("content", {})
                parsed.append({
                    "title":     content.get("title", item.get("title", "")),
                    "publisher": content.get("provider", {}).get("displayName", item.get("publisher", "")),
                    "link":      content.get("canonicalUrl", {}).get("url", item.get("link", "")),
                    "published": content.get("pubDate", item.get("providerPublishTime", "")),
                })
            results[ticker] = parsed
        except Exception:
            results[ticker] = []
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prices", action="store_true")
    parser.add_argument("--macro",  action="store_true")
    parser.add_argument("--period", default=DATA["price_history"])
    args = parser.parse_args()

    store = Store(DATA["db_path"])
    if args.prices or not args.macro:
        fetch_prices(store, args.period)
    if args.macro or not args.prices:
        fetch_macro(store)
    log.info("Done.")
