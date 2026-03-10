# data/store.py
# Thin wrapper around SQLite for price and macro storage.
# Uses SQLAlchemy so you can swap to Postgres later by changing one URL.

import os
import logging
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

log = logging.getLogger(__name__)


class Store:
    def __init__(self, db_path: str):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(f"sqlite:///{db_path}", echo=False)
        self._init_schema()

    def _init_schema(self):
        with self.engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS prices (
                    date   TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    close  REAL NOT NULL,
                    PRIMARY KEY (date, ticker)
                )
            """))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS macro (
                    date   TEXT NOT NULL,
                    series TEXT NOT NULL,
                    value  REAL NOT NULL,
                    PRIMARY KEY (date, series)
                )
            """))
            conn.commit()

    # ── Prices ────────────────────────────────────────────────────────────────

    def write_prices(self, df: pd.DataFrame) -> None:
        """
        Write wide price DataFrame (date x ticker) to DB.
        Upserts — safe to run repeatedly.
        """
        long = (
            df.reset_index()
            .melt(id_vars="Date", var_name="ticker", value_name="close")
            .dropna(subset=["close"])
            .rename(columns={"Date": "date"})
        )
        long["date"] = pd.to_datetime(long["date"]).dt.strftime("%Y-%m-%d")

        with self.engine.connect() as conn:
            for _, row in long.iterrows():
                conn.execute(text("""
                    INSERT OR REPLACE INTO prices (date, ticker, close)
                    VALUES (:date, :ticker, :close)
                """), {"date": row["date"], "ticker": row["ticker"], "close": row["close"]})
            conn.commit()

        log.info(f"Wrote {len(long)} price rows to DB")

    def read_prices(self, tickers: list[str] | None = None, start: str | None = None) -> pd.DataFrame:
        """
        Returns wide DataFrame: date (index) x ticker (columns).
        """
        query = "SELECT date, ticker, close FROM prices"
        conditions = []
        if tickers:
            placeholders = ",".join(f"'{t}'" for t in tickers)
            conditions.append(f"ticker IN ({placeholders})")
        if start:
            conditions.append(f"date >= '{start}'")
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY date"

        df = pd.read_sql(query, self.engine, parse_dates=["date"])
        if df.empty:
            return pd.DataFrame()

        wide = df.pivot(index="date", columns="ticker", values="close")
        wide.index = pd.to_datetime(wide.index)
        return wide

    def latest_prices(self) -> pd.Series:
        """Returns the most recent close for each ticker."""
        query = """
            SELECT ticker, close
            FROM prices
            WHERE date = (SELECT MAX(date) FROM prices WHERE ticker = prices.ticker)
        """
        df = pd.read_sql(query, self.engine)
        return df.set_index("ticker")["close"]

    # ── Macro ─────────────────────────────────────────────────────────────────

    def write_macro(self, series_name: str, s: pd.Series) -> None:
        """Write a named macro series (date -> value)."""
        records = [
            {"date": d.strftime("%Y-%m-%d"), "series": series_name, "value": float(v)}
            for d, v in s.dropna().items()
        ]
        with self.engine.connect() as conn:
            for r in records:
                conn.execute(text("""
                    INSERT OR REPLACE INTO macro (date, series, value)
                    VALUES (:date, :series, :value)
                """), r)
            conn.commit()
        log.info(f"Wrote {len(records)} rows for macro series '{series_name}'")

    def read_macro(self, series_name: str, start: str | None = None) -> pd.Series:
        """Read a macro series by name. Returns pd.Series indexed by date."""
        query = f"SELECT date, value FROM macro WHERE series = '{series_name}'"
        if start:
            query += f" AND date >= '{start}'"
        query += " ORDER BY date"

        df = pd.read_sql(query, self.engine, parse_dates=["date"])
        if df.empty:
            return pd.Series(dtype=float, name=series_name)

        s = df.set_index("date")["value"]
        s.name = series_name
        return s

    def available_macro(self) -> list[str]:
        with self.engine.connect() as conn:
            result = conn.execute(text("SELECT DISTINCT series FROM macro"))
            return [r[0] for r in result]
