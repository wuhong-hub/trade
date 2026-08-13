import sqlite3
from pathlib import Path

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS daily_bars(
    code TEXT NOT NULL, date TEXT NOT NULL,
    open REAL, high REAL, low REAL, close REAL, volume REAL,
    PRIMARY KEY(code, date));
CREATE TABLE IF NOT EXISTS pe_series(
    code TEXT NOT NULL, date TEXT NOT NULL, pe REAL,
    PRIMARY KEY(code, date));
CREATE TABLE IF NOT EXISTS financials(
    code TEXT NOT NULL, report_date TEXT NOT NULL,
    roe REAL, revenue_growth REAL,
    PRIMARY KEY(code, report_date));
CREATE TABLE IF NOT EXISTS constituents(
    code TEXT PRIMARY KEY, name TEXT, updated_at TEXT);
CREATE TABLE IF NOT EXISTS recommendations(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    rec_date TEXT, code TEXT, name TEXT, horizon TEXT, strategy TEXT,
    reason TEXT, entry_price REAL, position_pct REAL,
    stop_price REAL, exit_hint TEXT);
"""


def connect(db_path):
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    return conn


def upsert_bars(conn, code, df):
    rows = [
        (code, d.strftime("%Y-%m-%d"), float(r.open), float(r.high),
         float(r.low), float(r.close), float(r.volume))
        for d, r in zip(df["date"], df.itertuples())
    ]
    conn.executemany("INSERT OR REPLACE INTO daily_bars VALUES(?,?,?,?,?,?,?)", rows)
    conn.commit()


def load_bars(conn, code):
    df = pd.read_sql(
        "SELECT date, open, high, low, close, volume FROM daily_bars "
        "WHERE code=? ORDER BY date", conn, params=(code,))
    df["date"] = pd.to_datetime(df["date"])
    return df


def last_bar_date(conn, code):
    row = conn.execute(
        "SELECT MAX(date) FROM daily_bars WHERE code=?", (code,)).fetchone()
    return row[0]


def upsert_pe(conn, code, df):
    rows = [(code, d.strftime("%Y-%m-%d"), float(r.pe))
            for d, r in zip(df["date"], df.itertuples())]
    conn.executemany("INSERT OR REPLACE INTO pe_series VALUES(?,?,?)", rows)
    conn.commit()


def load_pe(conn, code):
    df = pd.read_sql(
        "SELECT date, pe FROM pe_series WHERE code=? ORDER BY date",
        conn, params=(code,))
    df["date"] = pd.to_datetime(df["date"])
    return df


def upsert_financials(conn, code, df):
    rows = [(code, d.strftime("%Y-%m-%d"), float(r.roe), float(r.revenue_growth))
            for d, r in zip(df["report_date"], df.itertuples())]
    conn.executemany("INSERT OR REPLACE INTO financials VALUES(?,?,?,?)", rows)
    conn.commit()


def load_financials(conn, code):
    df = pd.read_sql(
        "SELECT report_date, roe, revenue_growth FROM financials "
        "WHERE code=? ORDER BY report_date", conn, params=(code,))
    df["report_date"] = pd.to_datetime(df["report_date"])
    return df


def save_constituents(conn, df):
    conn.execute("DELETE FROM constituents")
    conn.executemany(
        "INSERT INTO constituents(code, name, updated_at) VALUES(?,?,date('now'))",
        [(str(r.code), str(r.name)) for r in df.itertuples()])
    conn.commit()


def load_constituents(conn):
    return pd.read_sql("SELECT code, name FROM constituents ORDER BY code", conn)
