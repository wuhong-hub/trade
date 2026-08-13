import pandas as pd

from . import store

MIN_BARS = 30


def build_stock_bars(conn, code):
    bars = store.load_bars(conn, code).sort_values("date").reset_index(drop=True)
    pe = store.load_pe(conn, code)
    fin = store.load_financials(conn, code)
    if len(pe):
        bars = pd.merge_asof(bars, pe.sort_values("date"), on="date")
    else:
        bars["pe"] = float("nan")
    if len(fin):
        fin = fin.sort_values("report_date").rename(columns={"report_date": "date"})
        bars = pd.merge_asof(bars, fin, on="date")
    else:
        bars["roe"] = float("nan")
        bars["revenue_growth"] = float("nan")
    return bars


def build_pool_bars(conn, codes):
    pool = {}
    for code in codes:
        bars = build_stock_bars(conn, code)
        if len(bars) >= MIN_BARS:
            pool[code] = bars
    return pool
