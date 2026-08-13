import pandas as pd

from . import store

MIN_BARS = 30
# 财报按报告期 + 90 天近似公告日合并，避免回测在公告前"看到"财报（前视偏差）。
# 这是粗略近似：实际公告滞后随报告期不同（季报 1 个月、年报最长 4 个月）。
ANNOUNCEMENT_LAG_DAYS = 90


def build_stock_bars(conn, code):
    """合并日线、PE 与财务指标。财务数据按 report_date + ANNOUNCEMENT_LAG_DAYS
    的近似的公告日 asof 前向填充，公告日之前不可见（NaN）。"""
    bars = store.load_bars(conn, code).sort_values("date").reset_index(drop=True)
    pe = store.load_pe(conn, code)
    fin = store.load_financials(conn, code)
    if len(pe):
        bars = pd.merge_asof(bars, pe.sort_values("date"), on="date")
    else:
        bars["pe"] = float("nan")
    if len(fin):
        fin = fin.sort_values("report_date").rename(columns={"report_date": "date"})
        fin["date"] = fin["date"] + pd.Timedelta(days=ANNOUNCEMENT_LAG_DAYS)
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
