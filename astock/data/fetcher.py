import time

import akshare as ak
import pandas as pd

RETRY = 3


def _retry(fn, **kwargs):
    last = None
    for i in range(RETRY):
        try:
            return fn(**kwargs)
        except Exception as e:  # akshare 底层异常类型不稳定，统一捕获重试
            last = e
            time.sleep(1 + i)
    raise last


def fetch_index_constituents(index_code="000300"):
    df = _retry(ak.index_stock_cons_csindex, symbol=index_code)
    out = df[["成分券代码", "成分券名称"]].rename(
        columns={"成分券代码": "code", "成分券名称": "name"})
    out["code"] = out["code"].astype(str).str.zfill(6)
    return out.reset_index(drop=True)


def fetch_daily_bars(code, start_date, end_date=None):
    start = start_date.replace("-", "")
    end = (end_date or pd.Timestamp.today().strftime("%Y-%m-%d")).replace("-", "")
    df = _retry(ak.stock_zh_a_hist, symbol=code, period="daily",
                start_date=start, end_date=end, adjust="qfq")
    cols = ["date", "open", "high", "low", "close", "volume"]
    if df is None or df.empty:
        return pd.DataFrame(columns=cols)
    df = df.rename(columns={"日期": "date", "开盘": "open", "最高": "high",
                            "最低": "low", "收盘": "close", "成交量": "volume"})
    df["date"] = pd.to_datetime(df["date"])
    return df[cols]


def fetch_pe_series(code):
    df = _retry(ak.stock_a_indicator_lg, symbol=code)
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "pe"])
    df = df.rename(columns={"trade_date": "date"})
    df["date"] = pd.to_datetime(df["date"])
    return df[["date", "pe"]]


def fetch_financials(code, start_year="2021"):
    df = _retry(ak.stock_financial_analysis_indicator,
                symbol=code, start_year=start_year)
    cols = ["report_date", "roe", "revenue_growth"]
    if df is None or df.empty:
        return pd.DataFrame(columns=cols)
    out = pd.DataFrame({
        "report_date": pd.to_datetime(df["日期"]),
        "roe": pd.to_numeric(df["净资产收益率(%)"], errors="coerce"),
        "revenue_growth": pd.to_numeric(df["主营业务收入增长率(%)"], errors="coerce"),
    })
    return out.dropna(subset=["report_date"]).reset_index(drop=True)
