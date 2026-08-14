import signal
import socket
import time

import akshare as ak
import pandas as pd

RETRY = 3

# akshare 多数接口未显式传 timeout，本机网络中间设备会丢包导致 requests
# 无限期挂起（实测 2 小时无响应）。setdefaulttimeout 对 urllib3 不生效，
# 故保留之余，在 _retry 内用 SIGALRM 做每次调用的硬超时。
socket.setdefaulttimeout(20)

CALL_TIMEOUT = 30  # 单次接口调用的硬超时（秒）


class _CallTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _CallTimeout()


# 东财 stock_zh_a_hist 在本机网络下稳定被阻断；一旦重试耗尽失败过一次，
# 同一进程内后续 fetch_daily_bars 调用直接走新浪降级路径，避免每次白等 3 次重试。
_EM_HIST_DEAD = False


def _retry(fn, **kwargs):
    last = None
    for i in range(RETRY):
        try:
            signal.signal(signal.SIGALRM, _alarm_handler)
            signal.alarm(CALL_TIMEOUT)
            try:
                return fn(**kwargs)
            finally:
                signal.alarm(0)
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


def _sina_symbol(code):
    """新浪日线接口要求带市场前缀：6 开头 -> sh，其余 -> sz。"""
    return ("sh" if code.startswith("6") else "sz") + code


def fetch_daily_bars(code, start_date, end_date=None):
    """获取 A 股日 K（前复权），首选东财 ak.stock_zh_a_hist。

    若东财接口重试 3 次后仍抛异常（如本机网络下被中间设备阻断），
    自动降级到新浪 ak.stock_zh_a_daily（symbol 加 sh/sz 前缀）。
    两条路径均重试 3 次，输出列契约一致：date/open/high/low/close/volume。
    同一进程内东财一旦失败过一次（_EM_HIST_DEAD 置位），后续调用直接走新浪。
    """
    global _EM_HIST_DEAD
    start = start_date.replace("-", "")
    end = (end_date or pd.Timestamp.today().strftime("%Y-%m-%d")).replace("-", "")
    cols = ["date", "open", "high", "low", "close", "volume"]
    if not _EM_HIST_DEAD:
        try:
            df = _retry(ak.stock_zh_a_hist, symbol=code, period="daily",
                        start_date=start, end_date=end, adjust="qfq")
            if df is None or df.empty:
                return pd.DataFrame(columns=cols)
            df = df.rename(columns={"日期": "date", "开盘": "open", "最高": "high",
                                    "最低": "low", "收盘": "close", "成交量": "volume"})
            df["date"] = pd.to_datetime(df["date"])
            return df[cols]
        except Exception:
            _EM_HIST_DEAD = True
    df = _retry(ak.stock_zh_a_daily, symbol=_sina_symbol(code),
                start_date=start, end_date=end, adjust="qfq")
    if df is None or df.empty:
        return pd.DataFrame(columns=cols)
    df["date"] = pd.to_datetime(df["date"])
    return df[cols]


def fetch_pe_series(code):
    """获取个股 PE(TTM) 序列。

    akshare 1.18.84 已移除 ak.stock_a_indicator_lg，
    改用东财估值接口 ak.stock_value_em（symbol 为 6 位代码，无市场前缀）。
    """
    df = _retry(ak.stock_value_em, symbol=code)
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "pe"])
    return pd.DataFrame({
        "date": pd.to_datetime(df["数据日期"]),
        "pe": pd.to_numeric(df["PE(TTM)"], errors="coerce"),
    })


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
