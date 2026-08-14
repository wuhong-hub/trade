import signal
import socket
import time
import urllib.request

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


def fetch_index_daily(symbol="sh000300"):
    """抓取指数日线（新浪接口），返回 date/close 两列的 DataFrame。"""
    df = _retry(ak.stock_zh_index_daily, symbol=symbol)
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "close"])
    out = pd.DataFrame({
        "date": pd.to_datetime(df["date"]),
        "close": pd.to_numeric(df["close"], errors="coerce"),
    })
    return out.dropna(subset=["date", "close"]).reset_index(drop=True)


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


SPOT_URL = "http://hq.sinajs.cn/list={symbols}"
SPOT_CHUNK = 50  # 新浪行情单次请求的股票数上限（经验值）


def _fetch_spot_raw(symbols):
    """新浪实时行情原始文本。注意必须用 http + Referer：本机 CA 证书过期
    https 校验失败，且该接口无 Referer 返回 403。行情为公开数据，无敏感信息。"""
    req = urllib.request.Request(
        SPOT_URL.format(symbols=",".join(symbols)),
        headers={"Referer": "https://finance.sina.com.cn"})
    return urllib.request.urlopen(req, timeout=20).read().decode("gbk")


def _parse_spot(text):
    """解析新浪行情文本。股票: 名称,今开,昨收,现价,最高,最低,买一,卖一,成交量,成交额,...,日期,时间
    指数(s_前缀): 名称,现价,涨跌,涨幅%,成交量(手),成交额(万)"""
    rows = []
    for line in text.strip().splitlines():
        if "=" not in line:
            continue
        var, _, payload = line.partition("=")
        symbol = var.replace("var hq_str_", "").strip()
        fields = payload.strip().strip(";").strip('"').split(",")
        if symbol.startswith("s_"):  # 指数简表
            if len(fields) >= 6 and fields[0]:
                rows.append({"code": symbol[2:], "name": fields[0],
                             "price": float(fields[1]),
                             "prev_close": float(fields[1]) - float(fields[2]),
                             "open": None, "high": None, "low": None,
                             "volume": float(fields[4]),
                             "amount": float(fields[5]) * 1e4,
                             "date": None, "time": None})
        else:
            if len(fields) >= 32 and fields[0]:
                rows.append({"code": symbol[2:], "name": fields[0],
                             "open": float(fields[1]),
                             "prev_close": float(fields[2]),
                             "price": float(fields[3]),
                             "high": float(fields[4]), "low": float(fields[5]),
                             "volume": float(fields[8]),
                             "amount": float(fields[9]),
                             "date": fields[30], "time": fields[31]})
    return rows


def fetch_spot_quotes(codes):
    """批量实时报价（交易时间内秒级刷新，收盘后为当日收盘价）。

    codes 为 6 位代码列表，返回 DataFrame：
    code/name/price/prev_close/open/high/low/volume/amount/date/time。
    """
    rows = []
    codes = [str(c).zfill(6) for c in codes]
    for i in range(0, len(codes), SPOT_CHUNK):
        symbols = [_sina_symbol(c) for c in codes[i:i + SPOT_CHUNK]]
        rows.extend(_parse_spot(_retry(_fetch_spot_raw, symbols=symbols)))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df[df["code"].isin(codes)].reset_index(drop=True)  # 过滤意外行


def fetch_index_spot(symbol="sh000300"):
    """指数实时点位（名称/现价/涨跌/涨幅/量额），返回单行 dict。"""
    rows = [r for r in _parse_spot(_retry(_fetch_spot_raw, symbols=["s_" + symbol]))
            if r["code"] == symbol]
    if not rows:
        raise RuntimeError(f"指数 {symbol} 实时行情解析为空")
    return rows[0]


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
