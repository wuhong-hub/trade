from dataclasses import dataclass

import pandas as pd

from ..backtest.engine import SHORT_STOP
from ..data.dataset import build_pool_bars
from ..data.store import load_constituents, load_index_daily
from ..strategies import STRATEGIES_BY_NAME

DISCLAIMER = "仅为量化策略参考，不构成投资建议"


@dataclass
class Rec:
    code: str
    name: str
    horizon: str
    strategy: str
    reason: str
    price: float
    position_pct: float
    stop_price: float | None
    exit_hint: str


RISK_PCT = 0.5        # 单票风险敞口占净值比例（%），等额风险仓位
MAX_POSITION_PCT = 15.0


def atr14(bars):
    """最新 ATR14（TR 的 14 日均值）。数据不足返回 None。"""
    if len(bars) < 15:
        return None
    h, l, c = bars["high"], bars["low"], bars["close"]
    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().iloc[-1]
    return None if pd.isna(atr) or atr <= 0 else float(atr)


def position_pct(price, atr):
    """等额风险仓位：单票仓位% = RISK_PCT × 现价 ÷ ATR14，上限 MAX_POSITION_PCT。

    含义：若价格反向波动一个 ATR，该票亏损 ≈ 净值的 RISK_PCT%。
    波动越大的票仓位越小。ATR 缺失时按中性 10% 处理。
    """
    if atr is None:
        return 10.0
    return round(min(MAX_POSITION_PCT, RISK_PCT * price / atr), 1)


def index_regime(conn, ma=60):
    """沪深300 趋势过滤：最新收盘 > 最近 ma 日均线返回 True，否则 False。

    数据不足（行数 < ma+1）返回 None，表示不过滤。
    """
    df = load_index_daily(conn)
    if len(df) < ma + 1:
        return None
    return bool(df["close"].iloc[-1] > df["close"].tail(ma).mean())


def generate_recommendations(conn, state, pool=None, top_n=10):
    cons = load_constituents(conn)
    names = dict(zip(cons["code"], cons["name"]))
    if pool is None:
        pool = build_pool_bars(conn, cons["code"].tolist())
    recs = {"short": [], "long": []}
    summary = {}
    regime = index_regime(conn)  # 沪深300 趋势过滤，仅作用于短线
    for horizon in ("short", "long"):
        best = state["best"][horizon]
        if best is None:  # 该方向所有策略评分无效（全部窗口无交易）
            summary[horizon] = {"strategy": None}
            continue
        strat = STRATEGIES_BY_NAME[best]
        entry = next(e for e in state["ranking"][horizon]
                     if e["strategy"] == best)
        # 取最后一个有交易的窗口；与 rank.score 的口径一致（0 交易窗口无效）
        last = next((w for w in reversed(entry["windows"]) if w["n_trades"] > 0),
                    None)
        if last is None:
            # 全部窗口无交易：胜率按中性 0.5 处理，summary 标注 n_trades=0
            last = dict(entry["windows"][-1], win_rate=0.5, n_trades=0)
        summary[horizon] = {"strategy": best, "win_rate": last["win_rate"],
                            "annual_return": last["annual_return"],
                            "max_drawdown": last["max_drawdown"],
                            "n_trades": last["n_trades"]}
        if horizon == "short":
            summary[horizon]["regime"] = (
                "unknown" if regime is None else ("bull" if regime else "bear"))
            if regime is False:  # 指数位于均线下方：空仓观望，不扫描个股信号
                continue
        for code, bars in pool.items():
            sigs = strat.signals(bars)
            if len(sigs) and sigs.iloc[-1] == 1:
                price = float(bars["close"].iloc[-1])
                stop = round(price * SHORT_STOP, 2) if horizon == "short" else None
                recs[horizon].append(Rec(
                    code, names.get(code, ""), horizon, best,
                    strat.reason(bars, bars["date"].iloc[-1]), price,
                    position_pct(price, atr14(bars)), stop, strat.exit_hint()))
        recs[horizon] = recs[horizon][:top_n]
    return recs, summary
