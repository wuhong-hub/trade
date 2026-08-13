from dataclasses import dataclass

from ..data.dataset import build_pool_bars
from ..data.store import load_constituents
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


def position_pct(win_rate):
    return round(min(15.0, max(5.0, 10.0 + (win_rate - 0.5) * 20)), 1)


def generate_recommendations(conn, state, pool=None, top_n=10):
    cons = load_constituents(conn)
    names = dict(zip(cons["code"], cons["name"]))
    if pool is None:
        pool = build_pool_bars(conn, cons["code"].tolist())
    recs = {"short": [], "long": []}
    summary = {}
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
        for code, bars in pool.items():
            sigs = strat.signals(bars)
            if len(sigs) and sigs.iloc[-1] == 1:
                price = float(bars["close"].iloc[-1])
                stop = round(price * 0.93, 2) if horizon == "short" else None
                recs[horizon].append(Rec(
                    code, names.get(code, ""), horizon, best,
                    strat.reason(bars, bars["date"].iloc[-1]), price,
                    position_pct(last["win_rate"]), stop, strat.exit_hint()))
        recs[horizon] = recs[horizon][:top_n]
    return recs, summary
