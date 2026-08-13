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
        strat = STRATEGIES_BY_NAME[best]
        entry = next(e for e in state["ranking"][horizon]
                     if e["strategy"] == best)
        last = entry["windows"][-1]
        summary[horizon] = {"strategy": best, "win_rate": last["win_rate"],
                            "annual_return": last["annual_return"],
                            "max_drawdown": last["max_drawdown"]}
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
