import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from ..data.dataset import build_pool_bars
from ..data.store import load_constituents
from ..strategies import ALL_STRATEGIES
from .engine import run_backtest


def rolling_windows(end=None, years=2, n=4):
    end = pd.Timestamp(end) if end is not None else pd.Timestamp.today().normalize()
    start = end - pd.DateOffset(years=years)
    bounds = pd.date_range(start, end, periods=n + 1)
    return [(bounds[i], bounds[i + 1]) for i in range(n)]


def score(results):
    valid = [r for r in results if r.n_trades > 0]
    if not valid:
        return float("-inf")
    return sum(r.sharpe for r in valid) / len(valid)


def run_iteration(conn, state_path, history_path):
    cons = load_constituents(conn)
    pool = build_pool_bars(conn, cons["code"].tolist())
    windows = rolling_windows()
    ranking = {"short": [], "long": []}
    for s in ALL_STRATEGIES:
        results = [run_backtest(s, pool, w[0], w[1]) for w in windows]
        ranking[s.meta.horizon].append({
            "strategy": s.meta.name,
            "score": score(results),
            "windows": [{
                "window": r.window, "n_trades": r.n_trades,
                "win_rate": r.win_rate, "annual_return": r.annual_return,
                "max_drawdown": r.max_drawdown, "sharpe": r.sharpe,
            } for r in results],
        })
    for h in ranking:
        ranking[h].sort(key=lambda e: e["score"], reverse=True)
    state = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        # 所有策略 score 均为 -inf（全部窗口无交易）时，该方向无有效策略
        "best": {h: (ranking[h][0]["strategy"]
                     if ranking[h] and ranking[h][0]["score"] > float("-inf")
                     else None)
                 for h in ranking},
        "ranking": ranking,
    }
    state_path = Path(state_path)
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2))
    with open(history_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(state, ensure_ascii=False) + "\n")
    return state
