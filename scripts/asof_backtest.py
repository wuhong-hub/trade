"""时点推荐回测：回答"如果在过去某个交易日运行 astock，它会推荐什么？结果如何？"

用法：.venv/bin/python scripts/asof_backtest.py 2026-07-13 [更多日期...]
完全离线，只读 ~/.astock/astock.db。
"""
import os
import sys
from pathlib import Path

import pandas as pd

from astock.backtest import engine, rank
from astock.data import dataset, store
from astock.strategies import ALL_STRATEGIES

COST = engine.COST
TOP_N = 10


def best_strategies_asof(pool, as_of):
    """用截至 as_of 的数据做滚动回测，返回 {horizon: (strategy, score)}。"""
    windows = rank.rolling_windows(end=as_of)
    best = {}
    for s in ALL_STRATEGIES:
        results = [engine.run_backtest(s, pool, w[0], w[1]) for w in windows]
        sc = rank.score(results)
        h = s.meta.horizon
        if h not in best or sc > best[h][1]:
            best[h] = (s, sc)
    return best


def recs_asof(strat, pool, as_of, top_n=TOP_N):
    """as_of 当日信号为买入的股票（与 recommend 同一判定）。"""
    out = []
    for code, bars in pool.items():
        b = bars[bars["date"] <= as_of]
        if len(b) < 30:
            continue
        sigs = strat.signals(b.reset_index(drop=True))
        if len(sigs) and sigs.iloc[-1] == 1:
            out.append((code, float(b["close"].iloc[-1])))
    return out[:top_n]


def eval_short(bars, as_of):
    """按短线规则模拟：信号次日开盘买入，-7% 止损 / 卖出信号 / 满 10 个交易日退出。"""
    hist = bars[bars["date"] <= as_of]
    fwd = bars[bars["date"] > as_of].head(12).reset_index(drop=True)
    if len(fwd) < 2:
        return None
    entry = float(fwd["open"].iloc[0])
    entry_date = fwd["date"].iloc[0]
    stop = entry * 0.93
    exit_price, exit_date, why = float(fwd["close"].iloc[-1]), fwd["date"].iloc[-1], "数据末"
    for i in range(len(fwd)):
        row = fwd.iloc[i]
        if row["low"] <= stop:
            exit_price, exit_date, why = stop, row["date"], "止损"
            break
        if i >= 10:
            exit_price, exit_date, why = float(row["close"]), row["date"], "满10日"
            break
    return {"entry_date": entry_date, "entry": entry, "exit_date": exit_date,
            "exit": exit_price, "why": why, "ret": exit_price / entry - 1 - COST}


def eval_long(bars, as_of, days=20):
    """长线简化评估：信号次日开盘买入，持有 days 个交易日收盘卖出。"""
    fwd = bars[bars["date"] > as_of].head(days + 1).reset_index(drop=True)
    if len(fwd) < 2:
        return None
    entry = float(fwd["open"].iloc[0])
    exit_price = float(fwd["close"].iloc[min(days, len(fwd) - 1)])
    return {"entry_date": fwd["date"].iloc[0], "entry": entry,
            "exit_date": fwd["date"].iloc[min(days, len(fwd) - 1)],
            "exit": exit_price, "why": f"持有{days}日",
            "ret": exit_price / entry - 1 - COST}


def run(as_of, pool, names):
    as_of = pd.Timestamp(as_of)
    print(f"\n{'='*70}\n时点：{as_of.date()}（只用该日之前的数据）")
    best = best_strategies_asof(pool, as_of)
    for h, label in (("short", "短线"), ("long", "长线")):
        strat, sc = best[h]
        recs = recs_asof(strat, pool, as_of)
        print(f"\n[{label}] 当时最优策略：{strat.meta.name}（综合分 {sc:.2f}），"
              f"当日触发 {len(recs)} 只（取前 {TOP_N}）")
        rows = []
        for code, price in recs:
            bars = pool[code]
            t = eval_short(bars, as_of) if h == "short" else eval_long(bars, as_of)
            if t is None:
                continue
            rows.append({"code": code, "name": names.get(code, ""), **t})
            print(f"  {code} {names.get(code, ''):<6} "
                  f"{t['entry_date'].date()} 买 {t['entry']:.2f} → "
                  f"{t['exit_date'].date()} 卖 {t['exit']:.2f}（{t['why']}） "
                  f"净收益 {t['ret']:+.1%}")
        if rows:
            df = pd.DataFrame(rows)
            hit = (df["ret"] > 0).mean()
            print(f"  → 命中率 {hit:.0%}（{(df['ret'] > 0).sum()}/{len(df)}），"
                  f"平均净收益 {df['ret'].mean():+.1%}，"
                  f"最好 {df['ret'].max():+.1%}，最差 {df['ret'].min():+.1%}")
            # 基准：同期全池等权平均收益作对照
            bench = []
            for code in pool:
                t = eval_short(pool[code], as_of) if h == "short" else eval_long(pool[code], as_of)
                if t:
                    bench.append(t["ret"])
            if bench:
                print(f"  → 同期全池等权基准：{pd.Series(bench).mean():+.1%}")
        else:
            print("  当时无推荐。")


def main():
    dates = sys.argv[1:] or ["2026-07-13"]
    home = Path(os.environ.get("ASTOCK_HOME", Path.home() / ".astock"))
    conn = store.connect(home / "astock.db")
    cons = store.load_constituents(conn)
    names = dict(zip(cons["code"], cons["name"]))
    print("加载股票池…")
    pool = dataset.build_pool_bars(conn, cons["code"].tolist())
    print(f"池内 {len(pool)} 只，开始逐时点回测…")
    for d in dates:
        run(d, pool, names)


if __name__ == "__main__":
    main()
