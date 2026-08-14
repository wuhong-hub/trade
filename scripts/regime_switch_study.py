"""三态 regime 切换策略实验：每日按沪深300 判定 bear/strong/weak，分状态切换策略参数。

状态判定（只用当天及以前的指数数据，无未来函数）：
- bear：close <= MA60 → 不开新仓
- strong：above60 且趋势强（6 种判定变体，见 STRONG_RULES）→ ma_trend_vol + atr=2.0 + hold=0
- weak：above60 但不满足 strong → momentum + stop=10% + hold=15

对照组（复用 portfolio_sim.simulate，与 variant_study 同口径）：
- momentum(stop=10%, hold=15) / ma_trend_vol(atr=2.0, hold=0) / momentum(stop=7%, hold=15)（基线）

评估区间：3 个半年区间（A/B/C）+ 2016-08-15→2026-08-13 按日历年分段。
输出：终端表格 + ~/.astock/regime_switch_study.csv。

用法：.venv/bin/python scripts/regime_switch_study.py
只读本地数据（~/.astock），无网络请求。
"""
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import portfolio_sim as ps  # noqa: E402

from astock.data import dataset, store  # noqa: E402
from astock.strategies import STRATEGIES_BY_NAME  # noqa: E402

INTERVALS = [
    ("A", "2025-01-01", "2025-07-01"),
    ("B", "2025-07-01", "2026-01-01"),
    ("C", "2026-01-01", None),
]

# 10 年逐年：2016-08-15 → 最新，按日历年分段
YEARLY = ([("2016H2", "2016-08-15", "2016-12-31")]
          + [(str(y), f"{y}-01-01", f"{y}-12-31") for y in range(2017, 2026)]
          + [("2026YTD", "2026-01-01", None)])

ALL_INTERVALS = INTERVALS + YEARLY

# 各状态使用的策略与退出参数（开仓时锁定，持有期内不随状态切换改变）
PARAMS_BY_STATE = {
    "strong": ("ma_trend_vol", dict(stop_pct=None, atr_mult=2.0, hold=0)),
    "weak": ("momentum", dict(stop_pct=0.10, atr_mult=None, hold=15)),
}

# strong 判定变体：在 above60 之上再满足对应条件
STRONG_RULES = [
    ("V1 slope20", lambda f: f["slope20"]),
    ("V2 slope10", lambda f: f["slope10"]),
    ("V3 slope60", lambda f: f["slope60"]),
    ("V4 ret20>3%", lambda f: f["ret20"] > 0.03),
    ("V5 ret20>5%", lambda f: f["ret20"] > 0.05),
    ("V6 slope20&ret20>0", lambda f: f["slope20"] & (f["ret20"] > 0)),
]

# 对照组：(标签, 策略, 参数)——与 variant_study 完全同口径
CONTROLS = [
    ("C1 momentum stop=10% hold=15", "momentum",
     dict(stop_pct=0.10, atr_mult=None, hold=15)),
    ("C2 ma_trend_vol atr=2.0 hold=0", "ma_trend_vol",
     dict(stop_pct=None, atr_mult=2.0, hold=0)),
    ("C3 momentum stop=7% hold=15", "momentum",
     dict(stop_pct=0.07, atr_mult=None, hold=15)),
]


def build_state_frame(index_df):
    """预计算状态判定所需特征。rolling/shift 均只用 ≤ 当日的数据。"""
    f = pd.DataFrame({"date": index_df["date"], "close": index_df["close"]})
    ma60 = f["close"].rolling(60).mean()
    f["above60"] = (f["close"] > ma60).fillna(False)
    for w in (10, 20, 60):
        f[f"slope{w}"] = (ma60 > ma60.shift(w)).fillna(False)
    f["ret20"] = f["close"] / f["close"].shift(20) - 1
    return f


def state_series(feat, strong_mask):
    """dict[date] -> 'bear'/'strong'/'weak'。strong_mask 为 bool Series（与 feat 对齐）。"""
    strong = (feat["above60"] & strong_mask.fillna(False))
    state = pd.Series("weak", index=feat.index)
    state[strong] = "strong"
    state[~feat["above60"]] = "bear"
    return dict(zip(feat["date"], state))


def simulate_switch(data_by_strat, days, states, params_by_state):
    """三态切换版组合模拟。每天开仓时按信号日（前一交易日）状态选策略与参数；
    持仓退出参数（止损价/持有上限）开仓时锁定，状态切换不强制平旧仓。

    data_by_strat: dict[策略名] -> prepare_data 结果；states: dict[date] -> 状态。
    返回格式与 portfolio_sim.simulate 一致。
    """
    capital = 100000
    cash = capital
    positions = {}   # code -> dict(..., hold, d=该票开仓策略的预计算数据)
    pending_sell = set()
    trades = []
    curve = []

    def mark(day):
        v = cash
        for code, p in positions.items():
            b = p["d"]["bars"]
            px = b["close"].loc[day] if day in b.index else p["entry_price"]
            v += p["shares"] * px
        return v

    for day in days:
        # 1) 执行昨日卖出信号（今日开盘卖出）
        for code in list(pending_sell):
            if code not in positions:
                continue
            b = positions[code]["d"]["bars"]
            if day in b.index:
                px = float(b["open"].loc[day])
                p = positions.pop(code)
                proceeds = p["shares"] * px * (1 - ps.SELL_FEE)
                cash += proceeds
                trades.append({k: v for k, v in p.items() if k != "d"} |
                              {"exit_date": day, "exit": px,
                               "ret": proceeds / p["cost"] - 1, "why": "卖出信号"})
            pending_sell.discard(code)
        # 2) 持仓的日内退出检查：止损 / 持有期满 / 卖出信号（按开仓时锁定的参数）
        for code, p in list(positions.items()):
            b = p["d"]["bars"]
            if day not in b.index:
                continue
            p["held"] += 1
            low = float(b["low"].loc[day])
            rec = {k: v for k, v in p.items() if k != "d"}
            if p["stop_px"] is not None and low <= p["stop_px"]:
                px = p["stop_px"]
                proceeds = p["shares"] * px * (1 - ps.SELL_FEE)
                cash += proceeds
                trades.append(rec | {"exit_date": day, "exit": px,
                                     "ret": proceeds / p["cost"] - 1, "why": "止损"})
                del positions[code]
            elif p["hold"] and p["held"] >= p["hold"]:
                px = float(b["close"].loc[day])
                proceeds = p["shares"] * px * (1 - ps.SELL_FEE)
                cash += proceeds
                trades.append(rec | {"exit_date": day, "exit": px,
                                     "ret": proceeds / p["cost"] - 1,
                                     "why": f"满{p['hold']}日"})
                del positions[code]
            elif p["d"]["sig"].get(day, 0) == -1:
                pending_sell.add(code)  # 明日开盘卖
        # 3) 开仓：昨日信号 + 昨日状态选策略，按量比排序，仓位 POS_PCT
        prev_days = [d for d in days if d < day]
        if prev_days and len(positions) < ps.MAX_POS:
            prev = prev_days[-1]
            state = states.get(prev, "bear")
            if state != "bear":
                strat_name, sp = params_by_state[state]
                data = data_by_strat[strat_name]
                cands = []
                for code, d in data.items():
                    if code in positions:
                        continue
                    if d["sig"].get(prev, 0) == 1 and prev in d["bars"].index \
                            and day in d["bars"].index:
                        vr = d["vol_ratio"].get(prev)
                        cands.append((vr if pd.notna(vr) else 0, code))
                cands.sort(reverse=True)
                equity = mark(day)
                for _, code in cands[:ps.MAX_POS - len(positions)]:
                    budget = min(equity * ps.POS_PCT, cash * 0.999)
                    if budget < 1000:
                        continue
                    px = float(data[code]["bars"]["open"].loc[day])
                    shares = budget / (px * (1 + ps.BUY_FEE))
                    cost = shares * px * (1 + ps.BUY_FEE)
                    if sp["stop_pct"] is not None:
                        stop_px = px * (1 - sp["stop_pct"])
                    else:
                        atr = data[code]["atr14"].get(day)
                        stop_px = (px - sp["atr_mult"] * atr
                                   if pd.notna(atr) else None)
                    cash -= cost
                    positions[code] = {"code": code, "entry_date": day,
                                       "entry_price": px, "shares": shares,
                                       "cost": cost, "held": 0,
                                       "stop_px": stop_px,
                                       "hold": sp["hold"], "state": state,
                                       "d": data[code]}
        curve.append({"date": day, "equity": mark(day)})

    eq = pd.DataFrame(curve).set_index("date")["equity"]
    final = eq.iloc[-1]
    return {
        "curve": eq,
        "trades": pd.DataFrame(trades),
        "positions": positions,
        "final": final,
        "ret": final / capital - 1,
        "maxdd": (eq / eq.cummax() - 1).min(),
    }


def run_one(simulate_fn, days):
    res = simulate_fn(days)
    tdf = res["trades"]
    return {
        "ret": res["ret"], "maxdd": res["maxdd"],
        "trades": len(tdf),
        "win_rate": (tdf["ret"] > 0).mean() if len(tdf) else float("nan"),
        "avg_ret": tdf["ret"].mean() if len(tdf) else float("nan"),
        "open_pos": len(res["positions"]),
    }


def main():
    home = Path(os.environ.get("ASTOCK_HOME", Path.home() / ".astock"))
    conn = store.connect(home / "astock.db")
    cons = store.load_constituents(conn)
    print("加载股票池…", flush=True)
    pool = dataset.build_pool_bars(conn, cons["code"].tolist())
    print(f"股票池 {len(pool)} 只，预计算信号（momentum / ma_trend_vol）…", flush=True)
    data_by_strat = {
        name: ps.prepare_data(pool, STRATEGIES_BY_NAME[name], need_atr=True)
        for name in ("momentum", "ma_trend_vol")
    }
    regime, index_df = ps.load_regime()   # 对照组用，与 variant_study 同口径
    feat = build_state_frame(store.load_index_daily(conn, "sh000300"))

    # 各变体的状态序列 + 10 年状态占比 + 最新交易日判定
    series_by_variant = {}
    print("\n=== 各状态时间占比（2016-08-15 → 最新）与最新交易日判定 ===", flush=True)
    for vlabel, rule in STRONG_RULES:
        states = state_series(feat, rule(feat))
        series_by_variant[vlabel] = states
        s = feat[(feat["date"] >= pd.Timestamp("2016-08-15"))]
        st = pd.Series([states[d] for d in s["date"]])
        share = st.value_counts(normalize=True)
        last = feat["date"].iloc[-1]
        print(f"{vlabel}: bear={share.get('bear', 0):.1%} "
              f"strong={share.get('strong', 0):.1%} weak={share.get('weak', 0):.1%}"
              f"  | 最新 {last.date()} 判定为 {states[last]}", flush=True)

    # 预生成各区间交易日与基准
    intervals = []
    for label, start, end in ALL_INTERVALS:
        days = ps.interval_days(index_df, start, end)
        real_end = days[-1]
        idx = index_df[(index_df["date"] >= pd.Timestamp(start))
                       & (index_df["date"] <= real_end)]
        bench = idx["close"].iloc[-1] / idx["close"].iloc[0] - 1
        intervals.append((label, str(pd.Timestamp(start).date()),
                          str(real_end.date()), days, bench))

    rows = []
    # 切换变体
    for vlabel in series_by_variant:
        states = series_by_variant[vlabel]
        for label, start, end_s, days, bench in intervals:
            row = run_one(lambda d: simulate_switch(
                data_by_strat, d, states, PARAMS_BY_STATE), days)
            rows.append({"group": "switch", "variant": vlabel, "interval": label,
                         "start": start, "end": end_s, "bench": bench, **row})
            print(f"[{label}] {vlabel}: ret={row['ret']:+.1%} dd={row['maxdd']:.1%} "
                  f"trades={row['trades']}", flush=True)
    # 对照组
    for clabel, strat_name, cparams in CONTROLS:
        for label, start, end_s, days, bench in intervals:
            row = run_one(lambda d: ps.simulate(
                data_by_strat[strat_name], d, regime,
                {"capital": 100000, **cparams}), days)
            rows.append({"group": "control", "variant": clabel, "interval": label,
                         "start": start, "end": end_s, "bench": bench, **row})
            print(f"[{label}] {clabel}: ret={row['ret']:+.1%} dd={row['maxdd']:.1%} "
                  f"trades={row['trades']}", flush=True)

    df = pd.DataFrame(rows)
    out_csv = home / "regime_switch_study.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nCSV 已存 {out_csv}")

    # markdown 汇总表
    print("\n| 组 | 变体 | 区间 | 收益率 | 最大回撤 | 笔数 | 胜率 | 平均单笔 | 期末持仓 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        wr = f"{r['win_rate']:.0%}" if pd.notna(r["win_rate"]) else "-"
        ar = f"{r['avg_ret']:+.2%}" if pd.notna(r["avg_ret"]) else "-"
        print(f"| {r['group']} | {r['variant']} | {r['interval']} | {r['ret']:+.1%} "
              f"| {r['maxdd']:.1%} | {r['trades']} | {wr} | {ar} | {r['open_pos']} |")


if __name__ == "__main__":
    main()
