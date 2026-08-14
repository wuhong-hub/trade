"""研究批次1·阶段2：执行侧三个候选的 10 年评估。

候选 4：移动止盈（基线 momentum stop=10% hold=15 regime 上加止损线上移）
  T1：浮盈 ≥8%（收盘口径）后，止损线上移至成本价（保本止损）
  T2：浮盈 ≥8% 后止损移至成本价；持仓期最高价 ≥ 成本+15% 后改跟踪止损（最高价回撤 8%）
  T3：全程跟踪止损：max(初始 -10%, 持仓期最高价回撤 10%)
  止损线每日收盘后用 ≤t 数据更新，次日起生效（无未来函数）；触发与成交规则同基线
  （日内 low ≤ 止损线 → 按止损价卖出）。

候选 5：ATR 波动率仓位（等额风险替代固定 10%）
  V1：单票股数 = 净值×1% ÷ ATR14；V2：净值×0.5% ÷ ATR14。
  ATR14 为截至前一日的 14 日均真实波幅（prepare_data 中 shift(1)，≤t 数据）。
  仓位上限 15%/票、最多 10 只、受现金约束；其余规则同基线。

候选 6：回测真实化（诚信检查）
  涨跌停：开盘价较昨收 ≥+9.8%（创业板 300/301、科创板 68 开头为 +19.8%）视为涨停，
  当日无法买入，信号顺延重试最多 2 日；开盘 ≤-9.8%/-19.8% 视为跌停无法卖出，顺延至可卖。
  止损触发日若开盘即跌停（锁死），顺延次日开盘卖。
  滑点：在现有费用（买 0.03%/卖 0.13%）外，对成交价双边不利方向加 0.1%/0.2%/0.3% 三档。

评估口径：3 个半年区间（A/B/C）+ 2016-08-15→最新按日历年 11 段。
对照基线（真实值）：A -14.4%、B +1.7%、C +14.0%。

用法：.venv/bin/python scripts/research_batch2.py [--candidate 4|5|6|all] [--check]
只读本地数据（~/.astock），无网络请求。结果 CSV 存 ~/.astock/research_batch2_*.csv。
"""
import argparse
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import portfolio_sim as ps  # noqa: E402

from astock.data import dataset, store  # noqa: E402
from astock.strategies import STRATEGIES_BY_NAME  # noqa: E402

BUY_FEE = ps.BUY_FEE
SELL_FEE = ps.SELL_FEE
MAX_POS = ps.MAX_POS

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

# 期望基线（WORKLOG 真实值，用于 --check 复核）
EXPECTED_BASELINE = {"A": -0.144, "B": 0.017, "C": 0.140}


def limit_pct(code):
    """涨跌停幅度阈值：创业板 300/301、科创板 68 为 20%，其余主板 10%（留 0.2% 余量）。"""
    return 0.198 if code.startswith(("300", "301", "68")) else 0.098


# ---------------------------------------------------------------- 扩展模拟器

def simulate2(data, days, regime, params):
    """ps.simulate 的扩展版：默认行为与之一致，新增可选执行侧规则。

    params 在 ps.simulate 基础上增加：
      trail:    None | "T1" | "T2" | "T3"  移动止盈规则（收盘后更新，次日生效）
      risk:     None | 每票风险敞口占净值比例（ATR14 等额风险仓位）
      pos_cap:  单票仓位占净值上限（固定仓位默认 0.10，ATR 模式默认 0.15）
      slippage: 成交价双边不利方向滑点（买 +s / 卖 -s）
      limits:   True 时启用涨跌停约束（涨停不可买顺延≤2日，跌停不可卖顺延）
    """
    capital = params.get("capital", 100000)
    stop_pct = params.get("stop_pct", 0.10)
    hold = params.get("hold", 15)
    no_regime = params.get("no_regime", False)
    trail = params.get("trail")
    risk = params.get("risk")
    pos_cap = params.get("pos_cap", 0.10 if risk is None else 0.15)
    slip = params.get("slippage", 0.0)
    limits = params.get("limits", False)
    max_defer = 2  # 涨停买入信号最多顺延 2 个交易日

    cash = capital
    positions = {}        # code -> dict(..., stop_px, hw=持仓期最高价)
    pending_sell = {}     # code -> 卖出原因（次日开盘卖；跌停顺延）
    deferred_buy = {}     # code -> 剩余顺延次数（涨停买入顺延）
    trades = []
    curve = []
    n_buy_defer = 0
    n_sell_defer = 0

    def mark(day):
        v = cash
        for code, p in positions.items():
            b = data[code]["bars"]
            px = b["close"].loc[day] if day in b.index else p["entry_price"]
            v += p["shares"] * px
        return v

    def record(p, day, px, why):
        trades.append({**p, "exit_date": day, "exit": px,
                       "ret": p["shares"] * px * (1 - SELL_FEE) / p["cost"] - 1,
                       "hw_ret": p["hw"] / p["entry_price"] - 1, "why": why})

    for day in days:
        # 1) 开盘执行挂卖（卖出信号/止损顺延），跌停则顺延
        for code, why in list(pending_sell.items()):
            if code not in positions:
                del pending_sell[code]
                continue
            b = data[code]["bars"]
            if day not in b.index:
                continue
            op = float(b["open"].loc[day])
            if limits:
                pc = data[code]["prev_close"].get(day)
                if pd.notna(pc) and op <= pc * (1 - limit_pct(code)):
                    n_sell_defer += 1
                    continue  # 开盘跌停，无法卖出，顺延
            p = positions.pop(code)
            px = op * (1 - slip)
            cash += p["shares"] * px * (1 - SELL_FEE)
            record(p, day, px, why)
            del pending_sell[code]
        # 2) 持仓日内退出检查：止损 / 持有期满 / 卖出信号
        for code, p in list(positions.items()):
            b = data[code]["bars"]
            if day not in b.index:
                continue
            p["held"] += 1
            low = float(b["low"].loc[day])
            high = float(b["high"].loc[day])
            close = float(b["close"].loc[day])
            if p["stop_px"] is not None and low <= p["stop_px"]:
                locked = False
                if limits:
                    pc = data[code]["prev_close"].get(day)
                    op = float(b["open"].loc[day])
                    # 开盘即跌停（锁死）：止损价无法成交，顺延次日开盘卖
                    locked = pd.notna(pc) and op <= pc * (1 - limit_pct(code))
                if locked:
                    pending_sell[code] = "止损(顺延)"
                    n_sell_defer += 1
                else:
                    px = p["stop_px"] * (1 - slip)
                    cash += p["shares"] * px * (1 - SELL_FEE)
                    record(p, day, px, "止损")
                    del positions[code]
            elif hold and p["held"] >= hold:
                px = close * (1 - slip)
                cash += p["shares"] * px * (1 - SELL_FEE)
                record(p, day, px, f"满{hold}日")
                del positions[code]
            elif data[code]["sig"].get(day, 0) == -1:
                pending_sell[code] = "卖出信号"
            # 收盘后更新持仓期最高价（所有变体都记录，供大单统计；只用 ≤t 数据）
            if code in positions and code not in pending_sell:
                p["hw"] = max(p["hw"], high)
            # 移动止盈：收盘后更新止损线（只用 ≤t 数据，次日起生效）
            if trail and code in positions and code not in pending_sell:
                entry = p["entry_price"]
                if trail in ("T1", "T2") and close >= entry * 1.08:
                    p["stop_px"] = max(p["stop_px"], entry)
                if trail == "T2" and p["hw"] >= entry * 1.15:
                    p["stop_px"] = max(p["stop_px"], p["hw"] * (1 - 0.08))
                if trail == "T3":
                    p["stop_px"] = max(p["stop_px"], p["hw"] * (1 - 0.10))
        # 3) 开仓：昨日信号 + regime，按量比排序；涨停顺延单优先
        prev_days = [d for d in days if d < day]
        if prev_days and len(positions) < MAX_POS:
            prev = prev_days[-1]
            cands = []
            if no_regime or regime.get(prev, False):
                for code, d in data.items():
                    if code in positions or code in deferred_buy:
                        continue
                    if d["sig"].get(prev, 0) == 1 and prev in d["bars"].index \
                            and day in d["bars"].index:
                        vr = d["vol_ratio"].get(prev)
                        cands.append((vr if pd.notna(vr) else 0, code))
                cands.sort(reverse=True)
            ordered = ([c for c in deferred_buy if c not in positions]
                       + [c for _, c in cands])
            equity = mark(day)
            for code in ordered[:MAX_POS - len(positions)]:
                d = data[code]
                if day not in d["bars"].index:
                    continue
                op = float(d["bars"]["open"].loc[day])
                if limits:
                    pc = d["prev_close"].get(day)
                    if pd.notna(pc) and op >= pc * (1 + limit_pct(code)):
                        n_buy_defer += 1
                        if code in deferred_buy:
                            deferred_buy[code] -= 1
                            if deferred_buy[code] <= 0:
                                del deferred_buy[code]  # 顺延次数用完，信号作废
                        else:
                            deferred_buy[code] = max_defer
                        continue
                px_eff = op * (1 + slip)
                if risk is not None:
                    atr = d["atr14"].get(day)  # 截至昨收的 ATR14，无未来数据
                    if not pd.notna(atr) or atr <= 0:
                        continue
                    shares = min(equity * risk / atr,
                                 min(equity * pos_cap, cash * 0.999) / px_eff)
                    cost = shares * px_eff * (1 + BUY_FEE)
                    if cost < 1000:
                        continue
                else:
                    budget = min(equity * pos_cap, cash * 0.999)
                    if budget < 1000:
                        continue
                    shares = budget / (px_eff * (1 + BUY_FEE))
                    cost = shares * px_eff * (1 + BUY_FEE)
                stop_px = op * (1 - stop_pct) if stop_pct is not None else None
                cash -= cost
                positions[code] = {"code": code, "entry_date": day,
                                   "entry_price": op, "shares": shares,
                                   "cost": cost, "held": 0, "stop_px": stop_px,
                                   "hw": op}
                deferred_buy.pop(code, None)
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
        "n_buy_defer": n_buy_defer,
        "n_sell_defer": n_sell_defer,
    }


# ---------------------------------------------------------------- 数据准备

def prepare2(pool, strat):
    """ps.prepare_data（need_atr=True）之外附加 prev_close（昨收，供涨跌停判定）。"""
    data = ps.prepare_data(pool, strat, need_atr=True)
    for code, d in data.items():
        d["prev_close"] = d["bars"]["close"].shift(1)
    return data


# ---------------------------------------------------------------- 评估骨架

def run_one(sim_fn, days):
    res = sim_fn(days)
    tdf = res["trades"]
    row = {
        "ret": res["ret"], "maxdd": res["maxdd"],
        "trades": len(tdf),
        "win_rate": (tdf["ret"] > 0).mean() if len(tdf) else float("nan"),
        "avg_ret": tdf["ret"].mean() if len(tdf) else float("nan"),
        "open_pos": len(res["positions"]),
        "n_stop": int((tdf["why"] == "止损").sum()) if len(tdf) else 0,
        "n_buy_defer": res["n_buy_defer"], "n_sell_defer": res["n_sell_defer"],
    }
    if len(tdf):
        big = tdf[tdf["hw_ret"] >= 0.15]   # 持仓期曾触及 +15% 的"大单"
        row["n_big"] = len(big)
        row["big_captured"] = int((big["ret"] >= 0.15).sum())
        row["big_avg_ret"] = big["ret"].mean() if len(big) else float("nan")
    else:
        row["n_big"] = 0
        row["big_captured"] = 0
        row["big_avg_ret"] = float("nan")
    return row


def build_intervals(index_df):
    intervals = []
    for label, start, end in ALL_INTERVALS:
        days = ps.interval_days(index_df, start, end)
        real_end = days[-1]
        idx = index_df[(index_df["date"] >= pd.Timestamp(start))
                       & (index_df["date"] <= real_end)]
        bench = idx["close"].iloc[-1] / idx["close"].iloc[0] - 1
        intervals.append((label, str(pd.Timestamp(start).date()),
                          str(real_end.date()), days, bench))
    return intervals


def eval_configs(configs, intervals):
    rows = []
    for group, variant, sim_fn in configs:
        for label, start, end_s, days, bench in intervals:
            row = run_one(sim_fn, days)
            rows.append({"group": group, "variant": variant, "interval": label,
                         "start": start, "end": end_s, "bench": bench, **row})
            print(f"[{label}] {variant}: ret={row['ret']:+.1%} "
                  f"dd={row['maxdd']:.1%} trades={row['trades']} "
                  f"stop={row['n_stop']} big={row['n_big']}"
                  f"(cap {row['big_captured']})", flush=True)
    return rows


def print_table(rows, title):
    print(f"\n=== {title} ===")
    print("| 组 | 变体 | 区间 | 收益率 | 最大回撤 | 笔数 | 胜率 | 止损数 | 大单数(捕获) | 大单均收 |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        wr = f"{r['win_rate']:.0%}" if pd.notna(r["win_rate"]) else "-"
        bar = f"{r['big_avg_ret']:+.1%}" if pd.notna(r["big_avg_ret"]) else "-"
        print(f"| {r['group']} | {r['variant']} | {r['interval']} | {r['ret']:+.1%} "
              f"| {r['maxdd']:.1%} | {r['trades']} | {wr} | {r['n_stop']} "
              f"| {r['n_big']}({r['big_captured']}) | {bar} |")


def save_csv(rows, home, name):
    out = home / name
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"CSV 已存 {out}", flush=True)


# ---------------------------------------------------------------- 变体定义

def make_configs(data, regime):
    """(candidate, group, variant, extra_params)。全部共用基线 stop=10%/hold=15/regime。"""
    defs = [
        ("4", "control", "基线 stop=10% hold=15 固定10%", {}),
        ("4", "trail", "T1 保本止损(≥8%)", {"trail": "T1"}),
        ("4", "trail", "T2 保本+跟踪(≥15%回撤8%)", {"trail": "T2"}),
        ("4", "trail", "T3 全程跟踪(回撤10%)", {"trail": "T3"}),
        ("5", "atr", "V1 等额风险1%/票", {"risk": 0.01}),
        ("5", "atr", "V2 等额风险0.5%/票", {"risk": 0.005}),
        ("6", "realism", "基线+涨跌停", {"limits": True}),
        ("6", "realism", "基线+滑点0.1%", {"slippage": 0.001}),
        ("6", "realism", "基线+滑点0.2%", {"slippage": 0.002}),
        ("6", "realism", "基线+滑点0.3%", {"slippage": 0.003}),
    ]
    configs = []
    for cand, group, variant, extra in defs:
        params = {"capital": 100000, "stop_pct": 0.10, "hold": 15, **extra}
        configs.append((cand, group, variant,
                        lambda d, pp=params: simulate2(data, d, regime, pp)))
    return configs


# ---------------------------------------------------------------- 主流程

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", default="all", choices=["4", "5", "6", "all"])
    ap.add_argument("--check", action="store_true",
                    help="只跑基线 A/B/C，复核 simulate2 与 ps.simulate 一致并对照真实基线")
    args = ap.parse_args()
    want = {"4", "5", "6"} if args.candidate == "all" else {args.candidate}

    home = Path(os.environ.get("ASTOCK_HOME", Path.home() / ".astock"))
    conn = store.connect(home / "astock.db")
    cons = store.load_constituents(conn)
    print("加载股票池…", flush=True)
    pool = dataset.build_pool_bars(conn, cons["code"].tolist())
    print(f"股票池 {len(pool)} 只", flush=True)

    print("预计算 momentum 信号（含 ATR14、昨收）…", flush=True)
    data = prepare2(pool, STRATEGIES_BY_NAME["momentum"])
    regime, index_df = ps.load_regime()
    intervals = build_intervals(index_df)

    if args.check:
        print("\n##### 基线复核（A/B/C）#####", flush=True)
        base_params = {"capital": 100000, "stop_pct": 0.10, "hold": 15}
        ok = True
        for label, start, end_s, days, bench in intervals[:3]:
            r_old = ps.simulate(data, days, regime, dict(base_params))
            r_new = simulate2(data, days, regime, dict(base_params))
            same = (abs(r_old["ret"] - r_new["ret"]) < 1e-9
                    and abs(r_old["maxdd"] - r_new["maxdd"]) < 1e-9
                    and len(r_old["trades"]) == len(r_new["trades"]))
            exp = EXPECTED_BASELINE[label]
            match = abs(r_new["ret"] - exp) < 0.002  # 0.2pt 容差（百分比显示口径）
            ok = ok and same and match
            print(f"[{label}] ps.simulate ret={r_old['ret']:+.2%} | "
                  f"simulate2 ret={r_new['ret']:+.2%} (一致={same}) | "
                  f"期望 {exp:+.1%} (符合={match})", flush=True)
        print("基线复核：" + ("通过" if ok else "未通过"), flush=True)
        return

    configs = make_configs(data, regime)
    for cand, csv_name, title in [
            ("4", "research_batch2_trailing.csv", "候选 4：移动止盈"),
            ("5", "research_batch2_atr.csv", "候选 5：ATR 波动率仓位"),
            ("6", "research_batch2_realism.csv", "候选 6：回测真实化")]:
        if cand not in want:
            continue
        print(f"\n##### {title} #####", flush=True)
        sel = [(g, v, fn) for c, g, v, fn in configs if c == cand or g == "control"]
        rows = eval_configs(sel, intervals)
        save_csv(rows, home, csv_name)
        print_table(rows, f"{title} 汇总")


if __name__ == "__main__":
    main()
