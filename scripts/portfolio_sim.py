"""组合级模拟：从某日起，用固定本金严格按策略信号操作，输出收益曲线与总结。

默认变体：momentum + 持有15日 + -7%止损 + regime过滤（沪深300>MA60才开仓），
仓位规则与 astock 一致：单票为当前净值的一定比例，最多 10 只，信号按量比排序取舍。

用法：.venv/bin/python scripts/portfolio_sim.py [--start 2026-01-01] [--end 2026-06-30] [--capital 100000]
只读本地数据（~/.astock），无网络请求。
"""
import argparse
import os
from pathlib import Path

import pandas as pd

from astock.data import dataset, store
from astock.strategies import STRATEGIES_BY_NAME

BUY_FEE = 0.0003          # 买入佣金
SELL_FEE = 0.0013         # 卖出佣金+印花税
STOP = 0.93               # -7% 止损
HOLD = 15                 # 持有上限（交易日）
MAX_POS = 10              # 最多同时持仓数
POS_PCT = 0.10            # 单票仓位占净值比例


def load_regime(ma=60):
    df = pd.read_csv("/home/wuhong/.astock/index_000300.csv", parse_dates=["date"])
    above = df["close"] > df["close"].rolling(ma).mean()
    regime = dict(zip(df["date"], above.fillna(False)))
    return regime, df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--capital", type=float, default=100000)
    ap.add_argument("--strategy", default="momentum")
    ap.add_argument("--no-regime", action="store_true", help="关闭指数 regime 过滤")
    args = ap.parse_args()
    start = pd.Timestamp(args.start)

    home = Path(os.environ.get("ASTOCK_HOME", Path.home() / ".astock"))
    conn = store.connect(home / "astock.db")
    cons = store.load_constituents(conn)
    names = dict(zip(cons["code"], cons["name"]))
    print("加载股票池并预计算信号…")
    pool = dataset.build_pool_bars(conn, cons["code"].tolist())
    strat = STRATEGIES_BY_NAME[args.strategy]

    regime, index_df = load_regime()
    if args.end:
        days = index_df[(index_df["date"] >= start)
                        & (index_df["date"] <= pd.Timestamp(args.end))]["date"].tolist()
    else:
        days = index_df[(index_df["date"] >= start)]["date"].tolist()
    end = days[-1]

    # 预计算：每股信号、量比、按日期索引的 bar
    data = {}
    for code, bars in pool.items():
        b = bars.set_index("date")
        vol_ma = b["volume"].rolling(20).mean()
        data[code] = {
            "bars": b,
            "sig": strat.signals(bars),
            "vol_ratio": b["volume"] / vol_ma,
        }

    cash = args.capital
    positions = {}   # code -> dict(entry_date, entry_price, shares, cost, held)
    pending_sell = set()
    trades = []
    curve = []

    def mark(day):
        v = cash
        for code, p in positions.items():
            b = data[code]["bars"]
            px = b["close"].loc[day] if day in b.index else p["entry_price"]
            v += p["shares"] * px
        return v

    for day in days:
        # 1) 执行昨日卖出信号（今日开盘卖出）
        for code in list(pending_sell):
            if code not in positions:
                continue
            b = data[code]["bars"]
            if day in b.index:
                px = float(b["open"].loc[day])
                p = positions.pop(code)
                proceeds = p["shares"] * px * (1 - SELL_FEE)
                cash += proceeds
                trades.append({**p, "exit_date": day, "exit": px,
                               "ret": proceeds / p["cost"] - 1, "why": "卖出信号"})
            pending_sell.discard(code)
        # 2) 持仓的日内退出检查：止损 / 持有期满
        for code, p in list(positions.items()):
            b = data[code]["bars"]
            if day not in b.index:
                continue
            p["held"] += 1
            low = float(b["low"].loc[day])
            if low <= p["entry_price"] * STOP:
                px = p["entry_price"] * STOP
                proceeds = p["shares"] * px * (1 - SELL_FEE)
                cash += proceeds
                trades.append({**p, "exit_date": day, "exit": px,
                               "ret": proceeds / p["cost"] - 1, "why": "止损"})
                del positions[code]
            elif p["held"] >= HOLD:
                px = float(b["close"].loc[day])
                proceeds = p["shares"] * px * (1 - SELL_FEE)
                cash += proceeds
                trades.append({**p, "exit_date": day, "exit": px,
                               "ret": proceeds / p["cost"] - 1, "why": "满15日"})
                del positions[code]
            elif data[code]["sig"].get(day, 0) == -1:
                pending_sell.add(code)  # 明日开盘卖
        # 3) 开仓：昨日信号 + regime，按量比排序，仓位 POS_PCT
        prev_days = [d for d in days if d < day]
        if prev_days and len(positions) < MAX_POS:
            prev = prev_days[-1]
            if args.no_regime or regime.get(prev, False):
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
                for _, code in cands[:MAX_POS - len(positions)]:
                    budget = min(equity * POS_PCT, cash * 0.999)
                    if budget < 1000:
                        continue
                    px = float(data[code]["bars"]["open"].loc[day])
                    shares = budget / (px * (1 + BUY_FEE))
                    cost = shares * px * (1 + BUY_FEE)
                    cash -= cost
                    positions[code] = {"code": code, "entry_date": day,
                                       "entry_price": px, "shares": shares,
                                       "cost": cost, "held": 0}
        curve.append({"date": day, "equity": mark(day)})

    # 收尾：按最后收盘价强平估值（不实际卖出，计入净值）
    eq = pd.DataFrame(curve).set_index("date")["equity"]
    tdf = pd.DataFrame(trades)
    final = eq.iloc[-1]
    ret = final / args.capital - 1
    dd = (eq / eq.cummax() - 1).min()
    idx = index_df[(index_df["date"] >= start) & (index_df["date"] <= end)]
    bench = idx["close"].iloc[-1] / idx["close"].iloc[0] - 1

    print(f"\n{'='*66}")
    print(f"区间 {start.date()} → {end.date()}  本金 {args.capital:,.0f}")
    print(f"策略 {args.strategy}（持{HOLD}日/止损-7%/{'regime过滤' if not args.no_regime else '无过滤'}）")
    print(f"最终净值 {final:,.0f}  收益率 {ret:+.1%}  最大回撤 {dd:.1%}  沪深300同期 {bench:+.1%}")
    if len(tdf):
        win = (tdf["ret"] > 0).mean()
        print(f"已完成交易 {len(tdf)} 笔  胜率 {win:.0%}  平均单笔 {tdf['ret'].mean():+.2%}  "
              f"最好 {tdf['ret'].max():+.1%}  最差 {tdf['ret'].min():+.1%}")
        by_why = tdf.groupby("why")["ret"].agg(["count", "mean"])
        print(by_why.to_string())
    print(f"期末持仓 {len(positions)} 只：" +
          "、".join(f"{c}{names.get(c, '')}" for c in positions) if positions else "期末空仓")
    eq.to_csv("/home/wuhong/.astock/portfolio_curve.csv")


if __name__ == "__main__":
    main()
