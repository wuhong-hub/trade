"""组合级模拟：从某日起，用固定本金严格按策略信号操作，输出收益曲线与总结。

默认变体：momentum + 持有15日 + -7%止损 + regime过滤（沪深300>MA60才开仓），
仓位规则与 astock 一致：单票为当前净值的一定比例，最多 10 只，信号按量比排序取舍。

用法：.venv/bin/python scripts/portfolio_sim.py [--start 2026-01-01] [--end 2026-06-30] [--capital 100000]
      [--stop 0.07 | --atr-stop 2.0] [--hold 15]   # --hold 0 表示不限持有期
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
DEFAULT_STOP = 0.10       # 默认 -10% 止损（与主程序 SHORT_STOP 一致）
DEFAULT_HOLD = 15         # 默认持有上限（交易日）
MAX_POS = 10              # 最多同时持仓数
POS_PCT = 0.10            # 单票仓位占净值比例


def load_regime(ma=60):
    df = pd.read_csv("/home/wuhong/.astock/index_000300.csv", parse_dates=["date"])
    above = df["close"] > df["close"].rolling(ma).mean()
    regime = dict(zip(df["date"], above.fillna(False)))
    return regime, df


def prepare_data(pool, strat, need_atr=False):
    """预计算：每股信号、量比、按日期索引的 bar；need_atr 时附 ATR14（截至前一日的14日均值，不用未来数据）。"""
    data = {}
    for code, bars in pool.items():
        b = bars.set_index("date")
        vol_ma = b["volume"].rolling(20).mean()
        d = {
            "bars": b,
            "sig": strat.signals(bars),
            "vol_ratio": b["volume"] / vol_ma,
        }
        if need_atr:
            prev_close = b["close"].shift(1)
            tr = pd.concat([
                b["high"] - b["low"],
                (b["high"] - prev_close).abs(),
                (b["low"] - prev_close).abs(),
            ], axis=1).max(axis=1)
            d["atr14"] = tr.rolling(14).mean().shift(1)
        data[code] = d
    return data


def simulate(data, days, regime, params):
    """组合模拟主体。params: capital, stop_pct|None, atr_mult|None, hold(0=不限), no_regime。

    返回 dict(curve, trades, positions, final, ret, maxdd)。
    """
    capital = params.get("capital", 100000)
    stop_pct = params.get("stop_pct")
    atr_mult = params.get("atr_mult")
    hold = params.get("hold", DEFAULT_HOLD)
    no_regime = params.get("no_regime", False)

    cash = capital
    positions = {}   # code -> dict(entry_date, entry_price, shares, cost, held, stop_px)
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
        # 2) 持仓的日内退出检查：止损 / 持有期满 / 卖出信号
        for code, p in list(positions.items()):
            b = data[code]["bars"]
            if day not in b.index:
                continue
            p["held"] += 1
            low = float(b["low"].loc[day])
            if p["stop_px"] is not None and low <= p["stop_px"]:
                px = p["stop_px"]
                proceeds = p["shares"] * px * (1 - SELL_FEE)
                cash += proceeds
                trades.append({**p, "exit_date": day, "exit": px,
                               "ret": proceeds / p["cost"] - 1, "why": "止损"})
                del positions[code]
            elif hold and p["held"] >= hold:
                px = float(b["close"].loc[day])
                proceeds = p["shares"] * px * (1 - SELL_FEE)
                cash += proceeds
                trades.append({**p, "exit_date": day, "exit": px,
                               "ret": proceeds / p["cost"] - 1, "why": f"满{hold}日"})
                del positions[code]
            elif data[code]["sig"].get(day, 0) == -1:
                pending_sell.add(code)  # 明日开盘卖
        # 3) 开仓：昨日信号 + regime，按量比排序，仓位 POS_PCT
        prev_days = [d for d in days if d < day]
        if prev_days and len(positions) < MAX_POS:
            prev = prev_days[-1]
            if no_regime or regime.get(prev, False):
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
                    if stop_pct is not None:
                        stop_px = px * (1 - stop_pct)
                    else:
                        atr = data[code]["atr14"].get(day)
                        stop_px = px - atr_mult * atr if pd.notna(atr) else None
                    cash -= cost
                    positions[code] = {"code": code, "entry_date": day,
                                       "entry_price": px, "shares": shares,
                                       "cost": cost, "held": 0, "stop_px": stop_px}
        curve.append({"date": day, "equity": mark(day)})

    # 收尾：按最后收盘价强平估值（不实际卖出，计入净值）
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


def interval_days(index_df, start, end=None):
    start = pd.Timestamp(start)
    mask = index_df["date"] >= start
    if end:
        mask &= index_df["date"] <= pd.Timestamp(end)
    return index_df[mask]["date"].tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--capital", type=float, default=100000)
    ap.add_argument("--strategy", default="momentum")
    ap.add_argument("--no-regime", action="store_true", help="关闭指数 regime 过滤")
    stop_group = ap.add_mutually_exclusive_group()
    stop_group.add_argument("--stop", type=float, default=None,
                            help=f"固定百分比止损，默认 {DEFAULT_STOP}")
    stop_group.add_argument("--atr-stop", type=float, default=None,
                            help="ATR 止损倍数：止损价=买入开盘价-MULT×ATR14（与 --stop 互斥）")
    ap.add_argument("--hold", type=int, default=DEFAULT_HOLD,
                    help="持有上限交易日，0 表示不限制")
    args = ap.parse_args()
    start = pd.Timestamp(args.start)

    if args.atr_stop is not None:
        stop_pct, atr_mult = None, args.atr_stop
    else:
        stop_pct = args.stop if args.stop is not None else DEFAULT_STOP
        atr_mult = None

    home = Path(os.environ.get("ASTOCK_HOME", Path.home() / ".astock"))
    conn = store.connect(home / "astock.db")
    cons = store.load_constituents(conn)
    names = dict(zip(cons["code"], cons["name"]))
    print("加载股票池并预计算信号…")
    pool = dataset.build_pool_bars(conn, cons["code"].tolist())
    strat = STRATEGIES_BY_NAME[args.strategy]
    data = prepare_data(pool, strat, need_atr=atr_mult is not None)

    regime, index_df = load_regime()
    days = interval_days(index_df, start, args.end)
    end = days[-1]

    res = simulate(data, days, regime, {
        "capital": args.capital, "stop_pct": stop_pct,
        "atr_mult": atr_mult, "hold": args.hold,
        "no_regime": args.no_regime,
    })
    eq, tdf, positions = res["curve"], res["trades"], res["positions"]
    final, ret, dd = res["final"], res["ret"], res["maxdd"]
    idx = index_df[(index_df["date"] >= start) & (index_df["date"] <= end)]
    bench = idx["close"].iloc[-1] / idx["close"].iloc[0] - 1

    hold_desc = f"持{args.hold}日" if args.hold else "不限持有期"
    stop_desc = (f"止损-{stop_pct:.0%}" if stop_pct is not None
                 else f"止损{atr_mult:g}×ATR14")
    print(f"\n{'='*66}")
    print(f"区间 {start.date()} → {end.date()}  本金 {args.capital:,.0f}")
    print(f"策略 {args.strategy}（{hold_desc}/{stop_desc}/{'regime过滤' if not args.no_regime else '无过滤'}）")
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
