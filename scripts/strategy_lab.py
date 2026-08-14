"""策略实验室：用本地 2 年数据逐日模拟"每日推荐"，对比不同规则变体。

不做任何网络请求，只读 ~/.astock/astock.db。
用法：.venv/bin/python scripts/strategy_lab.py
"""
import os
from pathlib import Path

import pandas as pd

from astock.backtest import engine
from astock.data import dataset, store
from astock.strategies import STRATEGIES_BY_NAME

COST = engine.COST


def features(bars):
    """每根 bar 的入场日特征：距20日线幅度、当日涨幅、量比。"""
    df = bars.set_index("date")
    ma20 = df["close"].rolling(20).mean()
    vol_ma = df["volume"].rolling(20).mean()
    return pd.DataFrame({
        "dist_ma20": df["close"] / ma20 - 1,
        "day_gain": df["close"] / df["close"].shift(1) - 1,
        "vol_ratio": df["volume"] / vol_ma,
    }, index=df.index)


def simulate(bars, signals, stop, max_hold):
    """与 engine 同规则，但止损/持有期可配。返回 trade list。"""
    df = bars.reset_index(drop=True)
    sigs = df["date"].map(signals).fillna(0)
    trades = []
    in_pos = False
    entry_i = entry_price = None
    for i in range(len(df)):
        if not in_pos:
            if sigs.iloc[i] == 1 and i + 1 < len(df):
                in_pos, entry_i = True, i + 1
                entry_price = float(df["open"].iloc[i + 1])
            continue
        exit_i = exit_price = None
        if df["low"].iloc[i] <= entry_price * stop:
            exit_i, exit_price = i, entry_price * stop
        elif sigs.iloc[i] == -1 and i + 1 < len(df):
            exit_i, exit_price = i + 1, float(df["open"].iloc[i + 1])
        elif i - entry_i >= max_hold:
            exit_i, exit_price = i, float(df["close"].iloc[i])
        elif i == len(df) - 1:
            exit_i, exit_price = i, float(df["close"].iloc[i])
        if exit_i is not None and exit_i > entry_i:
            trades.append({"signal_date": df["date"].iloc[entry_i - 1],
                           "entry_date": df["date"].iloc[entry_i],
                           "ret": exit_price / entry_price - 1 - COST})
            in_pos = False
    return trades


def report(name, trades):
    if not trades:
        print(f"  {name:<38} 无交易")
        return
    df = pd.DataFrame(trades)
    n = len(df)
    win = (df["ret"] > 0).mean()
    avg = df["ret"].mean()
    gains = df.loc[df["ret"] > 0, "ret"].sum()
    losses = -df.loc[df["ret"] <= 0, "ret"].sum()
    pf = gains / losses if losses > 0 else float("inf")
    # 按信号日聚合：每天等权买入当日信号股，看"每日推荐组合"的表现
    daily = df.groupby("signal_date")["ret"].mean()
    print(f"  {name:<38} 交易 {n:>4} 笔  胜率 {win:.0%}  均收益 {avg:+.2%}  "
          f"盈亏比 {pf:.2f}  信号日 {len(daily)} 天  日均组合 {daily.mean():+.2%}")


def report_yearly(name, trades):
    """按年份拆分，检验结论是否跨行情 regime 稳健。"""
    if not trades:
        return
    df = pd.DataFrame(trades)
    df["year"] = pd.to_datetime(df["signal_date"]).dt.year
    print(f"  {name} 分年：")
    for y, g in df.groupby("year"):
        win = (g["ret"] > 0).mean()
        print(f"    {y}: 交易 {len(g):>5} 笔  胜率 {win:.0%}  均收益 {g['ret'].mean():+.2%}")


def load_regime(path="/home/wuhong/.astock/index_000300.csv", ma=60):
    """沪深300 收盘 > MA60 的日期集合（牛市 regime）。"""
    df = pd.read_csv(path, parse_dates=["date"])
    above = df["close"] > df["close"].rolling(ma).mean()
    return set(df.loc[above.fillna(False), "date"])


def main():
    home = Path(os.environ.get("ASTOCK_HOME", Path.home() / ".astock"))
    conn = store.connect(home / "astock.db")
    cons = store.load_constituents(conn)
    print("加载股票池…")
    pool = dataset.build_pool_bars(conn, cons["code"].tolist())
    print(f"池内 {len(pool)} 只，逐股预计算信号与特征…")

    # 每股预计算：两个策略的信号 + 特征
    per_stock = {}
    for code, bars in pool.items():
        per_stock[code] = {
            "bars": bars,
            "feat": features(bars),
            "sigs": {name: strat.signals(bars)
                     for name, strat in STRATEGIES_BY_NAME.items()
                     if strat.meta.horizon == "short"},
        }

    bull = load_regime()  # 沪深300 > MA60 的日期集合

    def collect(strat_name, stop, max_hold, filt=None, regime=False):
        all_trades = []
        for code, d in per_stock.items():
            for t in simulate(d["bars"], d["sigs"][strat_name], stop, max_hold):
                if regime and t["signal_date"] not in bull:
                    continue
                if filt is not None:
                    f = d["feat"].loc[t["signal_date"]]
                    if not filt(f):
                        continue
                all_trades.append(t)
        return all_trades

    # 过滤器定义
    no_ext = lambda f: f["dist_ma20"] <= 0.10          # 距20日线不超10%（不追高）
    no_spike = lambda f: f["day_gain"] <= 0.05          # 当日涨幅不超5%
    strong_vol = lambda f: f["vol_ratio"] >= 2.0        # 放量门槛2倍
    combo = lambda f: no_ext(f) and no_spike(f)

    for strat in ("momentum", "ma_trend"):
        print(f"\n{'='*78}\n策略：{strat}")
        report("基准（现状：止损-7% 持10日 无过滤）", collect(strat, 0.93, 10))
        report("止损 -5%", collect(strat, 0.95, 10))
        report("止损 -10%", collect(strat, 0.90, 10))
        report("持有 5 日", collect(strat, 0.93, 5))
        report("持有 15 日", collect(strat, 0.93, 15))
        report("过滤追高（距MA20≤10%）", collect(strat, 0.93, 10, no_ext))
        report("过滤当日大涨（≤5%）", collect(strat, 0.93, 10, no_spike))
        report("放量门槛 2 倍", collect(strat, 0.93, 10, strong_vol))
        report("组合过滤（不追高+不大涨）", collect(strat, 0.93, 10, combo))
        report("组合过滤+止损-5%+持5日", collect(strat, 0.95, 5, combo))
        report("★ regime过滤（指数>MA60才买）", collect(strat, 0.93, 10, regime=True))
        report("★ regime过滤+持15日", collect(strat, 0.93, 15, regime=True))

    # 分年稳健性：代表性变体 + regime 过滤
    print(f"\n{'='*78}\n分年稳健性")
    report_yearly("momentum 基准（-7%/10日）", collect("momentum", 0.93, 10))
    report_yearly("momentum 持有15日", collect("momentum", 0.93, 15))
    report_yearly("momentum regime+15日", collect("momentum", 0.93, 15, regime=True))
    report_yearly("ma_trend 放量2倍确认", collect("ma_trend", 0.93, 10, strong_vol))
    report_yearly("ma_trend 放量2倍+regime+15日", collect("ma_trend", 0.93, 15, strong_vol, regime=True))


if __name__ == "__main__":
    main()
