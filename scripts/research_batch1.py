"""研究批次1·阶段1：信号侧三个候选的 10 年评估。

候选 1：短期反转策略 R1/R2/R3（研究脚本内实现，不入主程序）
  R1：ret20 < -10% 且当日收阳（close>open），边沿触发 → 买
  R2：RSI14（Wilder）< 30 后首次上穿 30 → 买
  R3：前 3 个交易日内创 20 日新低，当日收盘上穿 5 日均线 → 买
  退出/费用/仓位与基线完全相同（stop=10%、hold=15、次日开盘买、指数 MA60 regime）。
  反转策略不产生卖出信号（-1），退出只靠止损/持有期满。

候选 2：市场宽度 regime。breadth(t)=股票池中当日 close>各自 MA60 的比例（只用 ≤t 数据）。
  变体 B40/B50/B60（breadth 超阈值才开仓）与"指数MA60 且 breadth>0.5"双条件；
  策略固定 momentum(stop=10%, hold=15)，只换 regime 序列，对照"指数MA60 单条件"基线。

候选 3：ma_trend_vol + ADX14（Wilder）趋势强度过滤。
  买入信号日 ADX14>20 / >25 才生效；两组代表参数：atr=2.0/hold=0 与 stop=10%/hold=15；
  与无 ADX 的同参数组对比。

评估口径：3 个半年区间（A/B/C）+ 2016-08-15→最新按日历年分段（11 段）。
严禁未来函数：所有 rolling/shift/ewm 只用 ≤ 当日的数据（逐处见注释）。

用法：.venv/bin/python scripts/research_batch1.py [--candidate 1|2|3|all]
只读本地数据（~/.astock），无网络请求。结果 CSV 存 ~/.astock/research_batch1_*.csv。
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

BASELINE_PARAMS = dict(stop_pct=0.10, atr_mult=None, hold=15)


# ---------------------------------------------------------------- 信号函数
# 约定：输入单票 bars（含 date 列），输出 index=date、值 1/0 的买入信号 Series。
# 全部为时点 t 的判定，只用 ≤t 的数据；simulate 在 t+1 开盘买入。

def sig_r1(bars):
    """R1：ret20<-10% 且收阳，边沿触发（条件由假变真的首日）。"""
    df = bars.set_index("date")
    ret20 = df["close"] / df["close"].shift(20) - 1      # ≤t
    cond = ((ret20 < -0.10) & (df["close"] > df["open"])).fillna(False)
    buy = cond & ~cond.shift(1, fill_value=False)        # 边沿触发
    return pd.Series(0, index=df.index).mask(buy, 1)


def _rsi14_wilder(close):
    """Wilder RSI14：ewm(alpha=1/14) 等价 Wilder 平滑，只用 ≤t 数据。"""
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gain / loss.replace(0, float("nan"))
    return 100 - 100 / (1 + rs)


def sig_r2(bars):
    """R2：RSI14 < 30 后首次上穿 30 → 买（上穿本身即边沿）。"""
    df = bars.set_index("date")
    rsi = _rsi14_wilder(df["close"])
    buy = ((rsi >= 30) & (rsi.shift(1) < 30)).fillna(False)
    return pd.Series(0, index=df.index).mask(buy, 1)


def sig_r3(bars):
    """R3：前 1~3 个交易日内创 20 日新低，当日收盘上穿 MA5 → 买。"""
    df = bars.set_index("date")
    newlow = (df["low"] <= df["low"].rolling(20).min()).fillna(False)  # ≤t
    had_newlow = (newlow.shift(1) | newlow.shift(2)
                  | newlow.shift(3)).fillna(False)                     # ≤t-1
    ma5 = df["close"].rolling(5).mean()
    cross_up = ((df["close"] > ma5)
                & (df["close"].shift(1) <= ma5.shift(1))).fillna(False)
    return pd.Series(0, index=df.index).mask(had_newlow & cross_up, 1)


def _adx14(df):
    """Wilder ADX14：+DM/-DM → TR → 平滑 → DX → ADX。全部只用 ≤t 数据。"""
    high, low, close = df["high"], df["low"], df["close"]
    up = high.diff()
    dn = -low.diff()
    plus_dm = pd.Series(0.0, index=df.index)
    minus_dm = pd.Series(0.0, index=df.index)
    plus_dm[(up > dn) & (up > 0)] = up
    minus_dm[(dn > up) & (dn > 0)] = dn
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(),
                    (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / 14, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / 14, adjust=False).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return dx.ewm(alpha=1 / 14, adjust=False).mean()


def sig_mtv_adx(threshold):
    """ma_trend_vol 买入信号加 ADX14>threshold 过滤；卖出信号原样保留。"""
    base = STRATEGIES_BY_NAME["ma_trend_vol"]

    def fn(bars):
        df = bars.set_index("date")
        sig = base.signals(bars).copy()
        adx = _adx14(df)
        blocked = (sig == 1) & ~(adx > threshold).fillna(False)  # ADX 用 ≤t 数据
        sig[blocked] = 0
        return sig
    return fn


# ---------------------------------------------------------------- 数据准备

def prepare_with_sig(pool, sig_fn, need_atr=False):
    """与 ps.prepare_data 同构，但买入信号由 sig_fn 提供。"""
    data = {}
    for code, bars in pool.items():
        b = bars.set_index("date")
        d = {
            "bars": b,
            "sig": sig_fn(bars),
            "vol_ratio": b["volume"] / b["volume"].rolling(20).mean(),
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


def breadth_frame(pool, index_dates):
    """每票 above60（close>MA60，≤t）对齐到指数交易日，列=code。"""
    cols = {}
    for code, bars in pool.items():
        b = bars.set_index("date")
        above = (b["close"] > b["close"].rolling(60).mean())
        cols[code] = above.reindex(index_dates)
    return pd.DataFrame(cols, index=index_dates)


# ---------------------------------------------------------------- 评估骨架

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
    """configs: list of (group, variant, simulate_fn(days)->res)。逐区间评估并打印。"""
    rows = []
    for group, variant, sim_fn in configs:
        for label, start, end_s, days, bench in intervals:
            row = run_one(sim_fn, days)
            rows.append({"group": group, "variant": variant, "interval": label,
                         "start": start, "end": end_s, "bench": bench, **row})
            print(f"[{label}] {variant}: ret={row['ret']:+.1%} "
                  f"dd={row['maxdd']:.1%} trades={row['trades']}", flush=True)
    return rows


def print_table(rows, title):
    print(f"\n=== {title} ===")
    print("| 组 | 变体 | 区间 | 收益率 | 最大回撤 | 笔数 | 胜率 | 平均单笔 |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        wr = f"{r['win_rate']:.0%}" if pd.notna(r["win_rate"]) else "-"
        ar = f"{r['avg_ret']:+.2%}" if pd.notna(r["avg_ret"]) else "-"
        print(f"| {r['group']} | {r['variant']} | {r['interval']} | {r['ret']:+.1%} "
              f"| {r['maxdd']:.1%} | {r['trades']} | {wr} | {ar} |")


def save_csv(rows, home, name):
    out = home / name
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"CSV 已存 {out}", flush=True)


# ---------------------------------------------------------------- 主流程

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", default="all", choices=["1", "2", "3", "all"])
    args = ap.parse_args()
    want = {"1", "2", "3"} if args.candidate == "all" else {args.candidate}

    home = Path(os.environ.get("ASTOCK_HOME", Path.home() / ".astock"))
    conn = store.connect(home / "astock.db")
    cons = store.load_constituents(conn)
    print("加载股票池…", flush=True)
    pool = dataset.build_pool_bars(conn, cons["code"].tolist())
    print(f"股票池 {len(pool)} 只", flush=True)

    regime, index_df = ps.load_regime()   # 指数 MA60 单条件（基线 regime）
    intervals = build_intervals(index_df)

    # ---- 候选 1：短期反转 ------------------------------------------------
    if "1" in want:
        print("\n##### 候选 1：短期反转 R1/R2/R3 #####", flush=True)
        print("预计算信号（momentum 基线 + R1/R2/R3）…", flush=True)
        data_mom = ps.prepare_data(pool, STRATEGIES_BY_NAME["momentum"])
        reversal_data = {
            v: prepare_with_sig(pool, fn)
            for v, fn in [("R1 ret20<-10%+收阳", sig_r1),
                          ("R2 RSI14上穿30", sig_r2),
                          ("R3 新低后站上MA5", sig_r3)]
        }
        configs = [
            ("control", "基线 momentum stop=10% hold=15",
             lambda d: ps.simulate(data_mom, d, regime,
                                   {"capital": 100000, **BASELINE_PARAMS})),
        ]
        for v, dct in reversal_data.items():
            configs.append(("reversal", v, lambda d, dct=dct: ps.simulate(
                dct, d, regime, {"capital": 100000, **BASELINE_PARAMS})))
        rows = eval_configs(configs, intervals)
        save_csv(rows, home, "research_batch1_reversal.csv")
        print_table(rows, "候选 1 汇总")

    # ---- 候选 2：市场宽度 regime ----------------------------------------
    if "2" in want:
        print("\n##### 候选 2：市场宽度 regime #####", flush=True)
        if "1" not in want:
            print("预计算 momentum 信号…", flush=True)
            data_mom = ps.prepare_data(pool, STRATEGIES_BY_NAME["momentum"])
        idx_dates = pd.DatetimeIndex(index_df["date"])
        print("计算市场宽度（全池 close>MA60 比例）…", flush=True)
        bf = breadth_frame(pool, idx_dates)
        n_valid = bf.notna().sum(axis=1)
        # 早期指数交易日无股票数据（分母 0）→ 先换成 NaN 再除，结果为 NaN，
        # 后续 fillna(False) 处理
        breadth = bf.sum(axis=1) / n_valid.replace(0, float("nan"))

        # 宽度统计：2016-08-15 起各阈值以上时间占比 + 最新交易日宽度
        b_recent = breadth[breadth.index >= pd.Timestamp("2016-08-15")]
        print("\n=== 宽度序列统计（2016-08-15 → 最新）===")
        for thr in (0.4, 0.5, 0.6):
            print(f"breadth>{thr}: 时间占比 {(b_recent > thr).mean():.1%}")
        last_day = breadth.index[-1]
        print(f"最新交易日 {last_day.date()} 宽度 breadth={breadth.iloc[-1]:.3f} "
              f"（覆盖 {int(bf.notna().iloc[-1].sum())} 只有效股票）")

        idx_above = pd.Series(
            [regime.get(d, False) for d in idx_dates], index=idx_dates)
        regimes = {
            "B40 breadth>0.4": breadth > 0.4,
            "B50 breadth>0.5": breadth > 0.5,
            "B60 breadth>0.6": breadth > 0.6,
            "双条件 指数MA60&breadth>0.5": idx_above & (breadth > 0.5),
        }
        configs = [
            ("control", "基线 指数MA60 单条件",
             lambda d: ps.simulate(data_mom, d, regime,
                                   {"capital": 100000, **BASELINE_PARAMS})),
        ]
        for v, mask in regimes.items():
            rdict = dict(zip(mask.index, mask.fillna(False)))
            configs.append(("breadth", v, lambda d, rd=rdict: ps.simulate(
                data_mom, d, rd, {"capital": 100000, **BASELINE_PARAMS})))
        rows = eval_configs(configs, intervals)
        breadth_out = pd.DataFrame({"date": breadth.index,
                                    "breadth": breadth.values})
        breadth_out.to_csv(home / "research_batch1_breadth_series.csv", index=False)
        save_csv(rows, home, "research_batch1_breadth.csv")
        print_table(rows, "候选 2 汇总")

    # ---- 候选 3：ma_trend_vol + ADX 过滤 --------------------------------
    if "3" in want:
        print("\n##### 候选 3：ma_trend_vol + ADX 过滤 #####", flush=True)
        print("预计算信号（ma_trend_vol 无过滤 / ADX>20 / ADX>25）…", flush=True)
        data_sets = {
            "无ADX": ps.prepare_data(pool, STRATEGIES_BY_NAME["ma_trend_vol"],
                                    need_atr=True),
            "ADX>20": prepare_with_sig(pool, sig_mtv_adx(20), need_atr=True),
            "ADX>25": prepare_with_sig(pool, sig_mtv_adx(25), need_atr=True),
        }
        param_groups = [
            ("atr=2.0 hold=0", dict(stop_pct=None, atr_mult=2.0, hold=0)),
            ("stop=10% hold=15", dict(stop_pct=0.10, atr_mult=None, hold=15)),
        ]
        configs = []
        for plabel, pp in param_groups:
            for slabel, dct in data_sets.items():
                configs.append((f"adx[{plabel}]", f"{plabel} {slabel}",
                                lambda d, dct=dct, pp=pp: ps.simulate(
                                    dct, d, regime, {"capital": 100000, **pp})))
        rows = eval_configs(configs, intervals)
        save_csv(rows, home, "research_batch1_adx.csv")
        print_table(rows, "候选 3 汇总")


if __name__ == "__main__":
    main()
