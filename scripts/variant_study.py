"""变体矩阵研究：7 个止损/持有期/策略变体 × 3 个半年区间的组合模拟。

一次加载股票池、按策略预计算一次信号，进程内复用 portfolio_sim.simulate。
结果 CSV 存 ~/.astock/variant_study.csv，终端打印 markdown 汇总表。

用法：.venv/bin/python scripts/variant_study.py
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

# (标签, 策略, 止损/持有参数)
VARIANTS = [
    ("1 momentum stop=7% hold=15", "momentum",
     dict(stop_pct=0.07, atr_mult=None, hold=15)),
    ("2 momentum stop=10% hold=15", "momentum",
     dict(stop_pct=0.10, atr_mult=None, hold=15)),
    ("3 momentum atr=2.0 hold=15", "momentum",
     dict(stop_pct=None, atr_mult=2.0, hold=15)),
    ("4 momentum stop=7% hold=0", "momentum",
     dict(stop_pct=0.07, atr_mult=None, hold=0)),
    ("5 momentum atr=2.0 hold=0", "momentum",
     dict(stop_pct=None, atr_mult=2.0, hold=0)),
    ("6 ma_trend_vol stop=7% hold=0", "ma_trend_vol",
     dict(stop_pct=0.07, atr_mult=None, hold=0)),
    ("7 ma_trend_vol atr=2.0 hold=0", "ma_trend_vol",
     dict(stop_pct=None, atr_mult=2.0, hold=0)),
]


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
    regime, index_df = ps.load_regime()

    rows = []
    bench = {}
    for label, start, end in INTERVALS:
        days = ps.interval_days(index_df, start, end)
        real_end = days[-1]
        idx = index_df[(index_df["date"] >= pd.Timestamp(start))
                       & (index_df["date"] <= real_end)]
        bench[label] = idx["close"].iloc[-1] / idx["close"].iloc[0] - 1
        for vlabel, strat_name, vparams in VARIANTS:
            res = ps.simulate(data_by_strat[strat_name], days, regime,
                              {"capital": 100000, **vparams})
            tdf = res["trades"]
            row = {
                "variant": vlabel, "interval": label,
                "start": str(pd.Timestamp(start).date()),
                "end": str(real_end.date()),
                "ret": res["ret"], "maxdd": res["maxdd"],
                "trades": len(tdf),
                "win_rate": (tdf["ret"] > 0).mean() if len(tdf) else float("nan"),
                "avg_ret": tdf["ret"].mean() if len(tdf) else float("nan"),
                "open_pos": len(res["positions"]),
                "bench": bench[label],
            }
            rows.append(row)
            print(f"[{label}] {vlabel}: ret={row['ret']:+.1%} dd={row['maxdd']:.1%} "
                  f"trades={row['trades']} win={row['win_rate']:.0%}", flush=True)

    df = pd.DataFrame(rows)
    out_csv = home / "variant_study.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nCSV 已存 {out_csv}")

    # markdown 汇总表
    print("\n| 变体 | 区间 | 收益率 | 最大回撤 | 笔数 | 胜率 | 平均单笔 | 期末持仓 |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        wr = f"{r['win_rate']:.0%}" if pd.notna(r["win_rate"]) else "-"
        ar = f"{r['avg_ret']:+.2%}" if pd.notna(r["avg_ret"]) else "-"
        print(f"| {r['variant']} | {r['interval']} | {r['ret']:+.1%} | {r['maxdd']:.1%} "
              f"| {r['trades']} | {wr} | {ar} | {r['open_pos']} |")
    print("\n各区间沪深300 涨跌幅：" +
          "，".join(f"{label} {bench[label]:+.1%}" for label, _, _ in INTERVALS))


if __name__ == "__main__":
    main()
