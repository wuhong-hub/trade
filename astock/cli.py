import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from .backtest import rank
from .data import fetcher, store
from .recommend import engine as rec_engine
from .tracker import tracker

HISTORY_YEARS = 3
FETCH_SLEEP = 0.3
INDEX_CODES = ["000300", "000905"]  # 沪深300 + 中证500


def default_home():
    return Path(os.environ.get("ASTOCK_HOME", Path.home() / ".astock"))


def cmd_update(args):
    home = default_home()
    conn = store.connect(home / "astock.db")
    parts = []
    for c in INDEX_CODES:
        try:
            parts.append(fetcher.fetch_index_constituents(c))
        except Exception as e:
            print(f"[错误] 指数 {c} 成分股名单接口抓取失败：{e}")
            print("       请检查网络，或升级 akshare 后重试：pip install -U akshare")
            return 1
    cons = pd.concat(parts)
    cons = cons.drop_duplicates("code").reset_index(drop=True)
    store.save_constituents(conn, cons)
    print(f"成分股：{len(cons)} 只（沪深300+中证500 去重）")
    years = args.years if getattr(args, "years", None) else HISTORY_YEARS
    backfill_from = pd.Timestamp.today() - pd.DateOffset(years=years)
    earliest = backfill_from.strftime("%Y-%m-%d")
    start_year = str(backfill_from.year)
    ok, failed = 0, []
    for i, code in enumerate(cons["code"], 1):
        try:
            last = store.last_bar_date(conn, code)
            if getattr(args, "years", None):
                start = earliest  # --years 回填：强制从 N 年前抓，upsert 幂等
            else:
                start = earliest if last is None else (
                    pd.Timestamp(last) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            bars = fetcher.fetch_daily_bars(code, start)
            if len(bars):
                store.upsert_bars(conn, code, bars)
            store.upsert_pe(conn, code, fetcher.fetch_pe_series(code))
            if getattr(args, "years", None):
                fin = fetcher.fetch_financials(code, start_year)
            else:
                fin = fetcher.fetch_financials(code)
            store.upsert_financials(conn, code, fin)
            ok += 1
        except Exception as e:
            failed.append(code)
            print(f"  [警告] {code} 抓取失败：{e}")
        if i % 50 == 0:
            print(f"  进度 {i}/{len(cons)}")
        time.sleep(FETCH_SLEEP)
    print(f"完成：{ok} 成功 / {len(failed)} 失败")
    if failed:
        print(f"失败代码（前 10）：{failed[:10]}")
    return 0


def cmd_iterate(args):
    home = default_home()
    conn = store.connect(home / "astock.db")
    print("滚动回测中（最近 2 年，4 个半年窗口）…")
    state = rank.run_iteration(conn, home / "state.json", home / "history.jsonl")
    for horizon, label in (("short", "短线"), ("long", "长线")):
        print(f"\n{label}策略排名：")
        for e in state["ranking"][horizon]:
            mark = "★" if e["strategy"] == state["best"][horizon] else " "
            last = e["windows"][-1]
            print(f" {mark} {e['strategy']:<14} 综合分 {e['score']:>6.2f}  "
                  f"最近窗口: 胜率 {last['win_rate']:.0%}  "
                  f"年化 {last['annual_return']:.1%}  回撤 {last['max_drawdown']:.1%}")
    return 0


def _check_stale(conn):
    row = conn.execute("SELECT MAX(date) FROM daily_bars").fetchone()
    if not row or not row[0]:
        print("[提醒] 本地无行情数据，请先运行 astock update")
        return
    last = pd.Timestamp(row[0])
    if len(pd.bdate_range(last, pd.Timestamp.today())) - 1 > 3:
        print(f"[提醒] 行情数据停留在 {row[0]}，已超过 3 个交易日未更新，"
              f"建议先运行 astock update")


def cmd_recommend(args):
    home = default_home()
    state_path = home / "state.json"
    if not state_path.exists():
        print("尚未迭代过策略，请先运行 astock iterate")
        return 2
    conn = store.connect(home / "astock.db")
    _check_stale(conn)
    state = json.loads(state_path.read_text())
    recs, summary = rec_engine.generate_recommendations(conn, state)
    print(f"策略迭代时间：{state['updated_at']}")
    for horizon, label in (("short", "短线（持有数天~两周）"), ("long", "长线（数月）")):
        s = summary[horizon]
        if s["strategy"] is None:
            print(f"\n=== {label} | 暂无有效策略（回测全部窗口无交易）===")
            continue
        extra = "（样本 0 笔，胜率按中性 50% 计）" if s.get("n_trades") == 0 else ""
        print(f"\n=== {label} | 策略 {s['strategy']} "
              f"(最近窗口: 胜率 {s['win_rate']:.0%} 年化 {s['annual_return']:.1%} "
              f"最大回撤 {s['max_drawdown']:.1%}){extra} ===")
        if not recs[horizon]:
            print("  今日无符合条件的股票")
        for r in recs[horizon]:
            stop = f"止损 {r.stop_price}" if r.stop_price is not None else r.exit_hint
            print(f"  {r.code} {r.name}  现价 {r.price}  建议仓位 {r.position_pct}%")
            print(f"    理由：{r.reason}")
            print(f"    风控：{stop}")
    print(f"\n{rec_engine.DISCLAIMER}")
    tracker.record_recommendations(
        conn, datetime.now().strftime("%Y-%m-%d"), recs)
    return 0


def cmd_report(args):
    home = default_home()
    conn = store.connect(home / "astock.db")
    result = tracker.evaluate(conn)
    o = result["overall"]
    print(f"历史推荐 {o['n']} 条 | 命中率 {o['hit_rate']:.0%} | "
          f"平均收益 {o['avg_ret']:.1%}")
    for h, s in result["by_horizon"].items():
        label = "短线" if h == "short" else "长线"
        print(f"  {label}: {s['n']} 条, 命中率 {s['hit_rate']:.0%}, "
              f"平均收益 {s['avg_ret']:.1%}")
    if not result["detail"].empty:
        print("\n明细（最近 10 条）：")
        for r in result["detail"].tail(10).itertuples():
            print(f"  {r.rec_date} {r.code} [{r.horizon}/{r.strategy}] "
                  f"收益 {r.ret:+.1%}（{r.days} 个交易日）")
    print(f"\n{rec_engine.DISCLAIMER}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="astock", description="A股量化策略推荐工具")
    sub = parser.add_subparsers(dest="command", required=True)
    p_update = sub.add_parser("update", help="增量更新行情与财务数据")
    p_update.add_argument("--years", type=int, default=None, metavar="N",
                          help="回填 N 年历史数据（默认增量更新，不回填）")
    sub.add_parser("iterate", help="滚动回测并优选策略")
    sub.add_parser("recommend", help="输出当前最优策略的推荐")
    sub.add_parser("report", help="历史推荐效果跟踪")
    args = parser.parse_args(argv)
    return {"update": cmd_update, "iterate": cmd_iterate,
            "recommend": cmd_recommend, "report": cmd_report}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
