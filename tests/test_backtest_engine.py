import pandas as pd
import pytest
from astock.backtest import engine
from tests.conftest import make_bars


def _sig(bars, buy_i):
    s = pd.Series(0, index=bars["date"])
    s.iloc[buy_i] = 1
    return s


def test_short_max_hold_exit():
    # 每天 open=close=10+i（严格上涨）。index 0 买入信号 → index 1 开盘 11 买入；
    # 持有满 15 天（index 16）按收盘 26 卖出。
    bars = make_bars([10 + i for i in range(20)])
    trades = engine.simulate_trades("X", bars, _sig(bars, 0), "short")
    assert len(trades) == 1
    t = trades[0]
    assert t.entry_price == 11.0
    assert t.exit_price == 26.0
    assert t.holding_days == 15
    assert t.gross_ret == pytest.approx(26 / 11 - 1)
    assert t.net_ret == pytest.approx(26 / 11 - 1 - engine.COST)


def test_short_stop_loss():
    # 买入后次日 low 击穿 entry*0.93 → 按 entry*0.93 止损。
    bars = make_bars([100, 100, 90, 90, 90], opens=[100, 100, 90, 90, 90])
    bars.loc[2, "low"] = 92.0  # entry=100，93 止损线被击穿
    trades = engine.simulate_trades("X", bars, _sig(bars, 0), "short")
    assert len(trades) == 1
    assert trades[0].exit_price == 93.0
    assert trades[0].net_ret == pytest.approx(-0.07 - engine.COST)


def test_long_exits_only_on_sell_signal():
    bars = make_bars([10 + i for i in range(15)])
    sig = _sig(bars, 0)
    sig.iloc[5] = -1  # index 5 卖出信号 → index 6 开盘卖出
    trades = engine.simulate_trades("X", bars, sig, "long")
    assert len(trades) == 1
    assert trades[0].entry_price == 11.0
    assert trades[0].exit_price == 16.0  # open[6] = 10+6


def test_open_position_force_closed_at_end():
    bars = make_bars([10 + i for i in range(5)])
    trades = engine.simulate_trades("X", bars, _sig(bars, 2), "long")
    assert len(trades) == 1
    assert trades[0].exit_date == bars["date"].iloc[-1]


def test_equity_curve_and_metrics():
    # 单笔交易：entry_price=100，期间收盘 100→110→121。
    bars = make_bars([100, 110, 121])
    t = engine.Trade("X", bars["date"].iloc[0], 100.0,
                     bars["date"].iloc[2], 121.0, 0.21, 0.21 - engine.COST, 2)
    eq = engine.equity_curve([t], {"X": bars})
    assert len(eq) == 3
    assert eq.iloc[1] == pytest.approx(0.1)
    r = engine.compute_metrics("s", "w", [t], {"X": bars})
    assert r.n_trades == 1
    assert r.win_rate == 1.0
    cum = (1 + eq).cumprod()
    assert r.max_drawdown == pytest.approx((cum / cum.cummax() - 1).min())


def test_compute_metrics_empty():
    r = engine.compute_metrics("s", "w", [], {})
    assert r.n_trades == 0 and r.win_rate == 0.0 and r.sharpe == 0.0


def test_run_backtest_window_filter():
    # 两个窗口数据：只在窗口内产生交易。
    bars = make_bars([10 + i for i in range(40)])
    pool = {"X": bars}

    class BuyAtStart:
        class meta:
            name = "stub"
            horizon = "short"

        def signals(self, b):
            s = pd.Series(0, index=b["date"])
            s.iloc[0] = 1
            return s

    r = engine.run_backtest(BuyAtStart(), pool,
                            start=bars["date"].iloc[0], end=bars["date"].iloc[-1])
    assert r.n_trades >= 1
    assert r.strategy == "stub"
