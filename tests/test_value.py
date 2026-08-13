import pandas as pd
from astock.strategies.value import ValueStrategy
from tests.conftest import make_bars


def _bars_with_pe(pes, roe=12.0, growth=5.0):
    bars = make_bars([10.0] * len(pes))
    bars["pe"] = pes
    bars["roe"] = roe
    bars["revenue_growth"] = growth
    return bars


def test_buy_when_pe_below_own_q30():
    # 前 70 天 PE=20，第 71 天（index 70）起 PE=10。
    # index 70 时滚动窗口（61 个值：60 个 20 + 1 个 10，min_periods=60）的
    # q30 = 20 > 10 → 低估条件成立且为首次成立（边沿），产生买入信号。
    pes = [20.0] * 70 + [10.0] * 10
    sig = ValueStrategy().signals(_bars_with_pe(pes))
    assert sig.iloc[70] == 1
    assert (sig == 1).sum() == 1  # 条件持续但只在边沿触发一次


def test_no_buy_when_roe_low():
    pes = [20.0] * 70 + [10.0] * 10
    sig = ValueStrategy().signals(_bars_with_pe(pes, roe=5.0))
    assert (sig == 1).sum() == 0


def test_sell_when_pe_above_q70():
    # 前 70 天 PE=10，之后 PE=25 超过窗口 q70 → 卖出信号边沿。
    pes = [10.0] * 70 + [25.0] * 10
    sig = ValueStrategy().signals(_bars_with_pe(pes))
    assert (sig == -1).sum() >= 1
    assert sig[sig == -1].index[0] == pd.bdate_range("2024-01-01", periods=80)[70]
