from astock.strategies.ma_trend import MATrendStrategy
from tests.conftest import make_bars


def test_golden_cross_buy_signal():
    # 前 2 天收盘 12、随后 18 天收盘 10（让慢线在前期高于快线，避免下跌段触发死叉），
    # 第 21-25 天收盘 8，第 26-30 天收盘 12。
    # 手算：5 日均线在第 28 根（index 27）上穿 20 日均线（10.4 > 9.8），
    # 此前 fast<=slow，金叉恰好在 index 27 出现一次，全程无死叉。
    closes = [12] * 2 + [10] * 18 + [8] * 5 + [12] * 5
    bars = make_bars(closes)
    sig = MATrendStrategy().signals(bars)
    assert sig.iloc[27] == 1
    assert (sig == 1).sum() == 1
    assert (sig == -1).sum() == 0


def test_death_cross_sell_signal():
    # 与上面镜像：先低位 8，再拉高到 12（产生金叉），再跌回 8（产生死叉）。
    closes = [8] * 20 + [12] * 10 + [8] * 10
    bars = make_bars(closes)
    sig = MATrendStrategy().signals(bars)
    assert (sig == 1).sum() == 1
    assert (sig == -1).sum() == 1
    assert sig[sig == 1].index[0] < sig[sig == -1].index[0]
