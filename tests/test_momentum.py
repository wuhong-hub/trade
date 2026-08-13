from astock.strategies.momentum import MomentumStrategy
from tests.conftest import make_bars


def test_breakout_with_volume_buy():
    # 前 24 天收盘恒为 10、量恒为 1000；第 25 根（index 24）收盘 12 创 20 日新高，
    # 成交量 3000 > 1.5 * 20 日均量 1000 → 买入信号恰好在 index 24。
    closes = [10] * 24 + [12]
    volumes = [1000] * 24 + [3000]
    bars = make_bars(closes, volumes=volumes)
    sig = MomentumStrategy().signals(bars)
    assert sig.iloc[24] == 1
    assert (sig == 1).sum() == 1


def test_breakout_without_volume_no_buy():
    # 同样创新高但量没放大 → 无买入信号。
    closes = [10] * 24 + [12]
    bars = make_bars(closes)  # volume 恒 1000
    sig = MomentumStrategy().signals(bars)
    assert (sig == 1).sum() == 0


def test_sell_below_prior_10day_low():
    # 先制造买入，随后收盘跌破前 10 日最低价 → 卖出信号。
    closes = [10] * 24 + [12, 12, 8.0]
    volumes = [1000] * 24 + [3000, 1000, 1000]
    bars = make_bars(closes, volumes=volumes)
    sig = MomentumStrategy().signals(bars)
    assert (sig == 1).sum() == 1
    assert (sig == -1).sum() == 1
    assert sig[sig == -1].index[0] > sig[sig == 1].index[0]
