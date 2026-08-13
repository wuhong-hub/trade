from astock.strategies.ma_trend_vol import MATrendVolStrategy
from tests.conftest import make_bars


def test_golden_cross_with_volume_buy_signal():
    # 与 test_ma_trend 相同的价格序列：金叉恰在 index 27 出现一次。
    # index 27 当日放量 3000（其余 1000，20 日均量含当日约 1100，3000 >= 2*1100 成立）
    # → 恰好一次买入信号，无卖出信号。
    closes = [12] * 2 + [10] * 18 + [8] * 5 + [12] * 5
    volumes = [1000.0] * 30
    volumes[27] = 3000.0
    bars = make_bars(closes, volumes=volumes)
    sig = MATrendVolStrategy().signals(bars)
    assert sig.iloc[27] == 1
    assert (sig == 1).sum() == 1
    assert (sig == -1).sum() == 0


def test_golden_cross_without_volume_no_buy():
    # 同样的金叉，但量能全为 1000，不满足 >= 2 倍 20 日均量 → 无买入信号。
    closes = [12] * 2 + [10] * 18 + [8] * 5 + [12] * 5
    bars = make_bars(closes)
    sig = MATrendVolStrategy().signals(bars)
    assert (sig == 1).sum() == 0
    assert (sig == -1).sum() == 0
