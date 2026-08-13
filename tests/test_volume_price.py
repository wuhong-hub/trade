from astock.strategies.volume_price import VolumePriceStrategy
from tests.conftest import make_bars


def test_rebound_after_quiet_pullback_buy():
    # 前 25 天缓涨（close=10+0.1*i），量恒 1000；
    # 第 26-28 根（index 25-27）三连阴且缩量（900/800/700）；
    # 第 29 根（index 28）低开高走收阳，量 3000 > 2 * 20 日均量(970)=1940 → 买入。
    closes = [10 + 0.1 * i for i in range(25)] + [12.0, 11.5, 11.0, 11.5]
    opens = list(closes)
    opens[28] = 10.5  # 反弹日开盘低于收盘 → 阳线
    volumes = [1000] * 25 + [900, 800, 700, 3000]
    bars = make_bars(closes, volumes=volumes, opens=opens)
    sig = VolumePriceStrategy().signals(bars)
    assert sig.iloc[28] == 1


def test_no_pullback_no_buy():
    # 没有三连阴缩量回调，单独一根放量阳线不触发买入。
    closes = [10 + 0.1 * i for i in range(28)] + [13.0]
    opens = list(closes)
    opens[28] = 12.0
    volumes = [1000] * 28 + [3000]
    bars = make_bars(closes, volumes=volumes, opens=opens)
    sig = VolumePriceStrategy().signals(bars)
    assert (sig == 1).sum() == 0
