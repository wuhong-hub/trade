import pandas as pd

from .base import Strategy, StrategyMeta


class VolumePriceStrategy(Strategy):
    meta = StrategyMeta(
        name="volume_price", horizon="short",
        description="缩量三连阴回调后，放量（>2倍20日均量）阳线反弹买入，跌破10日均线卖出")

    def signals(self, bars):
        df = bars.set_index("date")
        down = df["close"] < df["close"].shift(1)
        vol_down = df["volume"] < df["volume"].shift(1)
        pullback = (down & down.shift(1) & down.shift(2)
                    & vol_down & vol_down.shift(1)).fillna(False)
        vol_ma = df["volume"].rolling(20).mean()
        rebound = ((df["close"] > df["open"])
                   & (df["volume"] > 2 * vol_ma)).fillna(False)
        buy = pullback.shift(1).fillna(False) & rebound
        ma10 = df["close"].rolling(10).mean()
        sell = ((df["close"] < ma10)
                & (df["close"].shift(1) >= ma10.shift(1))).fillna(False)
        sig = pd.Series(0, index=df.index)
        sig[buy] = 1
        sig[sell] = -1
        return sig

    def exit_hint(self):
        return "跌破10日均线卖出"
