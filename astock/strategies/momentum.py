import pandas as pd

from .base import Strategy, StrategyMeta


class MomentumStrategy(Strategy):
    meta = StrategyMeta(
        name="momentum", horizon="short",
        description="收盘价创20日新高且成交量放大至20日均量1.5倍以上买入，跌破前10日最低价卖出")

    def signals(self, bars):
        df = bars.set_index("date")
        high20 = df["close"].rolling(20).max()
        vol_ma = df["volume"].rolling(20).mean()
        low10_prior = df["low"].rolling(10).min().shift(1)
        buy = ((df["close"] >= high20)
               & (df["volume"] > 1.5 * vol_ma)).fillna(False)
        sell = (df["close"] < low10_prior).fillna(False)
        sig = pd.Series(0, index=df.index)
        sig[buy] = 1
        sig[sell] = -1
        return sig

    def exit_hint(self):
        return "跌破前10日最低价卖出"
