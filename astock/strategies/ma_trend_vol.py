import pandas as pd

from .base import Strategy, StrategyMeta


class MATrendVolStrategy(Strategy):
    meta = StrategyMeta(
        name="ma_trend_vol", horizon="short",
        description="5日均线上穿20日均线且放量（≥2倍20日均量）买入，死叉卖出")

    def __init__(self, fast=5, slow=20, vol_mult=2.0):
        self.fast = fast
        self.slow = slow
        self.vol_mult = vol_mult

    def signals(self, bars):
        df = bars.set_index("date")
        fast_ma = df["close"].rolling(self.fast).mean()
        slow_ma = df["close"].rolling(self.slow).mean()
        cross_up = ((fast_ma > slow_ma)
                    & (fast_ma.shift(1) <= slow_ma.shift(1))).fillna(False)
        cross_dn = ((fast_ma < slow_ma)
                    & (fast_ma.shift(1) >= slow_ma.shift(1))).fillna(False)
        vol_surge = (df["volume"]
                     >= self.vol_mult * df["volume"].rolling(self.slow).mean()).fillna(False)
        sig = pd.Series(0, index=df.index)
        sig[cross_up & vol_surge] = 1
        sig[cross_dn] = -1
        return sig

    def exit_hint(self):
        return "死叉（5日线下穿20日线）卖出"
