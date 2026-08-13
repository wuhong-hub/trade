import pandas as pd

from .base import Strategy, StrategyMeta


class MATrendStrategy(Strategy):
    meta = StrategyMeta(
        name="ma_trend", horizon="short",
        description="5日均线上穿20日均线（金叉）买入，下穿（死叉）卖出")

    def __init__(self, fast=5, slow=20):
        self.fast = fast
        self.slow = slow

    def signals(self, bars):
        df = bars.set_index("date")
        fast_ma = df["close"].rolling(self.fast).mean()
        slow_ma = df["close"].rolling(self.slow).mean()
        cross_up = ((fast_ma > slow_ma)
                    & (fast_ma.shift(1) <= slow_ma.shift(1))).fillna(False)
        cross_dn = ((fast_ma < slow_ma)
                    & (fast_ma.shift(1) >= slow_ma.shift(1))).fillna(False)
        sig = pd.Series(0, index=df.index)
        sig[cross_up] = 1
        sig[cross_dn] = -1
        return sig

    def exit_hint(self):
        return "死叉（5日线下穿20日线）卖出"
