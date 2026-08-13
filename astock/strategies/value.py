import pandas as pd

from .base import Strategy, StrategyMeta

# 参数：PE 分位窗口 756 个交易日（约 3 年），min_periods=60；
# 买入阈值 q30，卖出阈值 q70；ROE 买入线 10%、卖出线 8%；营收增长 > 0。
PE_WINDOW = 756      # 约 3 年交易日
MIN_PERIODS = 60
BUY_Q, SELL_Q = 0.3, 0.7
ROE_BUY, ROE_SELL = 10.0, 8.0


class ValueStrategy(Strategy):
    """价值筛选（长线）：PE 低于自身 3 年 30% 分位且 ROE>10%、营收增长>0 时买入；
    PE 高于自身 70% 分位或 ROE<8% 时卖出。信号在条件边沿触发一次。"""

    meta = StrategyMeta(
        name="value", horizon="long",
        description="PE低于自身3年30%分位且ROE>10%、营收增长>0时买入；"
                    "PE高于自身70%分位或ROE<8%时卖出")

    def signals(self, bars):
        df = bars.set_index("date")
        pe_q30 = df["pe"].rolling(PE_WINDOW, min_periods=MIN_PERIODS).quantile(BUY_Q)
        pe_q70 = df["pe"].rolling(PE_WINDOW, min_periods=MIN_PERIODS).quantile(SELL_Q)
        cheap = ((df["pe"] < pe_q30)
                 & (df["roe"] > ROE_BUY)
                 & (df["revenue_growth"] > 0)).fillna(False)
        expensive = ((df["pe"] > pe_q70) | (df["roe"] < ROE_SELL)).fillna(False)
        buy = cheap & ~cheap.shift(1, fill_value=False)          # 边沿触发
        sell = expensive & ~expensive.shift(1, fill_value=False)
        sig = pd.Series(0, index=df.index)
        sig[buy] = 1
        sig[sell] = -1
        return sig

    def exit_hint(self):
        return "PE回升至自身70%分位以上或ROE恶化时卖出"
