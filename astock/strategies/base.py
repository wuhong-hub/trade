from dataclasses import dataclass

import pandas as pd


@dataclass
class StrategyMeta:
    name: str
    horizon: str  # "short" 或 "long"
    description: str


class Strategy:
    """策略基类：纯函数，不碰网络与磁盘。

    signals() 输入单只股票的 bars（短线只需 OHLCV，长线含 pe/roe/revenue_growth），
    输出 index=date、值 1/-1/0 的信号 Series。
    """

    meta: StrategyMeta

    def signals(self, bars: pd.DataFrame) -> pd.Series:
        raise NotImplementedError

    def reason(self, bars: pd.DataFrame, date) -> str:
        return self.meta.description

    def exit_hint(self) -> str:
        return ""
