from dataclasses import dataclass, field

import pandas as pd

COMMISSION = 0.0003   # 单边佣金
STAMP_TAX = 0.001     # 卖出印花税
COST = COMMISSION * 2 + STAMP_TAX  # 单次往返成本 0.0016

SHORT_MAX_HOLD = 15
SHORT_STOP = 0.93     # -7% 止损


@dataclass
class Trade:
    code: str
    entry_date: object
    entry_price: float
    exit_date: object
    exit_price: float
    gross_ret: float
    net_ret: float
    holding_days: int


@dataclass
class BacktestResult:
    strategy: str
    window: str
    n_trades: int
    win_rate: float
    annual_return: float
    max_drawdown: float
    sharpe: float
    trades: list = field(default_factory=list)


def simulate_trades(code, bars, signals, horizon):
    df = bars.reset_index(drop=True)
    sigs = df["date"].map(signals).fillna(0)
    max_hold = SHORT_MAX_HOLD if horizon == "short" else None
    stop = SHORT_STOP if horizon == "short" else None
    trades = []
    in_pos = False
    entry_i = entry_price = None
    for i in range(len(df)):
        if not in_pos:
            if sigs.iloc[i] == 1 and i + 1 < len(df):
                in_pos = True
                entry_i = i + 1
                entry_price = float(df["open"].iloc[i + 1])
            continue
        exit_i = exit_price = None
        if stop is not None and df["low"].iloc[i] <= entry_price * stop:
            exit_i, exit_price = i, entry_price * stop
        elif sigs.iloc[i] == -1 and i + 1 < len(df):
            exit_i, exit_price = i + 1, float(df["open"].iloc[i + 1])
        elif max_hold is not None and i - entry_i >= max_hold:
            exit_i, exit_price = i, float(df["close"].iloc[i])
        elif i == len(df) - 1:
            exit_i, exit_price = i, float(df["close"].iloc[i])
        if exit_i is not None and exit_i > entry_i:
            gross = exit_price / entry_price - 1
            trades.append(Trade(code, df["date"].iloc[entry_i], entry_price,
                                df["date"].iloc[exit_i], exit_price,
                                gross, gross - COST, exit_i - entry_i))
            in_pos = False
    return trades


def equity_curve(trades, bars_by_code):
    """组合日收益率：各持仓个股日收益（收盘价近似）的等权平均。"""
    daily = {}
    for t in trades:
        closes = bars_by_code[t.code].set_index("date")["close"]
        window = closes.loc[t.entry_date:t.exit_date]
        if len(window) == 0:
            continue
        rets = window.pct_change()
        rets.iloc[0] = window.iloc[0] / t.entry_price - 1
        for d, r in rets.dropna().items():
            daily.setdefault(d, []).append(r)
    if not daily:
        return pd.Series(dtype=float)
    return pd.Series({d: sum(v) / len(v) for d, v in daily.items()}).sort_index()


def compute_metrics(strategy, window, trades, bars_by_code):
    if not trades:
        return BacktestResult(strategy, window, 0, 0.0, 0.0, 0.0, 0.0, [])
    wins = sum(1 for t in trades if t.net_ret > 0)
    eq = equity_curve(trades, bars_by_code)
    cum = (1 + eq).cumprod()
    total = float(cum.iloc[-1] - 1) if len(cum) else 0.0
    days = len(eq)
    annual = (1 + total) ** (252 / days) - 1 if days else 0.0
    dd = float((cum / cum.cummax() - 1).min()) if len(cum) else 0.0
    std = eq.std()
    sharpe = float(eq.mean() / std * (252 ** 0.5)) if std and std > 0 else 0.0
    return BacktestResult(strategy, window, len(trades), wins / len(trades),
                          annual, dd, sharpe, trades)


def run_backtest(strategy, pool_bars, start=None, end=None):
    trades_all = []
    for code, bars in pool_bars.items():
        b = bars
        if start is not None:
            b = b[b["date"] >= start]
        if end is not None:
            b = b[b["date"] <= end]
        if len(b) < 30:
            continue
        sigs = strategy.signals(b.reset_index(drop=True))
        trades_all.extend(
            simulate_trades(code, b.reset_index(drop=True), sigs,
                            strategy.meta.horizon))
    label = f"{getattr(start, 'date', lambda: start)()}~{getattr(end, 'date', lambda: end)()}"
    return compute_metrics(strategy.meta.name, label, trades_all, pool_bars)
