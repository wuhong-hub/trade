import pandas as pd


def make_bars(closes, volumes=None, opens=None, start="2024-01-01"):
    """按收盘价序列构造人造日线 bars，列：date/open/high/low/close/volume。
    默认 open=close、high=close*1.01、low=close*0.99、volume=1000。"""
    n = len(closes)
    dates = pd.bdate_range(start, periods=n)
    opens = opens if opens is not None else list(closes)
    volumes = volumes if volumes is not None else [1000.0] * n
    return pd.DataFrame({
        "date": dates,
        "open": [float(x) for x in opens],
        "high": [float(c) * 1.01 for c in closes],
        "low": [float(c) * 0.99 for c in closes],
        "close": [float(c) for c in closes],
        "volume": [float(v) for v in volumes],
    })
