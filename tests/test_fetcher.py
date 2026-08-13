import pandas as pd
import pytest
from astock.data import fetcher


class FakeAk:
    """模拟 akshare 接口返回的原始（中文列名）数据。"""

    def index_stock_cons_csindex(self, symbol):
        return pd.DataFrame({"成分券代码": ["600000", "1"], "成分券名称": ["浦发银行", "平安银行"]})

    def stock_zh_a_hist(self, symbol, period, start_date, end_date, adjust):
        assert adjust == "qfq"
        return pd.DataFrame({
            "日期": ["2024-01-02", "2024-01-03"],
            "开盘": [10.0, 11.0], "最高": [10.5, 11.5], "最低": [9.5, 10.5],
            "收盘": [10.2, 11.2], "成交量": [10000, 12000],
        })

    def stock_a_indicator_lg(self, symbol):
        return pd.DataFrame({"trade_date": ["2024-01-02"], "pe": [15.0], "total_mv": [1e10]})

    def stock_financial_analysis_indicator(self, symbol, start_year):
        return pd.DataFrame({
            "日期": ["2023-12-31"],
            "净资产收益率(%)": ["12.5"],
            "主营业务收入增长率(%)": ["8.0"],
        })


@pytest.fixture
def fake_ak(monkeypatch):
    monkeypatch.setattr(fetcher, "ak", FakeAk())
    monkeypatch.setattr(fetcher.time, "sleep", lambda s: None)


def test_fetch_index_constituents(fake_ak):
    df = fetcher.fetch_index_constituents("000300")
    assert df["code"].tolist() == ["600000", "000001"]  # 补齐 6 位
    assert df["name"].tolist() == ["浦发银行", "平安银行"]


def test_fetch_daily_bars_columns(fake_ak):
    df = fetcher.fetch_daily_bars("600000", "2024-01-01")
    assert list(df.columns) == ["date", "open", "high", "low", "close", "volume"]
    assert df["close"].tolist() == [10.2, 11.2]
    assert pd.api.types.is_datetime64_any_dtype(df["date"])


def test_fetch_pe_series(fake_ak):
    df = fetcher.fetch_pe_series("600000")
    assert list(df.columns) == ["date", "pe"]
    assert df["pe"].tolist() == [15.0]


def test_fetch_financials(fake_ak):
    df = fetcher.fetch_financials("600000")
    assert list(df.columns) == ["report_date", "roe", "revenue_growth"]
    assert df["roe"].tolist() == [12.5]
    assert df["revenue_growth"].tolist() == [8.0]


def test_retry_on_failure(monkeypatch):
    calls = {"n": 0}

    class FlakyAk:
        def stock_a_indicator_lg(self, symbol):
            calls["n"] += 1
            if calls["n"] < 3:
                raise ConnectionError("boom")
            return pd.DataFrame({"trade_date": ["2024-01-02"], "pe": [15.0]})

    monkeypatch.setattr(fetcher, "ak", FlakyAk())
    monkeypatch.setattr(fetcher.time, "sleep", lambda s: None)
    df = fetcher.fetch_pe_series("600000")
    assert calls["n"] == 3
    assert df["pe"].tolist() == [15.0]


def test_retry_exhausted_raises(monkeypatch):
    class BadAk:
        def stock_a_indicator_lg(self, symbol):
            raise ConnectionError("always fails")

    monkeypatch.setattr(fetcher, "ak", BadAk())
    monkeypatch.setattr(fetcher.time, "sleep", lambda s: None)
    with pytest.raises(ConnectionError):
        fetcher.fetch_pe_series("600000")
