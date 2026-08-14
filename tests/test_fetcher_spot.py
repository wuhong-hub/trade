import pandas as pd
import pytest
from astock.data import fetcher


SPOT_TEXT = (
    'var hq_str_sz300475="香农芯创,161.500,157.430,160.900,163.180,157.600,'
    '160.850,160.910,21316128,3419888894.400,400,160.850,400,160.840,2600,'
    '160.830,100,160.820,400,160.810,200,160.910,2900,160.920,7000,160.930,'
    '1300,160.950,900,160.960,2026-08-14,14:14:30,00";\n'
    'var hq_str_sh600000="浦发银行,10.00,9.90,10.10,10.20,9.80,'
    '10.09,10.10,50000,500000.0,400,10.09,400,10.08,2600,'
    '10.07,100,10.06,400,10.05,200,10.10,2900,10.11,7000,10.12,'
    '1300,10.13,900,10.14,2026-08-14,14:14:30,00";\n'
)

INDEX_SPOT_TEXT = (
    'var hq_str_s_sh000300="沪深300,4668.4681,4.5168,0.10,1487552,46039790";\n'
)


@pytest.fixture
def fake_spot(monkeypatch):
    monkeypatch.setattr(fetcher, "_fetch_spot_raw",
                        lambda symbols: SPOT_TEXT + INDEX_SPOT_TEXT)
    monkeypatch.setattr(fetcher.time, "sleep", lambda s: None)


def test_fetch_spot_quotes(fake_spot):
    df = fetcher.fetch_spot_quotes(["300475", "600000"])
    assert set(df["code"]) == {"300475", "600000"}
    r = df[df["code"] == "300475"].iloc[0]
    assert r["name"] == "香农芯创"
    assert r["price"] == 160.90
    assert r["prev_close"] == 157.43
    assert r["open"] == 161.50
    assert r["high"] == 163.18
    assert r["low"] == 157.60
    assert r["volume"] == 21316128
    assert r["date"] == "2026-08-14"
    assert r["time"] == "14:14:30"


def test_fetch_spot_quotes_excludes_index_rows(fake_spot):
    """s_ 前缀的指数行不应混进个股报价结果。"""
    df = fetcher.fetch_spot_quotes(["300475"])
    assert df["code"].tolist() == ["300475"]


def test_fetch_index_spot(fake_spot):
    idx = fetcher.fetch_index_spot("sh000300")
    assert idx["name"] == "沪深300"
    assert idx["price"] == 4668.4681
    assert idx["prev_close"] == pytest.approx(4668.4681 - 4.5168)


def test_fetch_index_spot_empty_raises(monkeypatch):
    monkeypatch.setattr(fetcher, "_fetch_spot_raw", lambda symbols: "")
    monkeypatch.setattr(fetcher.time, "sleep", lambda s: None)
    with pytest.raises(RuntimeError):
        fetcher.fetch_index_spot("sh000300")


def test_spot_chunking(monkeypatch):
    """超过 SPOT_CHUNK 个代码时分批请求。"""
    calls = []

    def fake_raw(symbols):
        calls.append(symbols)
        return SPOT_TEXT

    monkeypatch.setattr(fetcher, "_fetch_spot_raw", fake_raw)
    monkeypatch.setattr(fetcher, "SPOT_CHUNK", 2)
    monkeypatch.setattr(fetcher.time, "sleep", lambda s: None)
    fetcher.fetch_spot_quotes(["300475", "600000", "000001"])
    assert calls == [["sz300475", "sh600000"], ["sz000001"]]
