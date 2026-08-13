import pandas as pd
from astock.data import store
from tests.conftest import make_bars


def test_bars_roundtrip(tmp_path):
    conn = store.connect(tmp_path / "t.db")
    df = make_bars([10, 11, 12])
    store.upsert_bars(conn, "600000", df)
    out = store.load_bars(conn, "600000")
    assert list(out.columns) == ["date", "open", "high", "low", "close", "volume"]
    assert len(out) == 3
    assert out["close"].tolist() == [10.0, 11.0, 12.0]
    assert out["date"].iloc[0] == df["date"].iloc[0]


def test_upsert_is_idempotent(tmp_path):
    conn = store.connect(tmp_path / "t.db")
    store.upsert_bars(conn, "600000", make_bars([10, 11]))
    store.upsert_bars(conn, "600000", make_bars([10, 11]))
    assert len(store.load_bars(conn, "600000")) == 2


def test_last_bar_date(tmp_path):
    conn = store.connect(tmp_path / "t.db")
    assert store.last_bar_date(conn, "600000") is None
    store.upsert_bars(conn, "600000", make_bars([10, 11, 12]))
    assert store.last_bar_date(conn, "600000") == make_bars([10, 11, 12])["date"].iloc[-1].strftime("%Y-%m-%d")


def test_pe_and_financials_roundtrip(tmp_path):
    conn = store.connect(tmp_path / "t.db")
    pe = pd.DataFrame({"date": pd.to_datetime(["2024-01-02", "2024-01-03"]), "pe": [15.0, 15.5]})
    store.upsert_pe(conn, "600000", pe)
    out = store.load_pe(conn, "600000")
    assert out["pe"].tolist() == [15.0, 15.5]

    fin = pd.DataFrame({"report_date": pd.to_datetime(["2023-12-31"]), "roe": [12.0], "revenue_growth": [8.0]})
    store.upsert_financials(conn, "600000", fin)
    out = store.load_financials(conn, "600000")
    assert out["roe"].tolist() == [12.0]


def test_constituents_replace(tmp_path):
    conn = store.connect(tmp_path / "t.db")
    store.save_constituents(conn, pd.DataFrame({"code": ["600000"], "name": ["浦发银行"]}))
    store.save_constituents(conn, pd.DataFrame({"code": ["000001"], "name": ["平安银行"]}))
    out = store.load_constituents(conn)
    assert out["code"].tolist() == ["000001"]
    assert out["name"].tolist() == ["平安银行"]
