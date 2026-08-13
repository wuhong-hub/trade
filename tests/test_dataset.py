import pandas as pd
from astock.data import dataset, store
from tests.conftest import make_bars


def _seed(conn):
    store.upsert_bars(conn, "600000", make_bars([10, 11, 12], start="2024-01-01"))
    store.upsert_pe(conn, "600000", pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01"]), "pe": [15.0]}))
    store.upsert_financials(conn, "600000", pd.DataFrame({
        "report_date": pd.to_datetime(["2023-12-31"]),
        "roe": [12.0], "revenue_growth": [8.0]}))


def test_build_stock_bars_merges_pe_and_financials(tmp_path):
    conn = store.connect(tmp_path / "t.db")
    _seed(conn)
    df = dataset.build_stock_bars(conn, "600000")
    assert list(df.columns) == ["date", "open", "high", "low", "close",
                                "volume", "pe", "roe", "revenue_growth"]
    assert df["pe"].tolist() == [15.0, 15.0, 15.0]          # asof 前向填充
    assert df["roe"].tolist() == [12.0, 12.0, 12.0]


def test_build_stock_bars_missing_aux_columns_are_nan(tmp_path):
    conn = store.connect(tmp_path / "t.db")
    store.upsert_bars(conn, "600000", make_bars([10, 11, 12]))
    df = dataset.build_stock_bars(conn, "600000")
    assert df["pe"].isna().all()
    assert df["roe"].isna().all()


def test_build_pool_bars_skips_short_history(tmp_path):
    conn = store.connect(tmp_path / "t.db")
    _seed(conn)
    store.upsert_bars(conn, "600000", make_bars([10] * 30))  # 补足到 30 根
    store.upsert_bars(conn, "000001", make_bars([1] * 10))  # 只有 10 根，不足 30
    pool = dataset.build_pool_bars(conn, ["600000", "000001"])
    assert list(pool.keys()) == ["600000"]
