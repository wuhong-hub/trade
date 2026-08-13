import pandas as pd
import pytest
from astock.data import store
from astock.recommend.engine import Rec
from astock.tracker import tracker
from tests.conftest import make_bars


def _rec(code, price, horizon="short"):
    return Rec(code, "测试股", horizon, "stub", "理由", price, 10.0, None, "")


def test_record_and_evaluate(tmp_path):
    conn = store.connect(tmp_path / "t.db")
    # 推荐日后股价 100 → 110（+10%，命中）；另一只 100 → 90（未命中）
    store.upsert_bars(conn, "600000", make_bars([100, 110], start="2024-01-01"))
    store.upsert_bars(conn, "000001", make_bars([100, 90], start="2024-01-01"))
    rec_date = make_bars([100])["date"].iloc[0].strftime("%Y-%m-%d")
    tracker.record_recommendations(conn, rec_date,
                                   {"short": [_rec("600000", 100.0, "short")],
                                    "long": [_rec("000001", 100.0, "long")]})
    result = tracker.evaluate(conn)
    assert result["overall"]["n"] == 2
    assert result["overall"]["hit_rate"] == pytest.approx(0.5)
    assert result["overall"]["avg_ret"] == pytest.approx(0.0)
    assert result["by_horizon"]["short"]["hit_rate"] == 1.0
    assert result["by_horizon"]["long"]["hit_rate"] == 0.0
    assert len(result["detail"]) == 2


def test_evaluate_empty(tmp_path):
    conn = store.connect(tmp_path / "t.db")
    result = tracker.evaluate(conn)
    assert result["overall"]["n"] == 0
    assert result["by_horizon"] == {}
