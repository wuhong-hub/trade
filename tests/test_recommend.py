import pandas as pd
import pytest
from astock.data import store
from astock.recommend import engine
from astock.recommend.engine import Rec
from tests.conftest import make_bars


def test_position_pct_rule():
    assert engine.position_pct(0.5) == 10.0
    assert engine.position_pct(0.75) == 15.0   # 封顶
    assert engine.position_pct(0.25) == 5.0    # 封底
    assert engine.position_pct(0.6) == pytest.approx(12.0)


def _state(best_short="stub_short", win_rate=0.6):
    entry = {"strategy": best_short, "score": 1.0,
             "windows": [{"window": "w", "n_trades": 5, "win_rate": win_rate,
                          "annual_return": 0.2, "max_drawdown": -0.1,
                          "sharpe": 1.0}]}
    return {"updated_at": "t", "best": {"short": best_short, "long": "stub_long"},
            "ranking": {"short": [entry],
                        "long": [{"strategy": "stub_long", "score": 1.0,
                                  "windows": entry["windows"]}]}}


class _StubShort:
    class meta:
        name = "stub_short"
        horizon = "short"

    def signals(self, bars):
        s = pd.Series(0, index=bars["date"])
        if bars["close"].iloc[-1] > 100:  # 只在指定股票的最后一天发买入信号
            s.iloc[-1] = 1
        return s

    def reason(self, bars, date):
        return "测试理由"

    def exit_hint(self):
        return "测试离场"


class _StubLong(_StubShort):
    class meta:
        name = "stub_long"
        horizon = "long"


def test_generate_recommendations(tmp_path, monkeypatch):
    conn = store.connect(tmp_path / "t.db")
    store.save_constituents(conn, pd.DataFrame(
        {"code": ["600000", "000001"], "name": ["浦发银行", "平安银行"]}))
    pool = {"600000": make_bars([10] * 39 + [101]),   # 最后一天触发买入
            "000001": make_bars([10] * 40)}           # 不触发
    monkeypatch.setattr(engine, "STRATEGIES_BY_NAME",
                        {"stub_short": _StubShort(), "stub_long": _StubLong()})
    recs, summary = engine.generate_recommendations(conn, _state(), pool=pool)
    assert len(recs["short"]) == 1
    r = recs["short"][0]
    assert isinstance(r, Rec)
    assert r.code == "600000" and r.name == "浦发银行"
    assert r.price == 101.0
    assert r.stop_price == round(101.0 * 0.93, 2)   # 短线给数值止损
    assert r.position_pct == pytest.approx(12.0)     # win_rate=0.6
    assert r.reason == "测试理由"
    assert summary["short"]["strategy"] == "stub_short"
    assert summary["short"]["win_rate"] == 0.6
    # 长线 stop_price 为 None
    assert recs["long"][0].stop_price is None
