import pandas as pd
import pytest
from astock.data import store
from astock.recommend import engine
from astock.recommend.engine import Rec
from tests.conftest import make_bars


def test_position_pct_rule():
    """等额风险仓位：0.5% × 现价 ÷ ATR14，封顶 15%，ATR 缺失按中性 10%。"""
    assert engine.position_pct(100.0, 10.0) == 5.0     # 0.5×100/10
    assert engine.position_pct(100.0, 2.0) == 15.0     # 封顶
    assert engine.position_pct(100.0, None) == 10.0    # ATR 缺失回退


def test_atr14():
    # 前 39 天横盘（TR=0.2），最后一天 10→101 跳空（TR=92.01）
    atr = engine.atr14(make_bars([10] * 39 + [101]))
    assert atr == pytest.approx((13 * 0.2 + 92.01) / 14)
    assert engine.atr14(make_bars([10] * 10)) is None  # 数据不足


def _state(best_short="stub_short", win_rate=0.6):
    entry = {"strategy": best_short, "score": 1.0,
             "windows": [{"window": "w", "n_trades": 5, "win_rate": win_rate,
                          "annual_return": 0.2, "max_drawdown": -0.1,
                          "sharpe": 1.0}]}
    return {"updated_at": "t", "best": {"short": best_short, "long": "stub_long"},
            "ranking": {"short": [entry],
                        "long": [{"strategy": "stub_long", "score": 1.0,
                                  "windows": entry["windows"]}]}}


def _mk_window(n_trades, win_rate):
    return {"window": "w", "n_trades": n_trades, "win_rate": win_rate,
            "annual_return": 0.2, "max_drawdown": -0.1, "sharpe": 1.0}


def _setup_pool(tmp_path, monkeypatch):
    conn = store.connect(tmp_path / "t.db")
    store.save_constituents(conn, pd.DataFrame(
        {"code": ["600000"], "name": ["浦发银行"]}))
    pool = {"600000": make_bars([10] * 39 + [101])}  # 最后一天触发买入
    monkeypatch.setattr(engine, "STRATEGIES_BY_NAME",
                        {"stub_short": _StubShort(), "stub_long": _StubLong()})
    return conn, pool


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
    assert r.stop_price == round(101.0 * 0.90, 2)   # 短线给数值止损
    assert r.position_pct == pytest.approx(7.5)      # 等额风险：0.5×101/ATR14(≈6.76)
    assert r.reason == "测试理由"
    assert summary["short"]["strategy"] == "stub_short"
    assert summary["short"]["win_rate"] == 0.6
    # 长线 stop_price 为 None
    assert recs["long"][0].stop_price is None


def test_uses_last_window_with_trades(tmp_path, monkeypatch):
    """最近窗口 n_trades=0 时，回退去取最后一个有交易的窗口（与 rank.score 口径一致）。"""
    conn, pool = _setup_pool(tmp_path, monkeypatch)
    state = _state()
    windows = [_mk_window(5, 0.7), _mk_window(0, 0.0)]  # 末窗无交易
    state["ranking"]["short"][0]["windows"] = windows
    state["ranking"]["long"][0]["windows"] = windows
    recs, summary = engine.generate_recommendations(conn, state, pool=pool)
    assert summary["short"]["win_rate"] == 0.7
    assert summary["short"]["n_trades"] == 5
    # 仓位由 ATR 等额风险决定，与胜率无关
    assert recs["short"][0].position_pct == pytest.approx(7.5)


def test_all_windows_no_trades_uses_neutral_win_rate(tmp_path, monkeypatch):
    """全部窗口无交易：胜率按中性 0.5 处理，summary 标注 n_trades=0。"""
    conn, pool = _setup_pool(tmp_path, monkeypatch)
    state = _state()
    windows = [_mk_window(0, 0.0), _mk_window(0, 0.0)]
    state["ranking"]["short"][0]["windows"] = windows
    state["ranking"]["long"][0]["windows"] = windows
    recs, summary = engine.generate_recommendations(conn, state, pool=pool)
    assert summary["short"]["win_rate"] == 0.5
    assert summary["short"]["n_trades"] == 0
    assert recs["short"][0].position_pct == pytest.approx(7.5)   # ATR 等额风险仓位


def test_best_none_outputs_empty(tmp_path, monkeypatch):
    """某方向 best 为 None（所有策略评分无效）：输出空列表，summary 记 strategy=None。"""
    conn, pool = _setup_pool(tmp_path, monkeypatch)
    state = _state()
    state["best"]["short"] = None
    state["ranking"]["short"] = []
    recs, summary = engine.generate_recommendations(conn, state, pool=pool)
    assert recs["short"] == []
    assert summary["short"]["strategy"] is None
    assert len(recs["long"]) == 1   # 另一方向不受影响
