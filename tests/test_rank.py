import json
import pandas as pd
import pytest
from astock.backtest import rank
from astock.backtest.engine import BacktestResult
from astock.data import store
from tests.conftest import make_bars


def test_rolling_windows_split():
    ws = rank.rolling_windows(end=pd.Timestamp("2026-01-01"), years=2, n=4)
    assert len(ws) == 4
    assert ws[0][0] < ws[0][1] <= ws[1][0]
    assert ws[-1][1] == pd.Timestamp("2026-01-01")


def test_score_mean_sharpe():
    mk = lambda sharpe: BacktestResult("s", "w", 5, 0.5, 0.1, -0.1, sharpe)
    assert rank.score([mk(1.0), mk(3.0)]) == pytest.approx(2.0)
    assert rank.score([]) == float("-inf")


def test_run_iteration_writes_state(tmp_path, monkeypatch):
    conn = store.connect(tmp_path / "t.db")
    store.save_constituents(conn, pd.DataFrame({"code": ["600000"], "name": ["浦发银行"]}))
    # 600 个交易日、结束于今天附近，保证滚动窗口内有数据
    start = (pd.Timestamp.today() - pd.Timedelta(days=900)).strftime("%Y-%m-%d")
    store.upsert_bars(conn, "600000", make_bars([10 + 0.1 * i for i in range(600)],
                                                start=start))
    monkeypatch.setattr(rank, "ALL_STRATEGIES", [_StubShort(), _StubLong()])
    state = rank.run_iteration(conn, tmp_path / "state.json", tmp_path / "history.jsonl")
    assert state["best"]["short"] == "stub_short"
    assert state["best"]["long"] == "stub_long"
    saved = json.loads((tmp_path / "state.json").read_text())
    assert saved["best"] == state["best"]
    assert len(saved["ranking"]["short"][0]["windows"]) == 4
    lines = (tmp_path / "history.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1


class _StubShort:
    class meta:
        name = "stub_short"
        horizon = "short"

    def signals(self, bars):
        s = pd.Series(0, index=bars["date"])
        s.iloc[::40] = 1  # 周期性买入，保证每个窗口有交易
        return s


class _StubLong(_StubShort):
    class meta:
        name = "stub_long"
        horizon = "long"
