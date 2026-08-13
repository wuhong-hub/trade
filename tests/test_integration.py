import json
from datetime import datetime

import pandas as pd
import pytest
from astock import cli
from astock.data import fetcher
from tests.conftest import make_bars


class _FixedDatetime(datetime):
    """固定"今天"为 mock 行情区间内的日期，使推荐的 rec_date 之后仍有 bars，
    report 才能统计到该推荐。"""

    @classmethod
    def now(cls, tz=None):
        return cls(2025, 11, 15)


@pytest.fixture
def fake_env(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ASTOCK_HOME", str(tmp_path))
    monkeypatch.setattr(fetcher.time, "sleep", lambda s: None)
    monkeypatch.setattr(cli.time, "sleep", lambda s: None)
    # mock 行情最后一根 bar 为 2025-12-01，rec_date 需早于它
    monkeypatch.setattr(cli, "datetime", _FixedDatetime)

    cons = pd.DataFrame({"code": ["600000", "000001"],
                         "name": ["浦发银行", "平安银行"]})
    # 600000：稳定慢涨 + 末端放量新高（触发 momentum 买入）
    up = [10 + 0.05 * i for i in range(500)] + [36.0]
    up_vol = [1000] * 500 + [5000]
    # 000001：横盘，不触发任何信号
    flat = [10.0] * 501
    data = {"600000": (up, up_vol), "000001": (flat, None)}

    monkeypatch.setattr(fetcher, "fetch_index_constituents",
                        lambda index_code="000300": cons.copy())

    def fake_bars(code, start_date, end_date=None):
        closes, vols = data[code]
        return make_bars(closes, volumes=vols, start="2024-01-01")

    monkeypatch.setattr(fetcher, "fetch_daily_bars", fake_bars)
    monkeypatch.setattr(fetcher, "fetch_pe_series", lambda code: pd.DataFrame({
        "date": pd.bdate_range("2024-01-01", periods=501),
        "pe": [15.0] * 500 + [10.0]}))
    monkeypatch.setattr(fetcher, "fetch_financials", lambda code: pd.DataFrame({
        "report_date": pd.to_datetime(["2024-01-01"]),
        "roe": [12.0], "revenue_growth": [8.0]}))
    return capsys


def test_full_flow(fake_env, tmp_path, capsys):
    assert cli.main(["update"]) == 0
    assert "2 成功 / 0 失败" in capsys.readouterr().out

    assert cli.main(["iterate"]) == 0
    out = capsys.readouterr().out
    assert "短线策略排名" in out and "长线策略排名" in out
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["best"]["short"] is not None
    assert len((tmp_path / "history.jsonl").read_text().strip().splitlines()) == 1

    assert cli.main(["recommend"]) == 0
    out = capsys.readouterr().out
    assert "不构成投资建议" in out
    assert "600000" in out  # momentum 策略应命中 600000，推荐清单非空

    # 再 update 一次（模拟次日新数据），report 应有评估结果
    assert cli.main(["update"]) == 0
    capsys.readouterr()
    assert cli.main(["report"]) == 0
    out = capsys.readouterr().out
    assert "历史推荐" in out
    assert "历史推荐 0 条" not in out  # 推荐已落库，报告不应为空
