import json
import pandas as pd
from astock import cli
from astock.data import store
from tests.conftest import make_bars


def test_recommend_without_iterate_fails(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ASTOCK_HOME", str(tmp_path))
    rc = cli.main(["recommend"])
    assert rc == 2
    assert "iterate" in capsys.readouterr().out


def test_stale_data_warning(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ASTOCK_HOME", str(tmp_path))
    conn = store.connect(tmp_path / "astock.db")
    store.save_constituents(conn, pd.DataFrame({"code": ["600000"], "name": ["浦发银行"]}))
    store.upsert_bars(conn, "600000", make_bars([10] * 40, start="2020-01-01"))  # 很旧
    state = {"updated_at": "t",
             "best": {"short": "momentum", "long": "value"},
             "ranking": {"short": [{"strategy": "momentum", "score": 1.0,
                                    "windows": [{"window": "w", "n_trades": 1,
                                                 "win_rate": 0.5, "annual_return": 0.1,
                                                 "max_drawdown": -0.1, "sharpe": 1.0}]}],
                         "long": [{"strategy": "value", "score": 1.0,
                                   "windows": [{"window": "w", "n_trades": 1,
                                                "win_rate": 0.5, "annual_return": 0.1,
                                                "max_drawdown": -0.1, "sharpe": 1.0}]}]}}
    (tmp_path / "state.json").write_text(json.dumps(state))
    rc = cli.main(["recommend"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "update" in out          # 提示数据过旧
    assert "不构成投资建议" in out   # 固定声明
