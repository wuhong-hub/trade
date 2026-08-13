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


def test_update_years_backfill(tmp_path, monkeypatch, capsys):
    """--years N：强制从 N 年前开始抓取（回填），financials 的 start_year 同步前移。"""
    monkeypatch.setenv("ASTOCK_HOME", str(tmp_path))
    monkeypatch.setattr(cli.time, "sleep", lambda s: None)
    cons = pd.DataFrame({"code": ["600000"], "name": ["浦发银行"]})
    monkeypatch.setattr(cli.fetcher, "fetch_index_constituents",
                        lambda index_code="000300": cons.copy())
    calls = {"bars": [], "fin": []}

    def fake_bars(code, start_date, end_date=None):
        calls["bars"].append(start_date)
        return make_bars([10.0] * 5, start="2024-01-01")

    def fake_fin(code, start_year="2021"):
        calls["fin"].append(start_year)
        return pd.DataFrame({"report_date": pd.to_datetime(["2024-01-01"]),
                             "roe": [12.0], "revenue_growth": [8.0]})

    monkeypatch.setattr(cli.fetcher, "fetch_daily_bars", fake_bars)
    monkeypatch.setattr(cli.fetcher, "fetch_pe_series", lambda code: pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01"]), "pe": [15.0]}))
    monkeypatch.setattr(cli.fetcher, "fetch_financials", fake_fin)

    rc = cli.main(["update", "--years", "10"])
    assert rc == 0
    assert calls["bars"]
    start_year = pd.Timestamp(calls["bars"][0]).year
    assert pd.Timestamp.today().year - start_year >= 9  # 约 10 年前
    assert calls["fin"] == [str((pd.Timestamp.today() - pd.DateOffset(years=10)).year)]


def test_update_constituents_fetch_failure(tmp_path, monkeypatch, capsys):
    """指数成分股名单接口失败：打印指明指数与升级建议，返回非零，不 traceback。"""
    monkeypatch.setenv("ASTOCK_HOME", str(tmp_path))

    def boom(code):
        raise RuntimeError("ConnectionError: read timed out")

    monkeypatch.setattr(cli.fetcher, "fetch_index_constituents", boom)
    rc = cli.main(["update"])
    out = capsys.readouterr().out
    assert rc != 0
    assert "000300" in out          # 指明是哪个指数接口失败
    assert "akshare" in out         # 建议升级 akshare
    assert "Traceback" not in out
