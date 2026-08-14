import json
import pandas as pd
from astock import cli
from astock.data import store
from tests.conftest import make_bars


def _index_df(closes, start="2026-01-01"):
    return pd.DataFrame({"date": pd.bdate_range(start, periods=len(closes)),
                         "close": [float(c) for c in closes]})


def _state():
    return {"updated_at": "t",
            "best": {"short": "momentum", "long": "value"},
            "ranking": {"short": [{"strategy": "momentum", "score": 1.0,
                                   "windows": [{"window": "w", "n_trades": 1,
                                                "win_rate": 0.5, "annual_return": 0.1,
                                                "max_drawdown": -0.1, "sharpe": 1.0}]}],
                        "long": [{"strategy": "value", "score": 1.0,
                                  "windows": [{"window": "w", "n_trades": 1,
                                               "win_rate": 0.5, "annual_return": 0.1,
                                               "max_drawdown": -0.1, "sharpe": 1.0}]}]}}


def _setup_recommend(tmp_path, monkeypatch, index_closes=None):
    """600000 慢涨、末日放量新高（触发 momentum 买入信号）；index_closes 为
    None 时 index_daily 为空表（regime 数据不足，不过滤）。"""
    monkeypatch.setenv("ASTOCK_HOME", str(tmp_path))
    conn = store.connect(tmp_path / "astock.db")
    store.save_constituents(conn, pd.DataFrame({"code": ["600000"], "name": ["浦发银行"]}))
    store.upsert_bars(conn, "600000", make_bars([10 + 0.1 * i for i in range(40)],
                                                volumes=[1000.0] * 39 + [5000.0],
                                                start="2026-01-01"))
    if index_closes is not None:
        store.upsert_index_daily(conn, "sh000300", _index_df(index_closes))
    (tmp_path / "state.json").write_text(json.dumps(_state()))


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
    monkeypatch.setattr(cli.fetcher, "fetch_index_daily", lambda symbol="sh000300":
                        pd.DataFrame({"date": pd.bdate_range("2024-01-01", periods=80),
                                      "close": [float(100 + i) for i in range(80)]}))
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


def test_recommend_regime_bear(tmp_path, monkeypatch, capsys):
    """沪深300 位于 60 日均线下方：短线空仓观望，不扫描个股信号。"""
    _setup_recommend(tmp_path, monkeypatch,
                     index_closes=[100 - 0.5 * i for i in range(80)])
    rc = cli.main(["recommend"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "空仓观望" in out
    assert "600000" not in out  # 个股虽有买入信号，熊市直接跳过


def test_recommend_regime_bull(tmp_path, monkeypatch, capsys):
    """沪深300 位于 60 日均线上方：短线正常出推荐。"""
    _setup_recommend(tmp_path, monkeypatch,
                     index_closes=[100 + 0.5 * i for i in range(80)])
    rc = cli.main(["recommend"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "60 日均线上方" in out
    assert "600000" in out


def test_recommend_regime_no_index_data(tmp_path, monkeypatch, capsys):
    """index_daily 为空表：不报错、不过滤，正常出推荐。"""
    _setup_recommend(tmp_path, monkeypatch)
    rc = cli.main(["recommend"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "未启用趋势过滤" in out
    assert "600000" in out


def test_quote_command(monkeypatch, capsys):
    monkeypatch.setattr(cli.fetcher, "fetch_index_spot", lambda symbol: {
        "code": "sh000300", "name": "沪深300", "price": 4668.47,
        "prev_close": 4663.95, "open": None, "high": None, "low": None,
        "volume": 0, "amount": 0, "date": None, "time": None})
    monkeypatch.setattr(cli.fetcher, "fetch_spot_quotes", lambda codes:
                        pd.DataFrame([{"code": "300475", "name": "香农芯创",
                                       "price": 160.90, "prev_close": 157.43,
                                       "open": 161.50, "high": 163.18,
                                       "low": 157.60, "volume": 1, "amount": 1,
                                       "date": "2026-08-14", "time": "14:14:30"}]))
    rc = cli.main(["quote", "300475"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "沪深300" in out and "4668.47" in out
    assert "香农芯创" in out and "160.90" in out and "+2.20%" in out


def test_quote_command_index_only(monkeypatch, capsys):
    monkeypatch.setattr(cli.fetcher, "fetch_index_spot", lambda symbol: {
        "code": "sh000300", "name": "沪深300", "price": 4668.47,
        "prev_close": 4663.95, "open": None, "high": None, "low": None,
        "volume": 0, "amount": 0, "date": None, "time": None})
    rc = cli.main(["quote"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "沪深300" in out
