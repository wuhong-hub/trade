# A 股量化推荐 CLI 工具（astock）实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现一个本地 CLI 工具：自动抓取沪深300+中证500 成分股数据，用经典量化策略库滚动回测优选策略，并按需输出股票推荐与效果跟踪。

**Architecture:** 单一 Python 包 `astock`，分层为 data（akshare 抓取 + SQLite 缓存）、strategies（纯函数策略库）、backtest（向量化自研回测引擎）、recommend（推荐生成）、tracker（推荐效果跟踪）、cli（argparse 入口）。策略是"日线 DataFrame → 信号 Series"的纯函数，回测与推荐复用同一策略代码。

**Tech Stack:** Python 3.10+，akshare（数据源）、pandas（计算）、SQLite3 标准库（存储）、pytest（测试）。

**Spec:** `docs/superpowers/specs/2026-08-12-astock-recommender-design.md`

## Global Constraints

- 费用模型：佣金双边各 0.03%，卖出印花税 0.1%，单次往返成本常数 `COST = 0.0016`。
- 短线策略：信号次日开盘价买入，最多持有 15 个交易日，止损 -7%（`entry * 0.93`）。
  （持有期原为 10 个交易日，2026-08-13 经数据实验调整为 15。）
- 长线策略：信号次日开盘价买入，出现卖出信号后次日开盘卖出。
- 滚动回测：最近 2 年分 4 个半年窗口，按各窗口平均夏普排名。
- 仓位规则：`min(15, max(5, 10 + (win_rate - 0.5) * 20))`，单位 %。
- 数据目录：`~/.astock/`（数据库 `astock.db`、状态 `state.json`、历史 `history.jsonl`），可用环境变量 `ASTOCK_HOME` 覆盖（测试用）。
- 选股范围：沪深300（000300）+ 中证500（000905）成分股，去重。
- 推荐输出固定包含声明："仅为量化策略参考，不构成投资建议"。
- 不做：实盘交易、盘中监控、Web 界面、LLM 接入。
- 日线 bars 的 DataFrame 列约定：`date(datetime64), open, high, low, close, volume`；长线策略用的合并后 bars 额外有 `pe, roe, revenue_growth` 三列。
- 信号 Series 约定：index 为 date，值 1=买入 / -1=卖出 / 0=无。

---

### Task 1: 项目脚手架与测试工具

**Files:**
- Create: `pyproject.toml`
- Create: `astock/__init__.py`
- Create: `astock/data/__init__.py`、`astock/strategies/__init__.py`、`astock/backtest/__init__.py`、`astock/recommend/__init__.py`、`astock/tracker/__init__.py`（均为空文件）
- Create: `tests/__init__.py`（空文件）
- Create: `tests/conftest.py`

**Interfaces:**
- Produces: `tests/conftest.py::make_bars(closes, volumes=None, opens=None, start="2024-01-01") -> pd.DataFrame`，后续所有测试用它造人造 K 线。

- [ ] **Step 1: 创建 pyproject.toml**

```toml
[project]
name = "astock"
version = "0.1.0"
description = "A股量化策略推荐 CLI 工具"
requires-python = ">=3.10"
dependencies = [
    "akshare>=1.14",
    "pandas>=2.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[project.scripts]
astock = "astock.cli:main"

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["astock*"]
```

- [ ] **Step 2: 创建包骨架**

创建上述所有 `__init__.py`（空文件）和目录。

- [ ] **Step 3: 写 tests/conftest.py**

```python
import pandas as pd


def make_bars(closes, volumes=None, opens=None, start="2024-01-01"):
    """按收盘价序列构造人造日线 bars，列：date/open/high/low/close/volume。
    默认 open=close、high=close*1.01、low=close*0.99、volume=1000。"""
    n = len(closes)
    dates = pd.bdate_range(start, periods=n)
    opens = opens if opens is not None else list(closes)
    volumes = volumes if volumes is not None else [1000.0] * n
    return pd.DataFrame({
        "date": dates,
        "open": [float(x) for x in opens],
        "high": [float(c) * 1.01 for c in closes],
        "low": [float(c) * 0.99 for c in closes],
        "close": [float(c) for c in closes],
        "volume": [float(v) for v in volumes],
    })
```

- [ ] **Step 4: 创建虚拟环境并安装依赖**

```bash
python3 --version   # 期望 >= 3.10
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

预期：安装成功。akshare 依赖较多，首次安装可能需要几分钟。

- [ ] **Step 5: 验证测试框架可用**

```bash
echo "def test_smoke(): assert True" > tests/test_smoke.py
.venv/bin/pytest tests/test_smoke.py -v
```

预期：1 passed。随后删除 `tests/test_smoke.py`。

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml astock tests
echo -e ".venv/\n__pycache__/\n*.egg-info/\n.pytest_cache/" > .gitignore
git add .gitignore
git commit -m "chore: 项目脚手架与测试工具"
```

---

### Task 2: data/store.py — SQLite 存储层

**Files:**
- Create: `astock/data/store.py`
- Test: `tests/test_store.py`

**Interfaces:**
- Produces（后续所有任务依赖这些签名）:
  - `connect(db_path) -> sqlite3.Connection`（自动建目录、建表）
  - `upsert_bars(conn, code: str, df: pd.DataFrame) -> None`
  - `load_bars(conn, code: str) -> pd.DataFrame`（列 date/open/high/low/close/volume，按日期升序）
  - `last_bar_date(conn, code: str) -> str | None`（'YYYY-MM-DD' 或 None）
  - `upsert_pe(conn, code: str, df: pd.DataFrame) -> None` / `load_pe(conn, code) -> pd.DataFrame`（列 date/pe）
  - `upsert_financials(conn, code: str, df: pd.DataFrame) -> None` / `load_financials(conn, code) -> pd.DataFrame`（列 report_date/roe/revenue_growth）
  - `save_constituents(conn, df: pd.DataFrame) -> None`（df 列 code/name，全量替换）
  - `load_constituents(conn) -> pd.DataFrame`（列 code/name）

- [ ] **Step 1: 写失败测试 tests/test_store.py**

```python
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
```

- [ ] **Step 2: 跑测试确认失败**

```bash
.venv/bin/pytest tests/test_store.py -v
```

预期：ImportError / AttributeError（store 不存在或函数未定义）。

- [ ] **Step 3: 实现 astock/data/store.py**

```python
import sqlite3
from pathlib import Path

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS daily_bars(
    code TEXT NOT NULL, date TEXT NOT NULL,
    open REAL, high REAL, low REAL, close REAL, volume REAL,
    PRIMARY KEY(code, date));
CREATE TABLE IF NOT EXISTS pe_series(
    code TEXT NOT NULL, date TEXT NOT NULL, pe REAL,
    PRIMARY KEY(code, date));
CREATE TABLE IF NOT EXISTS financials(
    code TEXT NOT NULL, report_date TEXT NOT NULL,
    roe REAL, revenue_growth REAL,
    PRIMARY KEY(code, report_date));
CREATE TABLE IF NOT EXISTS constituents(
    code TEXT PRIMARY KEY, name TEXT, updated_at TEXT);
CREATE TABLE IF NOT EXISTS recommendations(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    rec_date TEXT, code TEXT, name TEXT, horizon TEXT, strategy TEXT,
    reason TEXT, entry_price REAL, position_pct REAL,
    stop_price REAL, exit_hint TEXT);
"""


def connect(db_path):
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    return conn


def upsert_bars(conn, code, df):
    rows = [
        (code, d.strftime("%Y-%m-%d"), float(r.open), float(r.high),
         float(r.low), float(r.close), float(r.volume))
        for d, r in zip(df["date"], df.itertuples())
    ]
    conn.executemany("INSERT OR REPLACE INTO daily_bars VALUES(?,?,?,?,?,?,?)", rows)
    conn.commit()


def load_bars(conn, code):
    df = pd.read_sql(
        "SELECT date, open, high, low, close, volume FROM daily_bars "
        "WHERE code=? ORDER BY date", conn, params=(code,))
    df["date"] = pd.to_datetime(df["date"])
    return df


def last_bar_date(conn, code):
    row = conn.execute(
        "SELECT MAX(date) FROM daily_bars WHERE code=?", (code,)).fetchone()
    return row[0]


def upsert_pe(conn, code, df):
    rows = [(code, d.strftime("%Y-%m-%d"), float(r.pe))
            for d, r in zip(df["date"], df.itertuples())]
    conn.executemany("INSERT OR REPLACE INTO pe_series VALUES(?,?,?)", rows)
    conn.commit()


def load_pe(conn, code):
    df = pd.read_sql(
        "SELECT date, pe FROM pe_series WHERE code=? ORDER BY date",
        conn, params=(code,))
    df["date"] = pd.to_datetime(df["date"])
    return df


def upsert_financials(conn, code, df):
    rows = [(code, d.strftime("%Y-%m-%d"), float(r.roe), float(r.revenue_growth))
            for d, r in zip(df["report_date"], df.itertuples())]
    conn.executemany("INSERT OR REPLACE INTO financials VALUES(?,?,?,?)", rows)
    conn.commit()


def load_financials(conn, code):
    df = pd.read_sql(
        "SELECT report_date, roe, revenue_growth FROM financials "
        "WHERE code=? ORDER BY report_date", conn, params=(code,))
    df["report_date"] = pd.to_datetime(df["report_date"])
    return df


def save_constituents(conn, df):
    conn.execute("DELETE FROM constituents")
    conn.executemany(
        "INSERT INTO constituents(code, name, updated_at) VALUES(?,?,date('now'))",
        [(str(r.code), str(r.name)) for r in df.itertuples()])
    conn.commit()


def load_constituents(conn):
    return pd.read_sql("SELECT code, name FROM constituents ORDER BY code", conn)
```

- [ ] **Step 4: 跑测试确认通过**

```bash
.venv/bin/pytest tests/test_store.py -v
```

预期：5 passed。

- [ ] **Step 5: Commit**

```bash
git add astock/data/store.py tests/test_store.py
git commit -m "feat: SQLite 存储层"
```

---

### Task 3: data/fetcher.py — akshare 数据抓取

**Files:**
- Create: `astock/data/fetcher.py`
- Test: `tests/test_fetcher.py`

**Interfaces:**
- Consumes: 无（直接调 akshare，模块内以 `import akshare as ak` 引用，测试用 monkeypatch 替换 `fetcher.ak`）。
- Produces:
  - `fetch_index_constituents(index_code: str = "000300") -> pd.DataFrame`（列 code/name，code 为 6 位字符串）
  - `fetch_daily_bars(code: str, start_date: str, end_date: str | None = None) -> pd.DataFrame`（列 date/open/high/low/close/volume；无数据返回同列空 DataFrame）
  - `fetch_pe_series(code: str) -> pd.DataFrame`（列 date/pe）
  - `fetch_financials(code: str, start_year: str = "2021") -> pd.DataFrame`（列 report_date/roe/revenue_growth）

- [ ] **Step 1: 写失败测试 tests/test_fetcher.py**（全部 mock akshare，不访问网络）

```python
import pandas as pd
import pytest
from astock.data import fetcher


class FakeAk:
    """模拟 akshare 接口返回的原始（中文列名）数据。"""

    def index_stock_cons_csindex(self, symbol):
        return pd.DataFrame({"成分券代码": ["600000", "1"], "成分券名称": ["浦发银行", "平安银行"]})

    def stock_zh_a_hist(self, symbol, period, start_date, end_date, adjust):
        assert adjust == "qfq"
        return pd.DataFrame({
            "日期": ["2024-01-02", "2024-01-03"],
            "开盘": [10.0, 11.0], "最高": [10.5, 11.5], "最低": [9.5, 10.5],
            "收盘": [10.2, 11.2], "成交量": [10000, 12000],
        })

    def stock_a_indicator_lg(self, symbol):
        return pd.DataFrame({"trade_date": ["2024-01-02"], "pe": [15.0], "total_mv": [1e10]})

    def stock_financial_analysis_indicator(self, symbol, start_year):
        return pd.DataFrame({
            "日期": ["2023-12-31"],
            "净资产收益率(%)": ["12.5"],
            "主营业务收入增长率(%)": ["8.0"],
        })


@pytest.fixture
def fake_ak(monkeypatch):
    monkeypatch.setattr(fetcher, "ak", FakeAk())
    monkeypatch.setattr(fetcher.time, "sleep", lambda s: None)


def test_fetch_index_constituents(fake_ak):
    df = fetcher.fetch_index_constituents("000300")
    assert df["code"].tolist() == ["600000", "000001"]  # 补齐 6 位
    assert df["name"].tolist() == ["浦发银行", "平安银行"]


def test_fetch_daily_bars_columns(fake_ak):
    df = fetcher.fetch_daily_bars("600000", "2024-01-01")
    assert list(df.columns) == ["date", "open", "high", "low", "close", "volume"]
    assert df["close"].tolist() == [10.2, 11.2]
    assert pd.api.types.is_datetime64_any_dtype(df["date"])


def test_fetch_pe_series(fake_ak):
    df = fetcher.fetch_pe_series("600000")
    assert list(df.columns) == ["date", "pe"]
    assert df["pe"].tolist() == [15.0]


def test_fetch_financials(fake_ak):
    df = fetcher.fetch_financials("600000")
    assert list(df.columns) == ["report_date", "roe", "revenue_growth"]
    assert df["roe"].tolist() == [12.5]
    assert df["revenue_growth"].tolist() == [8.0]


def test_retry_on_failure(monkeypatch):
    calls = {"n": 0}

    class FlakyAk:
        def stock_a_indicator_lg(self, symbol):
            calls["n"] += 1
            if calls["n"] < 3:
                raise ConnectionError("boom")
            return pd.DataFrame({"trade_date": ["2024-01-02"], "pe": [15.0]})

    monkeypatch.setattr(fetcher, "ak", FlakyAk())
    monkeypatch.setattr(fetcher.time, "sleep", lambda s: None)
    df = fetcher.fetch_pe_series("600000")
    assert calls["n"] == 3
    assert df["pe"].tolist() == [15.0]


def test_retry_exhausted_raises(monkeypatch):
    class BadAk:
        def stock_a_indicator_lg(self, symbol):
            raise ConnectionError("always fails")

    monkeypatch.setattr(fetcher, "ak", BadAk())
    monkeypatch.setattr(fetcher.time, "sleep", lambda s: None)
    with pytest.raises(ConnectionError):
        fetcher.fetch_pe_series("600000")
```

- [ ] **Step 2: 跑测试确认失败**

```bash
.venv/bin/pytest tests/test_fetcher.py -v
```

预期：ModuleNotFoundError / AttributeError。

- [ ] **Step 3: 实现 astock/data/fetcher.py**

```python
import time

import akshare as ak
import pandas as pd

RETRY = 3


def _retry(fn, **kwargs):
    last = None
    for i in range(RETRY):
        try:
            return fn(**kwargs)
        except Exception as e:  # akshare 底层异常类型不稳定，统一捕获重试
            last = e
            time.sleep(1 + i)
    raise last


def fetch_index_constituents(index_code="000300"):
    df = _retry(ak.index_stock_cons_csindex, symbol=index_code)
    out = df[["成分券代码", "成分券名称"]].rename(
        columns={"成分券代码": "code", "成分券名称": "name"})
    out["code"] = out["code"].astype(str).str.zfill(6)
    return out.reset_index(drop=True)


def fetch_daily_bars(code, start_date, end_date=None):
    start = start_date.replace("-", "")
    end = (end_date or pd.Timestamp.today().strftime("%Y-%m-%d")).replace("-", "")
    df = _retry(ak.stock_zh_a_hist, symbol=code, period="daily",
                start_date=start, end_date=end, adjust="qfq")
    cols = ["date", "open", "high", "low", "close", "volume"]
    if df is None or df.empty:
        return pd.DataFrame(columns=cols)
    df = df.rename(columns={"日期": "date", "开盘": "open", "最高": "high",
                            "最低": "low", "收盘": "close", "成交量": "volume"})
    df["date"] = pd.to_datetime(df["date"])
    return df[cols]


def fetch_pe_series(code):
    df = _retry(ak.stock_a_indicator_lg, symbol=code)
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "pe"])
    df = df.rename(columns={"trade_date": "date"})
    df["date"] = pd.to_datetime(df["date"])
    return df[["date", "pe"]]


def fetch_financials(code, start_year="2021"):
    df = _retry(ak.stock_financial_analysis_indicator,
                symbol=code, start_year=start_year)
    cols = ["report_date", "roe", "revenue_growth"]
    if df is None or df.empty:
        return pd.DataFrame(columns=cols)
    out = pd.DataFrame({
        "report_date": pd.to_datetime(df["日期"]),
        "roe": pd.to_numeric(df["净资产收益率(%)"], errors="coerce"),
        "revenue_growth": pd.to_numeric(df["主营业务收入增长率(%)"], errors="coerce"),
    })
    return out.dropna(subset=["report_date"]).reset_index(drop=True)
```

注意：akshare 接口随网站改版可能变化，报错信息会直接暴露是哪个接口失败，按设计文档提示升级 akshare 即可。

- [ ] **Step 4: 跑测试确认通过**

```bash
.venv/bin/pytest tests/test_fetcher.py -v
```

预期：6 passed。

- [ ] **Step 5: Commit**

```bash
git add astock/data/fetcher.py tests/test_fetcher.py
git commit -m "feat: akshare 数据抓取层（带重试）"
```

---

### Task 4: strategies/base.py + data/dataset.py — 策略接口与数据集合并

**Files:**
- Create: `astock/strategies/base.py`
- Create: `astock/data/dataset.py`
- Test: `tests/test_dataset.py`

**Interfaces:**
- Produces:
  - `StrategyMeta` dataclass：字段 `name: str, horizon: str, description: str`（horizon 为 `"short"` 或 `"long"`）
  - `Strategy` 基类：类属性 `meta: StrategyMeta`；方法
    - `signals(self, bars: pd.DataFrame) -> pd.Series`（index=date，值 1/-1/0）
    - `reason(self, bars: pd.DataFrame, date) -> str`（默认返回 `meta.description`）
    - `exit_hint(self) -> str`（默认 `""`）
  - `dataset.build_stock_bars(conn, code: str) -> pd.DataFrame`：日线 + asof 合并 pe + asof 合并财务（roe/revenue_growth），无数据列填 NaN
  - `dataset.build_pool_bars(conn, codes: list[str]) -> dict[str, pd.DataFrame]`：跳过不足 30 根日线的股票

长线策略约定：`bars` 中已含 `pe`、`roe`、`revenue_growth` 列（由 dataset 合并），短线策略忽略这些列。这样所有策略的 `signals` 签名统一为只收 `bars`。

- [ ] **Step 1: 写失败测试 tests/test_dataset.py**

```python
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
    store.upsert_bars(conn, "000001", make_bars([1] * 10))  # 只有 10 根，不足 30
    pool = dataset.build_pool_bars(conn, ["600000", "000001"])
    assert list(pool.keys()) == ["600000"]
```

- [ ] **Step 2: 跑测试确认失败**

```bash
.venv/bin/pytest tests/test_dataset.py -v
```

- [ ] **Step 3: 实现 astock/strategies/base.py**

```python
from dataclasses import dataclass

import pandas as pd


@dataclass
class StrategyMeta:
    name: str
    horizon: str  # "short" 或 "long"
    description: str


class Strategy:
    """策略基类：纯函数，不碰网络与磁盘。

    signals() 输入单只股票的 bars（短线只需 OHLCV，长线含 pe/roe/revenue_growth），
    输出 index=date、值 1/-1/0 的信号 Series。
    """

    meta: StrategyMeta

    def signals(self, bars: pd.DataFrame) -> pd.Series:
        raise NotImplementedError

    def reason(self, bars: pd.DataFrame, date) -> str:
        return self.meta.description

    def exit_hint(self) -> str:
        return ""
```

- [ ] **Step 4: 实现 astock/data/dataset.py**

```python
import pandas as pd

from . import store

MIN_BARS = 30


def build_stock_bars(conn, code):
    bars = store.load_bars(conn, code).sort_values("date").reset_index(drop=True)
    pe = store.load_pe(conn, code)
    fin = store.load_financials(conn, code)
    if len(pe):
        bars = pd.merge_asof(bars, pe.sort_values("date"), on="date")
    else:
        bars["pe"] = float("nan")
    if len(fin):
        fin = fin.sort_values("report_date").rename(columns={"report_date": "date"})
        bars = pd.merge_asof(bars, fin, on="date")
    else:
        bars["roe"] = float("nan")
        bars["revenue_growth"] = float("nan")
    return bars


def build_pool_bars(conn, codes):
    pool = {}
    for code in codes:
        bars = build_stock_bars(conn, code)
        if len(bars) >= MIN_BARS:
            pool[code] = bars
    return pool
```

- [ ] **Step 5: 跑测试确认通过**

```bash
.venv/bin/pytest tests/test_dataset.py -v
```

预期：3 passed。

- [ ] **Step 6: Commit**

```bash
git add astock/strategies/base.py astock/data/dataset.py tests/test_dataset.py
git commit -m "feat: 策略基类与数据集合并层"
```

---

### Task 5: strategies/ma_trend.py — 双均线趋势策略

**Files:**
- Create: `astock/strategies/ma_trend.py`
- Test: `tests/test_ma_trend.py`

**Interfaces:**
- Consumes: `strategies.base.Strategy / StrategyMeta`；bars DataFrame（Task 1 的列约定）。
- Produces: `MATrendStrategy(fast=5, slow=20)`，`meta.name == "ma_trend"`，`horizon == "short"`。

- [ ] **Step 1: 写失败测试 tests/test_ma_trend.py**

```python
from astock.strategies.ma_trend import MATrendStrategy
from tests.conftest import make_bars


def test_golden_cross_buy_signal():
    # 前 20 天收盘 10，第 21-25 天收盘 8，第 26-30 天收盘 12。
    # 手算：5 日均线在第 28 根（index 27）上穿 20 日均线（10.4 > 9.8），
    # 此前 fast<=slow，金叉恰好在 index 27 出现一次，全程无死叉。
    closes = [10] * 20 + [8] * 5 + [12] * 5
    bars = make_bars(closes)
    sig = MATrendStrategy().signals(bars)
    assert sig.iloc[27] == 1
    assert (sig == 1).sum() == 1
    assert (sig == -1).sum() == 0


def test_death_cross_sell_signal():
    # 与上面镜像：先低位 8，再拉高到 12（产生金叉），再跌回 8（产生死叉）。
    closes = [8] * 20 + [12] * 10 + [8] * 10
    bars = make_bars(closes)
    sig = MATrendStrategy().signals(bars)
    assert (sig == 1).sum() == 1
    assert (sig == -1).sum() == 1
    assert sig[sig == 1].index[0] < sig[sig == -1].index[0]
```

- [ ] **Step 2: 跑测试确认失败**

```bash
.venv/bin/pytest tests/test_ma_trend.py -v
```

- [ ] **Step 3: 实现 astock/strategies/ma_trend.py**

```python
import pandas as pd

from .base import Strategy, StrategyMeta


class MATrendStrategy(Strategy):
    meta = StrategyMeta(
        name="ma_trend", horizon="short",
        description="5日均线上穿20日均线（金叉）买入，下穿（死叉）卖出")

    def __init__(self, fast=5, slow=20):
        self.fast = fast
        self.slow = slow

    def signals(self, bars):
        df = bars.set_index("date")
        fast_ma = df["close"].rolling(self.fast).mean()
        slow_ma = df["close"].rolling(self.slow).mean()
        cross_up = ((fast_ma > slow_ma)
                    & (fast_ma.shift(1) <= slow_ma.shift(1))).fillna(False)
        cross_dn = ((fast_ma < slow_ma)
                    & (fast_ma.shift(1) >= slow_ma.shift(1))).fillna(False)
        sig = pd.Series(0, index=df.index)
        sig[cross_up] = 1
        sig[cross_dn] = -1
        return sig

    def exit_hint(self):
        return "死叉（5日线下穿20日线）卖出"
```

- [ ] **Step 4: 跑测试确认通过**

```bash
.venv/bin/pytest tests/test_ma_trend.py -v
```

预期：2 passed。

- [ ] **Step 5: Commit**

```bash
git add astock/strategies/ma_trend.py tests/test_ma_trend.py
git commit -m "feat: 双均线趋势策略"
```

---

### Task 6: strategies/momentum.py — 动量突破策略

**Files:**
- Create: `astock/strategies/momentum.py`
- Test: `tests/test_momentum.py`

**Interfaces:**
- Consumes: `strategies.base.Strategy / StrategyMeta`。
- Produces: `MomentumStrategy()`，`meta.name == "momentum"`，`horizon == "short"`。

- [ ] **Step 1: 写失败测试 tests/test_momentum.py**

```python
from astock.strategies.momentum import MomentumStrategy
from tests.conftest import make_bars


def test_breakout_with_volume_buy():
    # 前 24 天收盘恒为 10、量恒为 1000；第 25 根（index 24）收盘 12 创 20 日新高，
    # 成交量 3000 > 1.5 * 20 日均量 1000 → 买入信号恰好在 index 24。
    closes = [10] * 24 + [12]
    volumes = [1000] * 24 + [3000]
    bars = make_bars(closes, volumes=volumes)
    sig = MomentumStrategy().signals(bars)
    assert sig.iloc[24] == 1
    assert (sig == 1).sum() == 1


def test_breakout_without_volume_no_buy():
    # 同样创新高但量没放大 → 无买入信号。
    closes = [10] * 24 + [12]
    bars = make_bars(closes)  # volume 恒 1000
    sig = MomentumStrategy().signals(bars)
    assert (sig == 1).sum() == 0


def test_sell_below_prior_10day_low():
    # 先制造买入，随后收盘跌破前 10 日最低价 → 卖出信号。
    closes = [10] * 24 + [12, 12, 8.0]
    volumes = [1000] * 24 + [3000, 1000, 1000]
    bars = make_bars(closes, volumes=volumes)
    sig = MomentumStrategy().signals(bars)
    assert (sig == 1).sum() == 1
    assert (sig == -1).sum() == 1
    assert sig[sig == -1].index[0] > sig[sig == 1].index[0]
```

- [ ] **Step 2: 跑测试确认失败**

```bash
.venv/bin/pytest tests/test_momentum.py -v
```

- [ ] **Step 3: 实现 astock/strategies/momentum.py**

```python
import pandas as pd

from .base import Strategy, StrategyMeta


class MomentumStrategy(Strategy):
    meta = StrategyMeta(
        name="momentum", horizon="short",
        description="收盘价创20日新高且成交量放大至20日均量1.5倍以上买入，跌破前10日最低价卖出")

    def signals(self, bars):
        df = bars.set_index("date")
        high20 = df["close"].rolling(20).max()
        vol_ma = df["volume"].rolling(20).mean()
        low10_prior = df["low"].rolling(10).min().shift(1)
        buy = ((df["close"] >= high20)
               & (df["volume"] > 1.5 * vol_ma)).fillna(False)
        sell = (df["close"] < low10_prior).fillna(False)
        sig = pd.Series(0, index=df.index)
        sig[buy] = 1
        sig[sell] = -1
        return sig

    def exit_hint(self):
        return "跌破前10日最低价卖出"
```

- [ ] **Step 4: 跑测试确认通过**

```bash
.venv/bin/pytest tests/test_momentum.py -v
```

预期：3 passed。

- [ ] **Step 5: Commit**

```bash
git add astock/strategies/momentum.py tests/test_momentum.py
git commit -m "feat: 动量突破策略"
```

---

### Task 7: strategies/volume_price.py — 量价配合策略

**Files:**
- Create: `astock/strategies/volume_price.py`
- Test: `tests/test_volume_price.py`

**Interfaces:**
- Consumes: `strategies.base.Strategy / StrategyMeta`。
- Produces: `VolumePriceStrategy()`，`meta.name == "volume_price"`，`horizon == "short"`。

- [ ] **Step 1: 写失败测试 tests/test_volume_price.py**

```python
from astock.strategies.volume_price import VolumePriceStrategy
from tests.conftest import make_bars


def test_rebound_after_quiet_pullback_buy():
    # 前 25 天缓涨（close=10+0.1*i），量恒 1000；
    # 第 26-28 根（index 25-27）三连阴且缩量（900/800/700）；
    # 第 29 根（index 28）低开高走收阳，量 3000 > 2 * 20 日均量(970)=1940 → 买入。
    closes = [10 + 0.1 * i for i in range(25)] + [12.0, 11.5, 11.0, 11.5]
    opens = list(closes)
    opens[28] = 10.5  # 反弹日开盘低于收盘 → 阳线
    volumes = [1000] * 25 + [900, 800, 700, 3000]
    bars = make_bars(closes, volumes=volumes, opens=opens)
    sig = VolumePriceStrategy().signals(bars)
    assert sig.iloc[28] == 1


def test_no_pullback_no_buy():
    # 没有三连阴缩量回调，单独一根放量阳线不触发买入。
    closes = [10 + 0.1 * i for i in range(28)] + [13.0]
    opens = list(closes)
    opens[28] = 12.0
    volumes = [1000] * 28 + [3000]
    bars = make_bars(closes, volumes=volumes, opens=opens)
    sig = VolumePriceStrategy().signals(bars)
    assert (sig == 1).sum() == 0
```

- [ ] **Step 2: 跑测试确认失败**

```bash
.venv/bin/pytest tests/test_volume_price.py -v
```

- [ ] **Step 3: 实现 astock/strategies/volume_price.py**

```python
import pandas as pd

from .base import Strategy, StrategyMeta


class VolumePriceStrategy(Strategy):
    meta = StrategyMeta(
        name="volume_price", horizon="short",
        description="缩量三连阴回调后，放量（>2倍20日均量）阳线反弹买入，跌破10日均线卖出")

    def signals(self, bars):
        df = bars.set_index("date")
        down = df["close"] < df["close"].shift(1)
        vol_down = df["volume"] < df["volume"].shift(1)
        pullback = (down & down.shift(1) & down.shift(2)
                    & vol_down & vol_down.shift(1)).fillna(False)
        vol_ma = df["volume"].rolling(20).mean()
        rebound = ((df["close"] > df["open"])
                   & (df["volume"] > 2 * vol_ma)).fillna(False)
        buy = pullback.shift(1).fillna(False) & rebound
        ma10 = df["close"].rolling(10).mean()
        sell = ((df["close"] < ma10)
                & (df["close"].shift(1) >= ma10.shift(1))).fillna(False)
        sig = pd.Series(0, index=df.index)
        sig[buy] = 1
        sig[sell] = -1
        return sig

    def exit_hint(self):
        return "跌破10日均线卖出"
```

- [ ] **Step 4: 跑测试确认通过**

```bash
.venv/bin/pytest tests/test_volume_price.py -v
```

预期：2 passed。

- [ ] **Step 5: Commit**

```bash
git add astock/strategies/volume_price.py tests/test_volume_price.py
git commit -m "feat: 量价配合策略"
```

---

### Task 8: strategies/value.py — 价值筛选策略 + 策略注册表

**Files:**
- Create: `astock/strategies/value.py`
- Modify: `astock/strategies/__init__.py`
- Test: `tests/test_value.py`

**Interfaces:**
- Consumes: `strategies.base.Strategy / StrategyMeta`；bars 含 `pe/roe/revenue_growth` 列（Task 4 约定）。
- Produces:
  - `ValueStrategy()`，`meta.name == "value"`，`horizon == "long"`
  - `astock.strategies.ALL_STRATEGIES: list[Strategy]`（4 个策略实例，供 Task 10/11 使用）

参数（实现常量，写入 docstring）：PE 分位窗口 756 个交易日（约 3 年），min_periods=60；买入阈值 q30，卖出阈值 q70；ROE 买入线 10%、卖出线 8%；营收增长 > 0。

- [ ] **Step 1: 写失败测试 tests/test_value.py**

```python
import pandas as pd
from astock.strategies.value import ValueStrategy
from tests.conftest import make_bars


def _bars_with_pe(pes, roe=12.0, growth=5.0):
    bars = make_bars([10.0] * len(pes))
    bars["pe"] = pes
    bars["roe"] = roe
    bars["revenue_growth"] = growth
    return bars


def test_buy_when_pe_below_own_q30():
    # 前 70 天 PE=20，第 71 天（index 70）起 PE=10。
    # index 70 时滚动窗口（61 个值：60 个 20 + 1 个 10，min_periods=60）的
    # q30 = 20 > 10 → 低估条件成立且为首次成立（边沿），产生买入信号。
    pes = [20.0] * 70 + [10.0] * 10
    sig = ValueStrategy().signals(_bars_with_pe(pes))
    assert sig.iloc[70] == 1
    assert (sig == 1).sum() == 1  # 条件持续但只在边沿触发一次


def test_no_buy_when_roe_low():
    pes = [20.0] * 70 + [10.0] * 10
    sig = ValueStrategy().signals(_bars_with_pe(pes, roe=5.0))
    assert (sig == 1).sum() == 0


def test_sell_when_pe_above_q70():
    # 前 70 天 PE=10，之后 PE=25 超过窗口 q70 → 卖出信号边沿。
    pes = [10.0] * 70 + [25.0] * 10
    sig = ValueStrategy().signals(_bars_with_pe(pes))
    assert (sig == -1).sum() >= 1
    assert sig[sig == -1].index[0] == pd.bdate_range("2024-01-01", periods=80)[70]
```

- [ ] **Step 2: 跑测试确认失败**

```bash
.venv/bin/pytest tests/test_value.py -v
```

- [ ] **Step 3: 实现 astock/strategies/value.py**

```python
import pandas as pd

from .base import Strategy, StrategyMeta

PE_WINDOW = 756      # 约 3 年交易日
MIN_PERIODS = 60
BUY_Q, SELL_Q = 0.3, 0.7
ROE_BUY, ROE_SELL = 10.0, 8.0


class ValueStrategy(Strategy):
    meta = StrategyMeta(
        name="value", horizon="long",
        description="PE低于自身3年30%分位且ROE>10%、营收增长>0时买入；"
                    "PE高于自身70%分位或ROE<8%时卖出")

    def signals(self, bars):
        df = bars.set_index("date")
        pe_q30 = df["pe"].rolling(PE_WINDOW, min_periods=MIN_PERIODS).quantile(BUY_Q)
        pe_q70 = df["pe"].rolling(PE_WINDOW, min_periods=MIN_PERIODS).quantile(SELL_Q)
        cheap = ((df["pe"] < pe_q30)
                 & (df["roe"] > ROE_BUY)
                 & (df["revenue_growth"] > 0)).fillna(False)
        expensive = ((df["pe"] > pe_q70) | (df["roe"] < ROE_SELL)).fillna(False)
        buy = cheap & ~cheap.shift(1).fillna(False)          # 边沿触发
        sell = expensive & ~expensive.shift(1).fillna(False)
        sig = pd.Series(0, index=df.index)
        sig[buy] = 1
        sig[sell] = -1
        return sig

    def exit_hint(self):
        return "PE回升至自身70%分位以上或ROE恶化时卖出"
```

- [ ] **Step 4: 写策略注册表 astock/strategies/__init__.py**

```python
from .ma_trend import MATrendStrategy
from .momentum import MomentumStrategy
from .value import ValueStrategy
from .volume_price import VolumePriceStrategy

ALL_STRATEGIES = [
    MomentumStrategy(),
    MATrendStrategy(),
    VolumePriceStrategy(),
    ValueStrategy(),
]

STRATEGIES_BY_NAME = {s.meta.name: s for s in ALL_STRATEGIES}
```

- [ ] **Step 5: 跑测试确认通过**

```bash
.venv/bin/pytest tests/test_value.py -v
.venv/bin/python -c "from astock.strategies import ALL_STRATEGIES, STRATEGIES_BY_NAME; print([s.meta.name for s in ALL_STRATEGIES])"
```

预期：3 passed；输出 `['momentum', 'ma_trend', 'volume_price', 'value']`。

- [ ] **Step 6: Commit**

```bash
git add astock/strategies tests/test_value.py
git commit -m "feat: 价值筛选策略与策略注册表"
```

---

### Task 9: backtest/engine.py — 回测引擎

**Files:**
- Create: `astock/backtest/engine.py`
- Test: `tests/test_backtest_engine.py`

**Interfaces:**
- Consumes: `Strategy`（Task 4），pool_bars（Task 4 `build_pool_bars` 的产物）。
- Produces:
  - 常量 `COMMISSION = 0.0003`、`STAMP_TAX = 0.001`、`COST = 0.0016`
  - `Trade` dataclass：`code, entry_date, entry_price, exit_date, exit_price, gross_ret, net_ret, holding_days`
  - `BacktestResult` dataclass：`strategy, window, n_trades, win_rate, annual_return, max_drawdown, sharpe, trades`
  - `simulate_trades(code: str, bars: pd.DataFrame, signals: pd.Series, horizon: str) -> list[Trade]`
  - `equity_curve(trades: list[Trade], bars_by_code: dict) -> pd.Series`（日频组合收益率；日内止损用收盘价近似，属已知简化）
  - `compute_metrics(strategy: str, window: str, trades: list[Trade], bars_by_code: dict) -> BacktestResult`
  - `run_backtest(strategy: Strategy, pool_bars: dict, start=None, end=None) -> BacktestResult`

回测规则（Global Constraints 的落实）：买入=信号次日开盘价；short：持有满 10 个交易日按当日收盘卖、`low <= entry*0.93` 按 `entry*0.93` 止损卖、卖出信号次日开盘卖；long：仅卖出信号次日开盘卖；数据末尾仍持仓按最后收盘强制平仓。

- [ ] **Step 1: 写失败测试 tests/test_backtest_engine.py**

```python
import pandas as pd
import pytest
from astock.backtest import engine
from tests.conftest import make_bars


def _sig(bars, buy_i):
    s = pd.Series(0, index=bars["date"])
    s.iloc[buy_i] = 1
    return s


def test_short_max_hold_exit():
    # 每天 open=close=10+i（严格上涨）。index 0 买入信号 → index 1 开盘 11 买入；
    # 持有满 10 天（index 11）按收盘 21 卖出。
    bars = make_bars([10 + i for i in range(15)])
    trades = engine.simulate_trades("X", bars, _sig(bars, 0), "short")
    assert len(trades) == 1
    t = trades[0]
    assert t.entry_price == 11.0
    assert t.exit_price == 21.0
    assert t.holding_days == 10
    assert t.gross_ret == pytest.approx(21 / 11 - 1)
    assert t.net_ret == pytest.approx(21 / 11 - 1 - engine.COST)


def test_short_stop_loss():
    # 买入后次日 low 击穿 entry*0.93 → 按 entry*0.93 止损。
    bars = make_bars([100, 100, 90, 90, 90], opens=[100, 100, 90, 90, 90])
    bars.loc[2, "low"] = 92.0  # entry=100，93 止损线被击穿
    trades = engine.simulate_trades("X", bars, _sig(bars, 0), "short")
    assert len(trades) == 1
    assert trades[0].exit_price == 93.0
    assert trades[0].net_ret == pytest.approx(-0.07 - engine.COST)


def test_long_exits_only_on_sell_signal():
    bars = make_bars([10 + i for i in range(15)])
    sig = _sig(bars, 0)
    sig.iloc[5] = -1  # index 5 卖出信号 → index 6 开盘卖出
    trades = engine.simulate_trades("X", bars, sig, "long")
    assert len(trades) == 1
    assert trades[0].entry_price == 11.0
    assert trades[0].exit_price == 16.0  # open[6] = 10+6


def test_open_position_force_closed_at_end():
    bars = make_bars([10 + i for i in range(5)])
    trades = engine.simulate_trades("X", bars, _sig(bars, 2), "long")
    assert len(trades) == 1
    assert trades[0].exit_date == bars["date"].iloc[-1]


def test_equity_curve_and_metrics():
    # 单笔交易：entry_price=100，期间收盘 100→110→121。
    bars = make_bars([100, 110, 121])
    t = engine.Trade("X", bars["date"].iloc[0], 100.0,
                     bars["date"].iloc[2], 121.0, 0.21, 0.21 - engine.COST, 2)
    eq = engine.equity_curve([t], {"X": bars})
    assert len(eq) == 3
    assert eq.iloc[1] == pytest.approx(0.1)
    r = engine.compute_metrics("s", "w", [t], {"X": bars})
    assert r.n_trades == 1
    assert r.win_rate == 1.0
    cum = (1 + eq).cumprod()
    assert r.max_drawdown == pytest.approx((cum / cum.cummax() - 1).min())


def test_compute_metrics_empty():
    r = engine.compute_metrics("s", "w", [], {})
    assert r.n_trades == 0 and r.win_rate == 0.0 and r.sharpe == 0.0


def test_run_backtest_window_filter():
    # 两个窗口数据：只在窗口内产生交易。
    bars = make_bars([10 + i for i in range(40)])
    pool = {"X": bars}

    class BuyAtStart:
        class meta:
            name = "stub"
            horizon = "short"

        def signals(self, b):
            s = pd.Series(0, index=b["date"])
            s.iloc[0] = 1
            return s

    r = engine.run_backtest(BuyAtStart(), pool,
                            start=bars["date"].iloc[0], end=bars["date"].iloc[-1])
    assert r.n_trades >= 1
    assert r.strategy == "stub"
```

- [ ] **Step 2: 跑测试确认失败**

```bash
.venv/bin/pytest tests/test_backtest_engine.py -v
```

- [ ] **Step 3: 实现 astock/backtest/engine.py**

```python
from dataclasses import dataclass, field

import pandas as pd

COMMISSION = 0.0003   # 单边佣金
STAMP_TAX = 0.001     # 卖出印花税
COST = COMMISSION * 2 + STAMP_TAX  # 单次往返成本 0.0016

SHORT_MAX_HOLD = 10
SHORT_STOP = 0.93     # -7% 止损


@dataclass
class Trade:
    code: str
    entry_date: object
    entry_price: float
    exit_date: object
    exit_price: float
    gross_ret: float
    net_ret: float
    holding_days: int


@dataclass
class BacktestResult:
    strategy: str
    window: str
    n_trades: int
    win_rate: float
    annual_return: float
    max_drawdown: float
    sharpe: float
    trades: list = field(default_factory=list)


def simulate_trades(code, bars, signals, horizon):
    df = bars.reset_index(drop=True)
    sigs = df["date"].map(signals).fillna(0)
    max_hold = SHORT_MAX_HOLD if horizon == "short" else None
    stop = SHORT_STOP if horizon == "short" else None
    trades = []
    in_pos = False
    entry_i = entry_price = None
    for i in range(len(df)):
        if not in_pos:
            if sigs.iloc[i] == 1 and i + 1 < len(df):
                in_pos = True
                entry_i = i + 1
                entry_price = float(df["open"].iloc[i + 1])
            continue
        exit_i = exit_price = None
        if stop is not None and df["low"].iloc[i] <= entry_price * stop:
            exit_i, exit_price = i, entry_price * stop
        elif sigs.iloc[i] == -1 and i + 1 < len(df):
            exit_i, exit_price = i + 1, float(df["open"].iloc[i + 1])
        elif max_hold is not None and i - entry_i >= max_hold:
            exit_i, exit_price = i, float(df["close"].iloc[i])
        elif i == len(df) - 1:
            exit_i, exit_price = i, float(df["close"].iloc[i])
        if exit_i is not None and exit_i > entry_i:
            gross = exit_price / entry_price - 1
            trades.append(Trade(code, df["date"].iloc[entry_i], entry_price,
                                df["date"].iloc[exit_i], exit_price,
                                gross, gross - COST, exit_i - entry_i))
            in_pos = False
    return trades


def equity_curve(trades, bars_by_code):
    """组合日收益率：各持仓个股日收益（收盘价近似）的等权平均。"""
    daily = {}
    for t in trades:
        closes = bars_by_code[t.code].set_index("date")["close"]
        window = closes.loc[t.entry_date:t.exit_date]
        if len(window) == 0:
            continue
        rets = window.pct_change()
        rets.iloc[0] = window.iloc[0] / t.entry_price - 1
        for d, r in rets.dropna().items():
            daily.setdefault(d, []).append(r)
    if not daily:
        return pd.Series(dtype=float)
    return pd.Series({d: sum(v) / len(v) for d, v in daily.items()}).sort_index()


def compute_metrics(strategy, window, trades, bars_by_code):
    if not trades:
        return BacktestResult(strategy, window, 0, 0.0, 0.0, 0.0, 0.0, [])
    wins = sum(1 for t in trades if t.net_ret > 0)
    eq = equity_curve(trades, bars_by_code)
    cum = (1 + eq).cumprod()
    total = float(cum.iloc[-1] - 1) if len(cum) else 0.0
    days = len(eq)
    annual = (1 + total) ** (252 / days) - 1 if days else 0.0
    dd = float((cum / cum.cummax() - 1).min()) if len(cum) else 0.0
    std = eq.std()
    sharpe = float(eq.mean() / std * (252 ** 0.5)) if std and std > 0 else 0.0
    return BacktestResult(strategy, window, len(trades), wins / len(trades),
                          annual, dd, sharpe, trades)


def run_backtest(strategy, pool_bars, start=None, end=None):
    trades_all = []
    for code, bars in pool_bars.items():
        b = bars
        if start is not None:
            b = b[b["date"] >= start]
        if end is not None:
            b = b[b["date"] <= end]
        if len(b) < 30:
            continue
        sigs = strategy.signals(b.reset_index(drop=True))
        trades_all.extend(
            simulate_trades(code, b.reset_index(drop=True), sigs,
                            strategy.meta.horizon))
    label = f"{getattr(start, 'date', lambda: start)()}~{getattr(end, 'date', lambda: end)()}"
    return compute_metrics(strategy.meta.name, label, trades_all, pool_bars)
```

- [ ] **Step 4: 跑测试确认通过**

```bash
.venv/bin/pytest tests/test_backtest_engine.py -v
```

预期：7 passed。

- [ ] **Step 5: Commit**

```bash
git add astock/backtest/engine.py tests/test_backtest_engine.py
git commit -m "feat: 回测引擎（模拟交易+绩效指标）"
```

---

### Task 10: backtest/rank.py — 滚动回测排名与迭代状态

**Files:**
- Create: `astock/backtest/rank.py`
- Test: `tests/test_rank.py`

**Interfaces:**
- Consumes: `store.load_constituents`、`dataset.build_pool_bars`、`engine.run_backtest`、`strategies.ALL_STRATEGIES`。
- Produces:
  - `rolling_windows(end=None, years=2, n=4) -> list[tuple[Timestamp, Timestamp]]`
  - `score(results: list[BacktestResult]) -> float`（有交易窗口的平均夏普；全无交易返回 `-inf`）
  - `run_iteration(conn, state_path, history_path) -> dict`：写 state.json、追加 history.jsonl，返回 state dict
  - state dict 结构：`{"updated_at": str, "best": {"short": name|None, "long": name|None}, "ranking": {"short": [entry...], "long": [entry...]}}`；entry = `{"strategy": str, "score": float, "windows": [{"window","n_trades","win_rate","annual_return","max_drawdown","sharpe"}]}`

- [ ] **Step 1: 写失败测试 tests/test_rank.py**

```python
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
```

- [ ] **Step 2: 跑测试确认失败**

```bash
.venv/bin/pytest tests/test_rank.py -v
```

- [ ] **Step 3: 实现 astock/backtest/rank.py**

```python
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from ..data.dataset import build_pool_bars
from ..data.store import load_constituents
from ..strategies import ALL_STRATEGIES
from .engine import run_backtest


def rolling_windows(end=None, years=2, n=4):
    end = pd.Timestamp(end) if end is not None else pd.Timestamp.today().normalize()
    start = end - pd.DateOffset(years=years)
    bounds = pd.date_range(start, end, periods=n + 1)
    return [(bounds[i], bounds[i + 1]) for i in range(n)]


def score(results):
    valid = [r for r in results if r.n_trades > 0]
    if not valid:
        return float("-inf")
    return sum(r.sharpe for r in valid) / len(valid)


def run_iteration(conn, state_path, history_path):
    cons = load_constituents(conn)
    pool = build_pool_bars(conn, cons["code"].tolist())
    windows = rolling_windows()
    ranking = {"short": [], "long": []}
    for s in ALL_STRATEGIES:
        results = [run_backtest(s, pool, w[0], w[1]) for w in windows]
        ranking[s.meta.horizon].append({
            "strategy": s.meta.name,
            "score": score(results),
            "windows": [{
                "window": r.window, "n_trades": r.n_trades,
                "win_rate": r.win_rate, "annual_return": r.annual_return,
                "max_drawdown": r.max_drawdown, "sharpe": r.sharpe,
            } for r in results],
        })
    for h in ranking:
        ranking[h].sort(key=lambda e: e["score"], reverse=True)
    state = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "best": {h: (ranking[h][0]["strategy"] if ranking[h] else None)
                 for h in ranking},
        "ranking": ranking,
    }
    state_path = Path(state_path)
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2))
    with open(history_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(state, ensure_ascii=False) + "\n")
    return state
```

- [ ] **Step 4: 跑测试确认通过**

```bash
.venv/bin/pytest tests/test_rank.py -v
```

预期：3 passed。

- [ ] **Step 5: Commit**

```bash
git add astock/backtest/rank.py tests/test_rank.py
git commit -m "feat: 滚动回测排名与迭代状态"
```

---

### Task 11: recommend/engine.py — 推荐生成

**Files:**
- Create: `astock/recommend/engine.py`
- Test: `tests/test_recommend.py`

**Interfaces:**
- Consumes: `store.load_constituents`、`dataset.build_pool_bars`、`strategies.STRATEGIES_BY_NAME`、state dict（Task 10 结构）。
- Produces:
  - `Rec` dataclass：`code, name, horizon, strategy, reason, price, position_pct, stop_price, exit_hint`
  - `position_pct(win_rate: float) -> float`（Global Constraints 的仓位规则）
  - `generate_recommendations(conn, state: dict, pool: dict | None = None, top_n: int = 10) -> tuple[dict[str, list[Rec]], dict]`
    - 返回 `(recs, summary)`；`recs` 键 `"short"/"long"`；`summary[horizon] = {"strategy", "win_rate", "annual_return", "max_drawdown"}`（取最近一个窗口的回测数据）
  - 规则：短线 stop_price = 最近收盘 × 0.93（保留 2 位小数）；长线 stop_price = None，用 exit_hint 表达离场条件。仅当某股票最后一根 bar 信号为 1 时入选。

- [ ] **Step 1: 写失败测试 tests/test_recommend.py**

```python
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
```

- [ ] **Step 2: 跑测试确认失败**

```bash
.venv/bin/pytest tests/test_recommend.py -v
```

- [ ] **Step 3: 实现 astock/recommend/engine.py**

```python
from dataclasses import dataclass

from ..data.dataset import build_pool_bars
from ..data.store import load_constituents
from ..strategies import STRATEGIES_BY_NAME

DISCLAIMER = "仅为量化策略参考，不构成投资建议"


@dataclass
class Rec:
    code: str
    name: str
    horizon: str
    strategy: str
    reason: str
    price: float
    position_pct: float
    stop_price: float | None
    exit_hint: str


def position_pct(win_rate):
    return round(min(15.0, max(5.0, 10.0 + (win_rate - 0.5) * 20)), 1)


def generate_recommendations(conn, state, pool=None, top_n=10):
    cons = load_constituents(conn)
    names = dict(zip(cons["code"], cons["name"]))
    if pool is None:
        pool = build_pool_bars(conn, cons["code"].tolist())
    recs = {"short": [], "long": []}
    summary = {}
    for horizon in ("short", "long"):
        best = state["best"][horizon]
        strat = STRATEGIES_BY_NAME[best]
        entry = next(e for e in state["ranking"][horizon]
                     if e["strategy"] == best)
        last = entry["windows"][-1]
        summary[horizon] = {"strategy": best, "win_rate": last["win_rate"],
                            "annual_return": last["annual_return"],
                            "max_drawdown": last["max_drawdown"]}
        for code, bars in pool.items():
            sigs = strat.signals(bars)
            if len(sigs) and sigs.iloc[-1] == 1:
                price = float(bars["close"].iloc[-1])
                stop = round(price * 0.93, 2) if horizon == "short" else None
                recs[horizon].append(Rec(
                    code, names.get(code, ""), horizon, best,
                    strat.reason(bars, bars["date"].iloc[-1]), price,
                    position_pct(last["win_rate"]), stop, strat.exit_hint()))
        recs[horizon] = recs[horizon][:top_n]
    return recs, summary
```

- [ ] **Step 4: 跑测试确认通过**

```bash
.venv/bin/pytest tests/test_recommend.py -v
```

预期：2 passed。

- [ ] **Step 5: Commit**

```bash
git add astock/recommend/engine.py tests/test_recommend.py
git commit -m "feat: 推荐生成引擎"
```

---

### Task 12: tracker/tracker.py — 推荐效果跟踪

**Files:**
- Create: `astock/tracker/tracker.py`
- Test: `tests/test_tracker.py`

**Interfaces:**
- Consumes: `store`（recommendations 表已在 Task 2 的 SCHEMA 建好）、`recommend.engine.Rec`。
- Produces:
  - `record_recommendations(conn, rec_date: str, recs: dict[str, list[Rec]]) -> None`
  - `evaluate(conn) -> dict`：`{"overall": {"n","hit_rate","avg_ret"}, "by_horizon": {h: {...}}, "detail": pd.DataFrame}`；无推荐记录时 `{"overall": {"n": 0, "hit_rate": 0.0, "avg_ret": 0.0}, "by_horizon": {}, "detail": 空DataFrame}`
  - 评价规则：对每条推荐，取 rec_date 之后的最新收盘价，`ret = latest_close / entry_price - 1`；hit = ret > 0。

- [ ] **Step 1: 写失败测试 tests/test_tracker.py**

```python
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
                                   {"short": [_rec("600000", 100.0)],
                                    "long": [_rec("000001", 100.0)]})
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
```

- [ ] **Step 2: 跑测试确认失败**

```bash
.venv/bin/pytest tests/test_tracker.py -v
```

- [ ] **Step 3: 实现 astock/tracker/tracker.py**

```python
import pandas as pd

from ..data import store


def record_recommendations(conn, rec_date, recs):
    rows = []
    for horizon, items in recs.items():
        for r in items:
            rows.append((rec_date, r.code, r.name, r.horizon, r.strategy,
                         r.reason, r.price, r.position_pct, r.stop_price,
                         r.exit_hint))
    conn.executemany(
        "INSERT INTO recommendations(rec_date, code, name, horizon, strategy,"
        " reason, entry_price, position_pct, stop_price, exit_hint)"
        " VALUES(?,?,?,?,?,?,?,?,?,?)", rows)
    conn.commit()


def evaluate(conn):
    recs = pd.read_sql("SELECT * FROM recommendations", conn)
    rows = []
    for r in recs.itertuples():
        bars = store.load_bars(conn, r.code)
        after = bars[bars["date"] > pd.Timestamp(r.rec_date)]
        if after.empty:
            continue
        latest = float(after["close"].iloc[-1])
        rows.append({"code": r.code, "horizon": r.horizon, "strategy": r.strategy,
                     "rec_date": r.rec_date, "ret": latest / r.entry_price - 1,
                     "days": len(after)})
    detail = pd.DataFrame(rows)
    if detail.empty:
        return {"overall": {"n": 0, "hit_rate": 0.0, "avg_ret": 0.0},
                "by_horizon": {}, "detail": detail}

    def summarize(g):
        return {"n": len(g), "hit_rate": float((g["ret"] > 0).mean()),
                "avg_ret": float(g["ret"].mean())}

    return {"overall": summarize(detail),
            "by_horizon": {h: summarize(g) for h, g in detail.groupby("horizon")},
            "detail": detail}
```

- [ ] **Step 4: 跑测试确认通过**

```bash
.venv/bin/pytest tests/test_tracker.py -v
```

预期：2 passed。

- [ ] **Step 5: Commit**

```bash
git add astock/tracker/tracker.py tests/test_tracker.py
git commit -m "feat: 推荐效果跟踪"
```

---

### Task 13: cli.py — 命令行入口

**Files:**
- Create: `astock/cli.py`
- Test: `tests/test_cli.py`（只测 recommend 的前置校验逻辑，全流程留给 Task 14 集成测试）

**Interfaces:**
- Consumes：前面所有任务（store/fetcher/dataset/rank/recommend/tracker）。
- Produces:
  - `main(argv: list[str] | None = None) -> int`（argparse 入口，返回退出码；`pyproject.toml` 已注册 `astock = astock.cli:main`）
  - `default_home() -> Path`：`Path(os.environ.get("ASTOCK_HOME", Path.home() / ".astock"))`
  - 子命令：`update` / `iterate` / `recommend` / `report`
  - 前置校验（设计文档第 8 节）：`recommend` 时无 state.json → 打印"请先运行 astock iterate"并返回 2；最新行情日期距今超过 3 个交易日 → 打印提醒（不阻断）。

- [ ] **Step 1: 写失败测试 tests/test_cli.py**

```python
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
```

- [ ] **Step 2: 跑测试确认失败**

```bash
.venv/bin/pytest tests/test_cli.py -v
```

- [ ] **Step 3: 实现 astock/cli.py**

```python
import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from .backtest import rank
from .data import fetcher, store
from .recommend import engine as rec_engine
from .tracker import tracker

HISTORY_YEARS = 3
FETCH_SLEEP = 0.3
INDEX_CODES = ["000300", "000905"]  # 沪深300 + 中证500


def default_home():
    return Path(os.environ.get("ASTOCK_HOME", Path.home() / ".astock"))


def cmd_update(args):
    home = default_home()
    conn = store.connect(home / "astock.db")
    cons = pd.concat([fetcher.fetch_index_constituents(c) for c in INDEX_CODES])
    cons = cons.drop_duplicates("code").reset_index(drop=True)
    store.save_constituents(conn, cons)
    print(f"成分股：{len(cons)} 只（沪深300+中证500 去重）")
    earliest = (pd.Timestamp.today() - pd.DateOffset(years=HISTORY_YEARS)).strftime("%Y-%m-%d")
    ok, failed = 0, []
    for i, code in enumerate(cons["code"], 1):
        try:
            last = store.last_bar_date(conn, code)
            start = earliest if last is None else (
                pd.Timestamp(last) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            bars = fetcher.fetch_daily_bars(code, start)
            if len(bars):
                store.upsert_bars(conn, code, bars)
            store.upsert_pe(conn, code, fetcher.fetch_pe_series(code))
            store.upsert_financials(conn, code, fetcher.fetch_financials(code))
            ok += 1
        except Exception as e:
            failed.append(code)
            print(f"  [警告] {code} 抓取失败：{e}")
        if i % 50 == 0:
            print(f"  进度 {i}/{len(cons)}")
        time.sleep(FETCH_SLEEP)
    print(f"完成：{ok} 成功 / {len(failed)} 失败")
    if failed:
        print(f"失败代码（前 10）：{failed[:10]}")
    return 0


def cmd_iterate(args):
    home = default_home()
    conn = store.connect(home / "astock.db")
    print("滚动回测中（最近 2 年，4 个半年窗口）…")
    state = rank.run_iteration(conn, home / "state.json", home / "history.jsonl")
    for horizon, label in (("short", "短线"), ("long", "长线")):
        print(f"\n{label}策略排名：")
        for e in state["ranking"][horizon]:
            mark = "★" if e["strategy"] == state["best"][horizon] else " "
            last = e["windows"][-1]
            print(f" {mark} {e['strategy']:<14} 综合分 {e['score']:>6.2f}  "
                  f"最近窗口: 胜率 {last['win_rate']:.0%}  "
                  f"年化 {last['annual_return']:.1%}  回撤 {last['max_drawdown']:.1%}")
    return 0


def _check_stale(conn):
    row = conn.execute("SELECT MAX(date) FROM daily_bars").fetchone()
    if not row or not row[0]:
        print("[提醒] 本地无行情数据，请先运行 astock update")
        return
    last = pd.Timestamp(row[0])
    if len(pd.bdate_range(last, pd.Timestamp.today())) - 1 > 3:
        print(f"[提醒] 行情数据停留在 {row[0]}，已超过 3 个交易日未更新，"
              f"建议先运行 astock update")


def cmd_recommend(args):
    home = default_home()
    state_path = home / "state.json"
    if not state_path.exists():
        print("尚未迭代过策略，请先运行 astock iterate")
        return 2
    conn = store.connect(home / "astock.db")
    _check_stale(conn)
    state = json.loads(state_path.read_text())
    recs, summary = rec_engine.generate_recommendations(conn, state)
    print(f"策略迭代时间：{state['updated_at']}")
    for horizon, label in (("short", "短线（持有数天~两周）"), ("long", "长线（数月）")):
        s = summary[horizon]
        print(f"\n=== {label} | 策略 {s['strategy']} "
              f"(最近窗口: 胜率 {s['win_rate']:.0%} 年化 {s['annual_return']:.1%} "
              f"最大回撤 {s['max_drawdown']:.1%}) ===")
        if not recs[horizon]:
            print("  今日无符合条件的股票")
        for r in recs[horizon]:
            stop = f"止损 {r.stop_price}" if r.stop_price is not None else r.exit_hint
            print(f"  {r.code} {r.name}  现价 {r.price}  建议仓位 {r.position_pct}%")
            print(f"    理由：{r.reason}")
            print(f"    风控：{stop}")
    print(f"\n{rec_engine.DISCLAIMER}")
    tracker.record_recommendations(
        conn, datetime.now().strftime("%Y-%m-%d"), recs)
    return 0


def cmd_report(args):
    home = default_home()
    conn = store.connect(home / "astock.db")
    result = tracker.evaluate(conn)
    o = result["overall"]
    print(f"历史推荐 {o['n']} 条 | 命中率 {o['hit_rate']:.0%} | "
          f"平均收益 {o['avg_ret']:.1%}")
    for h, s in result["by_horizon"].items():
        label = "短线" if h == "short" else "长线"
        print(f"  {label}: {s['n']} 条, 命中率 {s['hit_rate']:.0%}, "
              f"平均收益 {s['avg_ret']:.1%}")
    if not result["detail"].empty:
        print("\n明细（最近 10 条）：")
        for r in result["detail"].tail(10).itertuples():
            print(f"  {r.rec_date} {r.code} [{r.horizon}/{r.strategy}] "
                  f"收益 {r.ret:+.1%}（{r.days} 个交易日）")
    print(f"\n{rec_engine.DISCLAIMER}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="astock", description="A股量化策略推荐工具")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("update", help="增量更新行情与财务数据")
    sub.add_parser("iterate", help="滚动回测并优选策略")
    sub.add_parser("recommend", help="输出当前最优策略的推荐")
    sub.add_parser("report", help="历史推荐效果跟踪")
    args = parser.parse_args(argv)
    return {"update": cmd_update, "iterate": cmd_iterate,
            "recommend": cmd_recommend, "report": cmd_report}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: 跑测试确认通过**

```bash
.venv/bin/pytest tests/test_cli.py -v
```

预期：2 passed。

- [ ] **Step 5: Commit**

```bash
git add astock/cli.py tests/test_cli.py
git commit -m "feat: CLI 入口（update/iterate/recommend/report）"
```

---

### Task 14: 端到端集成测试 + README

**Files:**
- Create: `tests/test_integration.py`
- Create: `README.md`

**Interfaces:**
- Consumes：全部模块。集成测试用 monkeypatch 替换 `fetcher` 的三个抓取函数，全程不访问网络。

- [ ] **Step 1: 写失败测试 tests/test_integration.py**

```python
import json
import pandas as pd
import pytest
from astock import cli
from astock.data import fetcher
from tests.conftest import make_bars


@pytest.fixture
def fake_env(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ASTOCK_HOME", str(tmp_path))
    monkeypatch.setattr(fetcher.time, "sleep", lambda s: None)
    monkeypatch.setattr(cli.time, "sleep", lambda s: None)

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

    # 再 update 一次（模拟次日新数据），report 应有评估结果
    assert cli.main(["update"]) == 0
    capsys.readouterr()
    assert cli.main(["report"]) == 0
    out = capsys.readouterr().out
    assert "历史推荐" in out
```

- [ ] **Step 2: 跑测试确认失败**

```bash
.venv/bin/pytest tests/test_integration.py -v
```

- [ ] **Step 3: 调试直至通过**

本任务不新增实现代码；若失败，根据失败信息修正前面任务的真实缺陷（不允许改测试来迁就实现，除非测试本身与本计划接口矛盾）。常见点：

- `iterate` 中价值策略 rolling quantile 在 501 根数据上应正常工作（窗口 756、min_periods=60）
- `recommend` 前置校验：state.json 已存在 → 走正常流程

```bash
.venv/bin/pytest tests/test_integration.py -v
```

预期：1 passed。

- [ ] **Step 4: 全量回归**

```bash
.venv/bin/pytest -v
```

预期：全部通过（约 27 个测试）。

- [ ] **Step 5: 写 README.md**

```markdown
# astock — A股量化策略推荐工具

本地 CLI：抓取沪深300+中证500 成分股数据，内置量化策略库滚动回测优选，按需输出推荐。

## 安装

    python3 -m venv .venv
    .venv/bin/pip install -e ".[dev]"

## 使用

    astock update      # 首次全量 + 之后增量更新数据（约 800 只股票，首次需数分钟）
    astock iterate     # 滚动回测（最近 2 年 × 4 个半年窗口），优选短线/长线策略
    astock recommend   # 输出推荐：清单+理由+仓位+止损，附策略近期表现
    astock report      # 历史推荐的实际表现（命中率、平均收益）

数据存于 `~/.astock/`（可用环境变量 `ASTOCK_HOME` 覆盖）。

## 测试

    .venv/bin/pytest -v

> 仅为量化策略参考，不构成投资建议。
```

- [ ] **Step 6: Commit**

```bash
git add tests/test_integration.py README.md
git commit -m "test: 端到端集成测试与 README"
```

---

## 完成标准

- `.venv/bin/pytest -v` 全部通过
- `astock update && astock iterate && astock recommend && astock report` 在真实数据上跑通（需联网，由用户或执行者在最后验证）
- 设计文档第 2 节决策表中的每一项都有对应实现
