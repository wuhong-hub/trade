# astock — A股量化策略推荐工具

本地 CLI：抓取沪深300+中证500 成分股数据，内置量化策略库滚动回测优选，按需输出推荐。

## 安装

    python3 -m venv .venv
    .venv/bin/pip install -e ".[dev]"

## 使用

    astock update      # 首次全量 + 之后增量更新数据（约 800 只股票，首次需数分钟）
    astock update --years 10  # 回填 10 年历史
    astock iterate     # 滚动回测（最近 2 年 × 4 个半年窗口），优选短线/长线策略
    astock recommend   # 输出推荐：清单+理由+仓位+止损，附策略近期表现
    astock report      # 历史推荐的实际表现（命中率、平均收益）

数据存于 `~/.astock/`（可用环境变量 `ASTOCK_HOME` 覆盖）。

## 已知局限

- 财务指标（ROE、营收增长）按报告期 + 90 天的近似公告日合并进日线，公告日前不可见。
  该滞后是粗略近似（实际季报约 1 个月、年报最长 4 个月），回测结果仍可能偏乐观。
- 增量更新使用前复权（qfq）数据：除权季后新抓取的复权价与库中历史价口径可能不一致，
  拼接处会出现虚假跳空。建议除权季删除 `~/.astock/astock.db` 后重新运行
  `astock update` 全量重抓。

## 测试

    .venv/bin/pytest -v

> 仅为量化策略参考，不构成投资建议。
