# MarsFinance 架构与定位

> 本文梳理仓库的结构、分层、数据流与边界，回答"这个仓库是什么、不是什么"。
> 快速上手见 [README.md](README.md)。

## 一句话定位

MarsFinance 是一个面向 **A 股市场的回测研究框架**——不是交易系统。

它回答的问题是"**这个想法在历史数据上是否成立**"，不回答"今天买什么"。
它的核心资产是三样：

1. **与未来函数在结构上绝缘的回测引擎**（信号只能看到截至前一日的切片）；
2. **两条可选的数据路径**（akshare 默认可用，QMT/xtquant 可选）；
3. **一批如实标注"参数未校准"的策略与研报复现**，拒绝把回测结果包装成收益承诺。

## 分层结构

```
┌──────────────────────────── 应用层（app/，研究产物）──────────────────────────┐
│  notebook 集合、进化版引擎（每日持仓快照）、信号生成器、特征工程               │
│  成熟度参差；部分模块需要 QMT；不做为公共接口承诺稳定                         │
└──────────────────────────────────┬───────────────────────────────────────────┘
                                   │ 组合使用
┌───────────────────┬──────────────▼───────────────┬──────────────────────────┐
│  策略层            │        回测引擎层             │       研究层              │
│  strategies/      │        backtest/             │       research/          │
│  · 量价策略族      │  · StockBacktest             │  · FFScore（Piotroski）   │
│  · ETF 趋势策略    │    单票全仓、多空不可          │  · 截面分组回测           │
│  · 期货 CTA 策略   │  · PortfolioBacktest         │  · 研报复现（CSCV/风险    │
│  · 事件策略        │    等额资金 × N 个独立账户     │    预算等，见 src/）      │
│  · 经典基准        │  · FuturesBacktest           │  · 共享指标 metrics.py    │
│  纯 pandas，       │    多空双向、波动率目标仓位     │  · 时点对齐（PIT）        │
│  只输出 1/-1/0     │  · 事件级统计 event_stats      │                          │
│                   │  摩擦与风控建模                │                          │
└───────────────────┴──────────────┬───────────────┴──────────────────────────┘
                                   │
┌───────────────────┬──────────────▼───────────────┬──────────────────────────┐
│  数据层（QMT 路径） │        数据层（akshare 路径）  │       机器学习层          │
│  data_loader/     │        research/datafeed/    │       machine_learning/  │
│  需要 QMT/MiniQMT │  · akshare_source.py 端点封装 │  XGBoost 30 日收益预测    │
│  终端，延迟导入    │  · cache.py parquet 磁盘缓存   │  （yfinance 美股数据，    │
│  · 沪深300 全池    │  · proxy.py Windows 代理直连  │   与 A 股主链路独立）      │
│  · 业绩预告事件    │  · panel.py 面板/代码格式统一  │                          │
│                   │  · calendar/fundamentals     │                          │
└───────────────────┴──────────────────────────────┴──────────────────────────┘
```

依赖方向自上而下：策略层**不知道**数据从哪来，引擎层**不关心**策略怎么想，
数据层**不依赖**任何上层——各层只通过公开契约交互。

## 模块职责

| 包 | 职责 | 依赖（运行时） |
|---|---|---|
| `research/datafeed/` | akshare 取数 + 缓存 + 面板约定。**默认数据路径** | akshare、pyarrow、pandas |
| `data_loader/` | QMT/xtquant 取数 + 事件对齐。延迟导入，无 QMT 也能 import | xtquant（可选） |
| `backtest/` | 三种执行模型 + 事件级统计，摩擦与风控建模 | numpy、pandas |
| `strategies/` | 纯信号函数，统一契约，无 I/O、无状态 | pandas |
| `research/` | FFScore 因子、PIT 对齐、截面回测、指标、研报复现脚本 | scipy 等（见 requirements-research.txt） |
| `machine_learning/` | XGBoost 收益预测流水线 | yfinance、xgboost、ta、seaborn |
| `app/` | 研究 notebook 与早期引擎，**研究产物区**，不承诺稳定 | 混杂 |
| `examples/` | 四个可运行入口，新功能一律配示例 | 见下表 |
| `tests/` | 离线测试套件（默认零网络请求） | pytest |

## 核心契约：策略函数

所有策略与所有引擎之间只有一条契约：

```python
def strategy(data: pd.DataFrame) -> int:
    """data 是截至前一交易日的行情切片（引擎保证 data.iloc[:i]）。
    返回 1（做多）、-1（做空/卖出）、0（不操作）。"""
```

防未来函数不是靠事后检查，而是**结构性的**：引擎第 *i* 日调用策略时，
传进去的切片最多到第 *i−1* 日，成交发生在第 *i* 日。策略想偷看未来也偷不到。

同一契约下的三种执行模型（同一策略可跨引擎复用）：

| 引擎 | 仓位模型 | 方向 | 风控 |
|---|---|---|---|
| `StockBacktest` | 单票全仓（按股数） | 只多 | 可选 5% 固定止损 |
| `PortfolioBacktest` | 等额资金 × N 个独立单票账户 | 只多 | 同上 |
| `FuturesBacktest` | 整手合约，波动率目标仓位（ATR 倒推、保证金封顶） | 多空双向、可反手 | 保证金不足强制平仓 |

## 两条数据路径（关键结构决策）

- **akshare 路径（默认，开箱即用）**：新浪/腾讯源日线（股票、ETF、股指期货），
  全部经 parquet 磁盘缓存，一次取数、永久离线复跑。本机实测端点可用性在
  `research/datafeed/akshare_source.py` 的 docstring 里如实维护。
- **QMT 路径（可选，需要 QMT/MiniQMT 终端）**：沪深300 全池行情与业绩预告事件。
  xtquant 不在 PyPI，全部延迟导入，未安装不影响其余模块。

两条路径的产物都归一为同一约定（`{代码: DataFrame}` 或 MultiIndex 面板），
上层引擎与策略无感知差异。

## 典型工作流

| 场景 | 入口 | 数据 | 引擎 |
|---|---|---|---|
| 单资产趋势（ETF/期货） | `examples/hs300_etf_backtest.py` / `hs300_futures_cta.py` | akshare | Stock / Futures |
| 沪深300 全池量价策略 | `examples/basic_usage.py` | QMT | Portfolio |
| 业绩预告事件 | `examples/yjyg_event_example.py` | QMT + akshare | Portfolio + event_stats |
| Piotroski F-Score 截面 | `research/scripts/backtest_ffscore_pg.py` 或 ffscore 模块 | akshare / 私有 PG | `research/ffscore/backtest.py` |
| 美股收益预测 | `machine_learning/xgboost_prediction_framework.py` | yfinance | — |

## 边界：刻意不做什么

- **不做实盘/交易执行**：没有订单路由、没有风控系统、没有实盘接口。
- **不做规模化因子库**：因子研究以研报复现的深度为先，不做全市场批量因子的广度。
- **不做数据服务**：数据层自用，缓存是研究复跑的工具，不对外提供数据。
- **参数不承诺**：所有策略参数是手工调校的默认值，示例内置敏感性扫描来暴露这一点，
  而非隐藏它。任何参数不经样本外验证不进入生产——这一约束属于后继系统。
- **与后继系统的关系**：基本面驱动的投资决策系统在私有仓库（mars-invest-os）
  开发；`research/scripts/backtest_ffscore_pg.py` 是两者之间的桥——研究在本仓库，
  决策在后继系统。本仓库作为研究记录公开，后继系统不公开。

## 演进脉络

| 阶段 | 内容 |
|---|---|
| 2024.01–2025 | QMT/xtquant 数据接入、量价策略族、单票/组合引擎、XGBoost 流水线（`app/` 时代） |
| 2025–2026.03 | akshare 数据层（缓存/代理/面板）、FFScore 研报复现、事件策略、PIT 对齐（`research/` 时代） |
| 2026.08 | ETF 趋势策略族、股指期货 CTA 引擎（多空 + 波动率目标仓位）、仓库整理与文档 |
