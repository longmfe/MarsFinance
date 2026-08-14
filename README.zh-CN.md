# MarsFinance

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**[English →](README.md)**

面向 A 股市场的开源量化研究框架：akshare 行情接入（股票 / ETF / 股指期货）、
量价策略族、ETF 趋势策略、股指期货 CTA 引擎、带真实交易摩擦的事件循环回测，
以及 XGBoost 收益预测流水线。

2024.01–2026.03 期间作为我个人系统化投资实践的研究基础设施而构建，
2026.08 扩展了 ETF 趋势策略与股指期货 CTA 引擎，现作为研究记录保持公开。
后继系统（基本面驱动的投资决策系统）在私有仓库中开发。

## 核心特性

- **akshare 数据层。** 新浪/腾讯源的股票、ETF、股指期货（IF0 主力连续，
  已复权拼接）日线，parquet 磁盘缓存（原子写、可离线复跑）与直连代理处理。
  端点可用性在 `research/datafeed/akshare_source.py` 里如实记录——本机
  eastmoney `push2` 系端点不可用。（`research/datafeed/`）
- **结构性防未来函数。** 主循环第 *i* 日，策略函数只能看到 `data.iloc[:i]`
  （截至第 *i−1* 日）的数据，成交用第 *i* 日价格——信号与成交在结构上硬隔离，
  而非事后检查。（`backtest/stock_backtest.py`、`backtest/futures_backtest.py`）
- **真实交易摩擦。** 佣金与滑点双边建模：买入按 `price × (1 + slippage)`
  成交再计佣金，卖出按 `price × (1 − slippage)` 成交再扣佣金。
- **量价策略族。** 基础量价信号 + 三重加固：波动率自适应阈值、3σ 异常成交量
  过滤、多时间框架动量确认——信号冲突时宁可不做。
  （`strategies/volume_price_strategy.py`）
- **ETF 趋势策略。** 面向宽基指数 ETF（如 510310）的趋势跟随：双均线交叉、
  长期趋势过滤的均线交叉、唐奇安突破 + ATR 跟踪止损。附可运行回测示例
  （买入持有基准 + 参数敏感性扫描）。（`strategies/etf_trend_strategy.py`、
  `examples/hs300_etf_backtest.py`）
- **股指期货 CTA 引擎。** 多空双向、按手数开仓、波动率目标仓位（ATR 倒推、
  保证金封顶）、保证金不足强制平仓——让带杠杆的回测不被风险模型骗过去。
  配套多空对称的趋势策略。（`backtest/futures_backtest.py`、
  `strategies/futures_cta.py`、`examples/hs300_futures_cta.py`）
- **XGBoost 收益预测。** 以未来 30 日收益率为回归目标；动量 / 量能资金流 /
  技术指标 / 波动情绪四族特征；按时间顺序切分训练测试、标准化仅在训练集拟合、
  TimeSeriesSplit 时序交叉验证；评估除 MSE/MAE/R² 外重点看方向准确率。
  （`machine_learning/xgboost_prediction_framework.py`）
- **组合层。** 沪深300 股票池等额资金分配，聚合组合级指标（收益、夏普、回撤、
  胜率、正收益比例），归一化净值曲线与基准对比。
- **研报复现**（`app/` 下的 notebook）：Piotroski F-Score 的 A 股实证、
  CSCV 回测过拟合概率、风险预算与机器学习资产配置——论文清单见
  [`src/research_papers/README.md`](src/research_papers/README.md)。

## 目录结构

```
MarsFinance/
├── backtest/             # 单票、组合与股指期货回测引擎
├── strategies/           # 量价策略族、ETF 趋势、期货 CTA、经典基准
├── machine_learning/     # XGBoost 收益预测流水线（yfinance 美股数据）
├── data_loader/          # QMT/xtquant 行情加载（可选，延迟导入）
├── research/             # akshare 数据层、FFScore、截面回测、指标
├── app/                  # 研究 notebook + 进化版引擎（部分需要 QMT）
├── examples/             # 可运行示例（默认走 akshare）
├── tests/                # 离线测试套件（pytest，默认不发网络请求）
└── src/research_papers/  # 复现论文（引用清单；不再分发 PDF）
```

完整的模块地图、分层数据流与项目边界见 [`ARCHITECTURE.md`](ARCHITECTURE.md)。

## 快速开始

```bash
pip install -r requirements.txt
```

默认数据路径是 akshare（无需 QMT 终端）。首次运行联网取数（每个序列约
数秒）并落盘缓存到 `data/akshare_cache`，之后全部离线：

```bash
python examples/hs300_etf_backtest.py       # 沪深300ETF（510310）趋势策略
python examples/hs300_futures_cta.py        # IF0 股指期货 CTA（多空双向）
```

`--force` 强制重新取数，`--plot` 保存净值曲线图。

装有 QMT/MiniQMT 终端时，xtquant 路径仍然可用：

```python
from data_loader.data_loader import DataLoader
from backtest.portfolio_backtest import PortfolioBacktest
from strategies.volume_price_strategy import enhanced_volume_price_strategy

loader = DataLoader()
stock_data = loader.load_hs300_data("20230101", "20231231")

backtest = PortfolioBacktest(initial_capital=1_000_000)
backtest.run_stock_universe_backtest(
    stock_data_dict=stock_data,
    strategy_function=enhanced_volume_price_strategy,
    capital_per_stock=20_000,
)
backtest.print_detailed_report()
```

可运行版本见 [`examples/basic_usage.py`](examples/basic_usage.py)。

## 示例

| 示例 | 标的 | 依赖 |
|---|---|---|
| `examples/hs300_etf_backtest.py` | 沪深300ETF（510310） | akshare |
| `examples/hs300_futures_cta.py` | 沪深300股指期货（IF0） | akshare |
| `examples/yjyg_event_example.py` | 业绩预告事件策略 | akshare + QMT（行情） |
| `examples/basic_usage.py` | 沪深300 股票池 | QMT/MiniQMT |

## 数据源

- **akshare（默认）：** 股票、ETF、股指期货日线（新浪/腾讯源）+ 东财基本面。
  本机实测：新浪与腾讯端点可用，eastmoney `push2` 端点不可用——可用性对照表
  见 `research/datafeed/akshare_source.py`。所有取数都经过 parquet 缓存，
  研究可离线复跑。
- **QMT/xtquant（可选）：** `xtquant`/`xtdata` 随 QMT / MiniQMT 交易终端分发
  （不在 PyPI，需从 QMT 安装目录复制或加入 `PYTHONPATH`）。已做延迟导入，
  未安装 xtquant 不影响包的其余部分使用。
- **美股实验：** `yfinance`（已在 `requirements.txt` 声明，XGBoost 流水线使用）。

依赖拆分：`requirements.txt` 覆盖核心栈（引擎、策略、akshare 数据层、示例）；
`requirements-research.txt` 是 `research/` 下研报复现的额外重依赖。

## 测试

```bash
pytest            # 默认离线；-m network 运行联网集成测试
```

## 项目状态（如实说明）

- 作为研究记录维护，非生产系统。
- `app/` 下的 notebook 是不同成熟度的研究产物，所有输出已清除。
- 策略参数为手工调校的默认值，未经系统化校准——示例脚本里带参数敏感性
  扫描，把这件事摆到明面上而不是藏起来。这一课已作为硬约束带入后继系统：
  任何参数不经样本外验证不得进入生产。

## 免责声明

仅供研究与学习。本仓库任何内容均不构成投资建议。

## 许可

[MIT](LICENSE) — © 黄隆 (Long Huang)
