# MarsFinance

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**[English →](README.md)**

面向 A 股市场的开源量化研究框架：QMT/xtquant 数据接入、量价策略族、
带真实交易摩擦的事件循环回测引擎，以及 XGBoost 收益预测流水线。

2024–2026.03 期间作为我个人系统化投资实践的研究基础设施而构建，
现作为研究记录保持公开。后继系统（基本面驱动的投资决策系统）在私有仓库中开发。

## 核心特性

- **结构性防未来函数。** 主循环第 *i* 日，策略函数只能看到 `data.iloc[:i]`
  （截至第 *i−1* 日）的数据，成交用第 *i* 日价格——信号与成交在结构上硬隔离，
  而非事后检查。（`backtest/stock_backtest.py`）
- **真实交易摩擦。** 佣金与滑点双边建模：买入按 `price × (1 + slippage)`
  成交再计佣金，卖出按 `price × (1 − slippage)` 成交再扣佣金。
- **量价策略族。** 基础量价信号 + 三重加固：波动率自适应阈值、3σ 异常成交量
  过滤、多时间框架动量确认——信号冲突时宁可不做。
  （`strategies/volume_price_strategy.py`）
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
├── backtest/             # 单票与组合回测引擎、多策略对比
├── strategies/           # 量价策略族 + 经典基准（双均线 / RSI / 布林带）
├── machine_learning/     # XGBoost 收益预测流水线
├── data_loader/          # QMT/xtdata 行情加载
├── app/                  # 研究 notebook + 进化版引擎（每日持仓快照）
├── examples/             # 可运行使用示例
└── src/research_papers/  # 复现论文（引用清单；不再分发 PDF）
```

## 快速开始

```bash
pip install -r requirements.txt
```

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

## 数据源

- **A 股：** `xtquant`/`xtdata`，随 QMT / MiniQMT 交易终端分发（不在 PyPI，
  需从 QMT 安装目录复制或加入 `PYTHONPATH`）。已做延迟导入，未安装 xtquant
  不影响包的其余部分使用。
- **美股实验：** `yfinance`。

## 项目状态（如实说明）

- **已归档**（开发期 2024.01–2026.03），作为研究记录维护，非生产系统。
- `app/` 下的 notebook 是不同成熟度的研究产物，所有输出已清除。
- 策略参数为手工调校的默认值，未经系统化校准。这一课已作为硬约束带入后继
  系统：任何参数不经样本外验证不得进入生产。

## 免责声明

仅供研究与学习。本仓库任何内容均不构成投资建议。

## 许可

[MIT](LICENSE) — © 黄隆 (Long Huang)
