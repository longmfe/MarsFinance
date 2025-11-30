# 🚀 MarsFinance 量化交易研究平台

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-1.5%2B-orange)](https://pandas.pydata.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.2%2B-yellow)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**开源量化交易研究框架** | 集成数据处理、策略开发、回测验证、参数优化的完整解决方案

## 📈 项目概述

MarsFinance 是一个专业的量化交易研究平台，旨在为量化研究员和算法交易员提供从策略构思到回测验证的完整工具链。平台集成了传统量化方法和现代机器学习技术，支持多市场、多策略的量化投资研究。

### 🎯 核心价值
- **完整流水线**: 数据获取 → 策略开发 → 回测验证 → 性能分析
- **生产就绪**: 考虑交易成本、滑点、仓位限制等现实因素
- **技术驱动**: 结合传统量化方法和现代机器学习技术
- **开源透明**: 代码可复现，算法可验证，结果可追溯

## 🛠 技术架构

### 核心技术栈
```python
# 核心依赖
Python >= 3.8
Pandas >= 1.5.0
NumPy >= 1.21.0
Scikit-learn >= 1.2.0
Matplotlib >= 3.5.0
Optuna >= 3.0.0
```

### 系统目录结构
```
MarsFinance/
├── 📊 data_loader/          # 数据获取与处理模块
│   ├── __init__.py
│   └── data_loader.py
├── 🤖 strategies/           # 策略库
│   ├── __init__.py
│   └── volume_price_strategy.py
├── 🔄 backtest/             # 回测引擎
│   ├── __init__.py
│   ├── stock_backtest.py
│   └── portfolio_backtest.py
├── ⚙️ optimization/         # 参数优化
│   ├── __init__.py
│   └── parameter_optimizer.py
├── 📈 visualization/        # 可视化分析
│   ├── __init__.py
│   └── performance_plotter.py
└── 📚 examples/            # 使用示例
    └── basic_usage.py
```

## 🚀 快速开始
### 安装依赖
```bash
pip install -r requirements.txt
```

### 基本使用
```python
from marsfinance import DataLoader, PortfolioBacktest
from marsfinance.strategies import EnhancedVolumePriceStrategy

# 加载数据
loader = DataLoader()
stock_data = loader.load_hs300_data('20230101', '20231231')

# 运行回测
backtest = PortfolioBacktest(initial_capital=1000000)
backtest.run_stock_universe_backtest(
    stock_data_dict=stock_data,
    strategy_function=EnhancedVolumePriceStrategy,
    capital_per_stock=20000
)

# 查看结果
backtest.print_detailed_report()
```
