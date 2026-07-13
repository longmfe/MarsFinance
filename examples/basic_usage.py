# -*- coding: utf-8 -*-
"""MarsFinance 基础使用示例：沪深300 股票池 x 量价策略 组合回测。

数据源为 QMT/xtquant，需本机安装 QMT/MiniQMT 终端后运行。
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from backtest.portfolio_backtest import PortfolioBacktest  # noqa: E402
from data_loader.data_loader import DataLoader  # noqa: E402
from strategies.volume_price_strategy import (  # noqa: E402
    enhanced_volume_price_strategy,
)


def main():
    # 1. 加载沪深300日线数据（QMT/xtdata）
    loader = DataLoader()
    stock_data = loader.load_hs300_data("20230101", "20231231")

    if not stock_data:
        print("未获取到行情数据（需要 QMT/xtquant 环境）")
        return

    # 2. 每只股票等额资金，全池回测量价策略
    backtest = PortfolioBacktest(initial_capital=1000000)
    backtest.run_stock_universe_backtest(
        stock_data_dict=stock_data,
        strategy_function=enhanced_volume_price_strategy,
        capital_per_stock=20000,
    )

    # 3. 输出组合级回测报告
    backtest.print_detailed_report()


if __name__ == "__main__":
    main()
