# -*- coding: utf-8 -*-
"""业绩预告事件驱动策略示例：沪深300 股票池。

行情来自 QMT/xtquant（需本机 QMT/MiniQMT 终端），
事件来自 akshare（免费、无需 token）。
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from backtest.event_stats import (  # noqa: E402
    event_summary_stats,
    print_event_report,
    summarize_events,
)
from backtest.portfolio_backtest import PortfolioBacktest  # noqa: E402
from data_loader.data_loader import DataLoader  # noqa: E402
from data_loader.event_align import (  # noqa: E402
    attach_yjyg_to_universe,
    print_coverage,
)
from data_loader.yjyg_loader import load_yjyg_events  # noqa: E402
from strategies.yjyg_event_strategy import yjyg_event_strategy  # noqa: E402

ANALYSIS_START = "20200101"
ANALYSIS_END = "20241231"
# 行情比分析区间提前一个季度：公告早于行情起点的事件会被丢弃（绝不夹到首根 bar），
# 提前加载可避免窗口开头的事件覆盖不全。
PRICE_START = "20191001"


def main():
    # 1. 加载行情。后复权：后复权不改写历史价格，是时点安全的；
    #    前复权用未来分红重写历史，本身带有轻微未来函数。
    loader = DataLoader()
    stock_data = loader.load_hs300_data(PRICE_START, ANALYSIS_END, dividend_type="back")

    if not stock_data:
        print("未获取到行情数据（需要 QMT/xtquant 环境）")
        return

    # 2. 加载业绩预告事件（按公告日期，不是报告期）
    events = load_yjyg_events(PRICE_START, ANALYSIS_END)
    print(f"\n加载业绩预告事件: {len(events)} 条")

    if events.empty:
        print("未获取到事件数据")
        return

    # 3. 按公告日期对齐到行情，并打印覆盖率诊断
    stock_data, diagnostics = attach_yjyg_to_universe(stock_data, events)
    print()
    print_coverage(diagnostics)

    # 4. 分别在关/开止损两种设定下回测，差值本身就是一个有用的数字
    for enable_stop in (False, True):
        label = "启用 5% 止损" if enable_stop else "关闭止损（策略本意）"
        print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")

        backtest = PortfolioBacktest(initial_capital=1000000)
        backtest.run_stock_universe_backtest(
            stock_data_dict=stock_data,
            strategy_function=lambda x: yjyg_event_strategy(x, holding_days=10),
            start_date=ANALYSIS_START,
            end_date=ANALYSIS_END,
            capital_per_stock=20000,
            enable_stop=enable_stop,
        )

        # 5. 事件级统计才是这里的正确口径（组合级平均意义不大，见 event_stats 模块说明）
        event_rows = summarize_events(backtest.all_trades, events)
        print_event_report(event_summary_stats(event_rows))


if __name__ == "__main__":
    main()
