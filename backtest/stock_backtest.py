# -*- coding: utf-8 -*-
"""单只股票回测引擎。

核心设计：
- 防未来函数（结构性）：主循环第 i 日，策略函数只能看到 ``data.iloc[:i]``
  （截至前一交易日）的数据，成交发生在第 i 日——信号决策与成交价在时间上硬隔离。
- 成交建模：买入按 ``price*(1+slippage)`` 成交再计佣金，卖出按 ``price*(1-slippage)``
  成交再扣佣金，双边都向不利方向调整，避免高估策略收益。
- 仓位模型：单票全仓（空仓才买入、持仓才卖出），可选 5% 固定止损。

组合层批量回测见 ``portfolio_backtest.PortfolioBacktest``；
带每日持仓快照的进化版本见 ``app/backtest.py``。
"""

import re

import numpy as np
import pandas as pd
from dateutil import parser


class StockBacktest:
    """单只股票的事件循环回测器。"""

    def __init__(self, initial_capital=100000, commission=0.001, slippage=0.001):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions = 0
        self.trades = []
        self.portfolio_values = []
        self.dates = []
        self.commission = commission
        self.slippage = slippage
        self.position_price = 0
        self.current_stock = None

    @staticmethod
    def is_date_string_advanced(date_str):
        """判断字符串是否可解析为日期（dateutil，兼容多种格式）。"""
        try:
            if re.match(r"^\d+$", str(date_str)):
                # 纯数字字符串只接受 4/6/8 位（年 / 年月 / 年月日）
                if len(str(date_str)) not in (4, 6, 8):
                    return False
            parser.parse(str(date_str), fuzzy=False)
            return True
        except (ValueError, TypeError, OverflowError):
            return False

    def run_backtest(self, data, strategy_function, stock_code=None, enable_stop=True):
        """运行单只股票的回测。

        Args:
            data: 行情 DataFrame，必须含 'close' 列（可含 'date' 列或日期索引）。
            strategy_function: 策略函数，输入截至前一日的行情，返回 1/-1/0。
            stock_code: 股票代码，仅用于交易记录。
            enable_stop: 是否启用 5% 固定止损。

        Returns:
            dict: ``calculate_metrics()`` 输出的回测指标。
        """
        self.current_stock = stock_code
        self.data = data.copy()
        self.capital = self.initial_capital
        self.positions = 0
        self.position_price = 0
        self.trades = []
        self.portfolio_values = []
        self.dates = []

        for i in range(1, len(data)):
            # 信号只能看到截至前一日的数据——结构性防未来函数
            current_data = data.iloc[:i]
            current_price = data.iloc[i]["close"]

            if "date" in data.columns:
                current_date = data.iloc[i]["date"]
            elif self.is_date_string_advanced(data.index[i]):
                current_date = data.index[i]
            else:
                current_date = i

            signal = strategy_function(current_data)

            # 固定止损：持仓成本价下跌 5% 强制平仓
            trade_subtype = None
            if (
                enable_stop
                and self.positions > 0
                and current_price < self.position_price * 0.95
            ):
                signal = -1
                trade_subtype = "STOP_LOSS"

            self.execute_trade(signal, trade_subtype, current_price, current_date)

            portfolio_value = self.capital + self.positions * current_price
            self.portfolio_values.append(portfolio_value)
            self.dates.append(current_date)

        return self.calculate_metrics()

    def execute_trade(self, signal, trade_subtype, price, date):
        """执行交易：滑点调整成交价，佣金计入成本/收入。"""
        if signal == 1 and self.positions == 0:  # 买入信号，且当前空仓
            execution_price = price * (1 + self.slippage)
            max_shares = self.capital // (execution_price * (1 + self.commission))

            if max_shares > 0:
                self.positions = max_shares
                cost = self.positions * execution_price * (1 + self.commission)
                self.capital -= cost
                self.position_price = execution_price
                self.trades.append(
                    {
                        "type": "BUY",
                        "date": date,
                        "price": execution_price,
                        "shares": self.positions,
                        "cost": cost,
                        "stock": self.current_stock,
                    }
                )

        elif signal == -1 and self.positions > 0:  # 卖出信号，且当前持仓
            execution_price = price * (1 - self.slippage)
            revenue = self.positions * execution_price * (1 - self.commission)
            self.capital += revenue

            # 盈亏基于买入成交价（已含买入侧滑点与佣金）
            profit = revenue - (self.positions * self.position_price)

            self.trades.append(
                {
                    "type": "SELL",
                    "sub_type": trade_subtype,
                    "date": date,
                    "price": execution_price,
                    "shares": self.positions,
                    "revenue": revenue,
                    "profit": profit,
                    "stock": self.current_stock,
                }
            )

            self.positions = 0
            self.position_price = 0

    def calculate_metrics(self):
        """计算回测指标：总收益、年化、夏普、最大回撤、胜率等。"""
        if len(self.portfolio_values) == 0:
            return {}

        returns = pd.Series(self.portfolio_values).pct_change().dropna()

        if len(returns) == 0:
            return {}

        total_return = (
            self.portfolio_values[-1] - self.initial_capital
        ) / self.initial_capital
        trading_days = len(self.portfolio_values)

        # 年化收益率（按实际交易天数折算）
        annual_return = (
            (1 + total_return) ** (252 / trading_days) - 1 if trading_days > 0 else 0
        )

        if returns.std() > 0:
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
        else:
            sharpe_ratio = 0

        max_drawdown = self.calculate_max_drawdown()

        winning_trades = len([t for t in self.trades if t.get("profit", 0) > 0])
        total_trades = len([t for t in self.trades if "profit" in t])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        return {
            "stock_code": self.current_stock,
            "initial_capital": self.initial_capital,
            "final_value": (
                self.portfolio_values[-1]
                if self.portfolio_values
                else self.initial_capital
            ),
            "total_return": total_return,
            "annual_return": annual_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "total_trades": len(self.trades),
            "win_rate": win_rate,
            "avg_trade_profit": (
                np.mean([t.get("profit", 0) for t in self.trades if "profit" in t])
                if total_trades > 0
                else 0
            ),
        }

    def calculate_max_drawdown(self):
        """计算最大回撤（峰值到谷值的最大跌幅）。"""
        if not self.portfolio_values:
            return 0

        peak = self.portfolio_values[0]
        max_dd = 0

        for value in self.portfolio_values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd

        return max_dd
