import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

import datetime
import dateutil
import os
import requests
from datetime import datetime, timedelta
import time
from dateutil import parser
import re

class StockBacktest:
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

    def is_date_string_advanced(date_str):
        """
        使用dateutil库判断（更智能，能解析更多格式）
        """
        try:
            # 先做一些基本检查，避免解析像"12345"这样的纯数字字符串
            if re.match(r'^\d+$', date_str):
                # 纯数字字符串，只接受特定长度的
                if len(date_str) not in (4, 6, 8):
                    return False
            
            parser.parse(date_str, fuzzy=False)
            return True
        except (ValueError, TypeError, OverflowError):
            return False
        
    def run_backtest(self, data, strategy_function, stock_code=None, enable_stop=True):
        """
        运行单只股票的回测
        data: 包含价格数据的DataFrame，必须有'close'列
        strategy_function: 策略函数
        stock_code: 股票代码，用于记录
        """
        self.current_stock = stock_code
        self.data = data.copy()
        self.capital = self.initial_capital
        self.positions = 0
        self.trades = []
        self.portfolio_values = []
        self.dates = []
        
        for i in range(1, len(data)):
            current_data = data.iloc[:i]
            current_price = data.iloc[i]['close']
            
            # 获取日期
            if 'date' in data.columns:
                current_date = data.iloc[i]['date']
            elif self.is_date_string_advanced(data.index[i]):
                current_date = data.index[i]
            else:
                current_date = i
                
            # 获取交易信号
            signal = strategy_function(current_data)

            trade_subtype = None

            # 判断止损信号
            stop_loss = current_price < self.position_price * 0.95
            if stop_loss & enable_stop:
                signal = -1
                trade_subtype = 'STOP_LOSS'
            
            # 执行交易逻辑
            self.execute_trade(signal, trade_subtype, current_price, current_date)
            
            # 记录组合价值
            portfolio_value = self.capital + self.positions * current_price
            self.portfolio_values.append(portfolio_value)
            self.dates.append(current_date)
            
        return self.calculate_metrics()
    
    def execute_trade(self, signal, trade_subtype, price, date):
        """执行交易，考虑交易成本"""
        if signal == 1 and self.positions == 0:  # 买入信号，空仓
            # 考虑滑点和佣金
            execution_price = price * (1 + self.slippage)
            max_shares = self.capital // (execution_price * (1 + self.commission))
            
            if max_shares > 0:
                self.positions = max_shares
                cost = self.positions * execution_price * (1 + self.commission)
                self.capital -= cost
                self.position_price = execution_price
                self.trades.append({
                    'type': 'BUY', 
                    'date': date, 
                    'price': execution_price, 
                    'shares': self.positions,
                    'cost': cost,
                    'stock': self.current_stock
                })
            
        elif signal == -1 and self.positions > 0:  # 卖出信号，持仓
            execution_price = price * (1 - self.slippage)
            revenue = self.positions * execution_price * (1 - self.commission)
            self.capital += revenue
            
            # 计算这次交易的盈亏
            profit = revenue - (self.positions * self.position_price)
            
            self.trades.append({
                'type': 'SELL', 
                'sub_type': trade_subtype,
                'date': date, 
                'price': execution_price, 
                'shares': self.positions,
                'revenue': revenue,
                'profit': profit,
                'stock': self.current_stock
            })
            self.positions = 0
            self.position_price = 0
    
    def calculate_metrics(self):
        """计算回测指标"""
        if len(self.portfolio_values) == 0:
            return {}
            
        returns = pd.Series(self.portfolio_values).pct_change().dropna()
        
        if len(returns) == 0:
            return {}
        
        total_return = (self.portfolio_values[-1] - self.initial_capital) / self.initial_capital
        trading_days = len(self.portfolio_values)
        
        # 年化收益率（考虑实际交易天数）
        annual_return = (1 + total_return) ** (252 / trading_days) - 1 if trading_days > 0 else 0
        
        # 夏普比率
        if returns.std() > 0:
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
        else:
            sharpe_ratio = 0
            
        max_drawdown = self.calculate_max_drawdown()
        
        # 胜率计算
        winning_trades = len([t for t in self.trades if t.get('profit', 0) > 0])
        total_trades = len([t for t in self.trades if 'profit' in t])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        return {
            'stock_code': self.current_stock,
            'initial_capital': self.initial_capital,
            'final_value': self.portfolio_values[-1] if self.portfolio_values else self.initial_capital,
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': len(self.trades),
            'win_rate': win_rate,
            'avg_trade_profit': np.mean([t.get('profit', 0) for t in self.trades if 'profit' in t]) if total_trades > 0 else 0
        }
    
    def calculate_max_drawdown(self):
        """计算最大回撤"""
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


