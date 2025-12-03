import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

from xtquant import xtdata
import datetime
import dateutil

import pandas as pd
import numpy as np
import os
import requests
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time

class PortfolioBacktest:
    def __init__(self, initial_capital=1000000):
        self.initial_capital = initial_capital
        self.stock_results = {}
        self.all_trades = []
        self.portfolio_values = {}
        
    def run_stock_universe_backtest(self, stock_data_dict, strategy_function, 
                                  start_date=None, end_date=None, 
                                  capital_per_stock=100000):
        """
        遍历股票池进行回测
        
        Parameters:
        stock_data_dict: 字典，键为股票代码，值为包含股票数据的DataFrame
        strategy_function: 策略函数
        start_date: 开始日期
        end_date: 结束日期
        capital_per_stock: 每只股票分配的资金
        """
        print(f"开始回测，股票数量: {len(stock_data_dict)}")
        print(f"时间范围: {start_date} 到 {end_date}")
        print(f"每只股票资金: {capital_per_stock:,}")
        print("=" * 60)
        
        total_stocks = len(stock_data_dict)
        completed = 0
        
        for stock_code, data in stock_data_dict.items():
            # 过滤时间范围
            if start_date and end_date:
                if 'date' in data.columns:
                    mask = (data['date'] >= start_date) & (data['date'] <= end_date)
                    filtered_data = data[mask].copy()
                else:
                    mask = (data.index >= start_date) & (data.index <= end_date)
                    filtered_data = data[mask].copy()
            else:
                filtered_data = data.copy()
            
            if len(filtered_data) < 50:  # 确保有足够的数据
                continue
                
            # 运行单只股票回测
            backtest = StockBacktest(initial_capital=capital_per_stock)
            metrics = backtest.run_backtest(filtered_data, strategy_function, stock_code)
            
            # 保存结果
            self.stock_results[stock_code] = metrics
            self.all_trades.extend(backtest.trades)
            
            # 保存组合价值序列（归一化以便比较）
            if backtest.portfolio_values:
                initial_value = backtest.portfolio_values[0] if backtest.portfolio_values else 1
                normalized_values = [v / initial_value for v in backtest.portfolio_values]
                self.portfolio_values[stock_code] = {
                    'dates': backtest.dates,
                    'values': normalized_values
                }
            
            completed += 1
            if completed % 10 == 0:
                print(f"进度: {completed}/{total_stocks}")
    
    def get_portfolio_metrics(self):
        """计算组合级别的回测指标"""
        if not self.stock_results:
            return {}
            
        # 组合总收益
        total_final_value = sum(result['final_value'] for result in self.stock_results.values())
        total_initial_value = sum(result['initial_capital'] for result in self.stock_results.values())
        portfolio_return = (total_final_value - total_initial_value) / total_initial_value

        # 平均指标
        avg_annual_return = np.mean([r['annual_return'] for r in self.stock_results.values()])
        avg_sharpe = np.mean([r['sharpe_ratio'] for r in self.stock_results.values()])
        avg_max_dd = np.mean([r['max_drawdown'] for r in self.stock_results.values()])
        avg_win_rate = np.mean([r['win_rate'] for r in self.stock_results.values()])
        
        # 正收益股票比例
        positive_returns = len([r for r in self.stock_results.values() if r['total_return'] > 0])
        positive_ratio = positive_returns / len(self.stock_results)

        return {
            'portfolio_total_return': portfolio_return,
            'avg_annual_return': avg_annual_return,
            'avg_sharpe_ratio': avg_sharpe,
            'avg_max_drawdown': avg_max_dd,
            'avg_win_rate': avg_win_rate,
            'positive_return_ratio': positive_ratio,
            'total_stocks_tested': len(self.stock_results),
            'total_trades': len(self.all_trades),
            'total_final_value': total_final_value
        }
    
    def print_detailed_report(self):
        """打印详细的组合回测报告"""
        portfolio_metrics = self.get_portfolio_metrics()
        
        print("=" * 70)
        print("PORTFOLIO BACKTEST REPORT - 沪深300股票池")
        print("=" * 70)
        print(f"测试股票数量: {portfolio_metrics['total_stocks_tested']}")
        print(f"总交易次数: {portfolio_metrics['total_trades']}")
        print(f"组合总收益率: {portfolio_metrics['portfolio_total_return']:.2%}")
        print(f"平均年化收益率: {portfolio_metrics['avg_annual_return']:.2%}")
        print(f"平均夏普比率: {portfolio_metrics['avg_sharpe_ratio']:.2f}")
        print(f"平均最大回撤: {portfolio_metrics['avg_max_drawdown']:.2%}")
        print(f"平均胜率: {portfolio_metrics['avg_win_rate']:.2%}")
        print(f"正收益股票比例: {portfolio_metrics['positive_return_ratio']:.2%}")
        print(f"最终组合价值: ${portfolio_metrics['total_final_value']:,.2f}")
        print("=" * 70)
        
        # 显示表现最好和最差的股票
        if self.stock_results:
            sorted_stocks = sorted(self.stock_results.items(), 
                                 key=lambda x: x[1]['total_return'], 
                                 reverse=True)
            
            print("\n表现最好的5只股票:")
            for stock, metrics in sorted_stocks[:5]:
                print(f"  {stock}: {metrics['total_return']:.2%} (交易次数: {metrics['total_trades']})")
            
            print("\n表现最差的5只股票:")
            for stock, metrics in sorted_stocks[-5:]:
                print(f"  {stock}: {metrics['total_return']:.2%} (交易次数: {metrics['total_trades']})")
    
    def plot_portfolio_performance(self, benchmark_data_dict):
        """绘制组合表现"""
        if not self.portfolio_values:
            print("没有足够的数据进行绘图")
            return
            
        plt.figure(figsize=(15, 12))
        
        # 1. 所有股票的归一化收益曲线
        plt.subplot(2, 2, 1)
        for stock_code, data in list(self.portfolio_values.items())[:20]:  # 只显示前20只股票
            if len(data['dates']) > 0:
                plt.plot(data['dates'], data['values'], alpha=0.3, linewidth=1)
        
        # 计算平均曲线
        all_dates = set()
        for data in self.portfolio_values.values():
            all_dates.update(data['dates'])
        
        if all_dates:
            sorted_dates = sorted(all_dates)
            avg_values = []
            for date in sorted_dates:
                day_values = []
                for data in self.portfolio_values.values():
                    if date in data['dates']:
                        idx = data['dates'].index(date)
                        day_values.append(data['values'][idx])
                if day_values:
                    avg_values.append(np.mean(day_values))
            
            if len(avg_values) == len(sorted_dates):
                plt.plot(sorted_dates, avg_values, 'b-', linewidth=3, label='AvgReturn')

        if benchmark_data_dict:
            for benchmark_code, data in benchmark_data_dict.items():
                data['norm_value'] = data['close'] / data['close'][0]
                plt.plot(data['norm_value'], 'r--', linewidth=2, label='Benchmark')
                plt.xticks(range(1, len(data.index), 5), rotation=75)
              
        plt.title('Norm Curve')
        plt.ylabel('Norm')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. 收益率分布
        plt.subplot(2, 2, 2)
        returns = [metrics['total_return'] for metrics in self.stock_results.values()]
        plt.hist(returns, bins=30, alpha=0.7, edgecolor='black')
        plt.axvline(x=np.mean(returns), color='red', linestyle='--', label=f'AvgReturn: {np.mean(returns):.2%}')
        plt.title('Stock Return Dist')
        plt.xlabel('Return')
        plt.ylabel('Num of Stocks')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. 夏普比率分布
        plt.subplot(2, 2, 3)
        sharpes = [metrics['sharpe_ratio'] for metrics in self.stock_results.values()]
        plt.hist(sharpes, bins=30, alpha=0.7, edgecolor='black', color='green')
        plt.axvline(x=np.mean(sharpes), color='red', linestyle='--', label=f'AvgSharpe: {np.mean(sharpes):.2f}')
        plt.title('Sharpe Ratio Dist')
        plt.xlabel('Sharpe Ratio')
        plt.ylabel('Num of Stocks')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 4. 最大回撤分布
        plt.subplot(2, 2, 4)
        drawdowns = [metrics['max_drawdown'] for metrics in self.stock_results.values()]
        plt.hist(drawdowns, bins=30, alpha=0.7, edgecolor='black', color='orange')
        plt.axvline(x=np.mean(drawdowns), color='red', linestyle='--', label=f'avg: {np.mean(drawdowns):.2%}')
        plt.title('Max Drawdown Dist')
        plt.xlabel('Max Drawdown')
        plt.ylabel('Num of Stocks')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

