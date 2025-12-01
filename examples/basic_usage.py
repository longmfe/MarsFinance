"""
MarsFinance 基础使用示例
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from marsfinance import DataLoader, PortfolioBacktest
from marsfinance.strategies import EnhancedVolumePriceStrategy

def main():
    print("🚀 MarsFinance 基础使用示例")
    print("=" * 40)
    
    # 1. 初始化数据加载器
    print("1. 初始化数据加载器...")
    loader = DataLoader()
    
    # 2. 加载数据
    print("2. 加载沪深300数据...")
    stock_data = loader.load_hs300_data('20230101', '20231231')
    
    # 3. 运行回测
    print("3. 运行策略回测...")
    backtest = PortfolioBacktest(initial_capital=1000000)
    
    # 注意：这里需要实际数据才能运行，目前是框架演示
    if stock_data:
        backtest.run_stock_universe_backtest(
            stock_data_dict=stock_data,
            strategy_function=EnhancedVolumePriceStrategy,
            capital_per_stock=20000
        )
        
        # 4. 查看结果
        backtest.print_detailed_report()
    else:
        print("⚠️  暂无数据，回测框架就绪")
    
    print("✅ 基础示例完成")

if __name__ == "__main__":
    main()
