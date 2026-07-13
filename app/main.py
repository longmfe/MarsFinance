def run_advanced_strategy_test():
    """运行高级策略测试"""
    print("🚀 开始高级策略回测比较")

    # 获取股票列表
    stock_codes = get_hs300_stock_list()[:20]  # 使用前20只股票进行快速测试
    print(f"使用 {len(stock_codes)} 只股票进行测试")

    # 获取数据
    stock_data_dict = generate_hs300_sample_data(
        stock_codes, start_date="20250101", end_date="20251231"
    )

    if not stock_data_dict:
        print("错误: 无法获取股票数据")
        return

    # TODO:股票回测

    # 创建组合回测实例
    portfolio_backtest = PortfolioBacktest(initial_capital=1000000)

    # 运行组合回测
    print("开始组合回测...")
    portfolio_backtest.run_stock_universe_backtest(
        stock_data_dict=stock_data_dict,
        strategy_function=lambda x: AdvancedStrategies.volume_price_confirmation(x, 20),
        start_date="20250101",
        end_date="20251231",
        capital_per_stock=20000,
    )

    # 生成报告
    portfolio_backtest.print_detailed_report()

    # 绘制结果
    benchmark_codes = ["000300.SH"]
    benchmark_data_dict = generate_hs300_sample_data(
        benchmark_codes, start_date="20250101", end_date="20251231"
    )
    portfolio_backtest.plot_portfolio_performance(benchmark_data_dict)

    # 定义策略集合
    strategies = {
        "双均线交叉": lambda x: AdvancedStrategies.dual_moving_average_cross(x, 5, 20),
        "RSI均值回归": lambda x: AdvancedStrategies.rsi_mean_reversion(x, 14, 30, 70),
        "布林带突破": lambda x: AdvancedStrategies.bollinger_breakout(x, 20, 2),
        "MACD交叉": lambda x: AdvancedStrategies.macd_crossover(x, 12, 26, 9),
        "量价确认": lambda x: AdvancedStrategies.volume_price_confirmation(x, 20),
        "均值回归": lambda x: AdvancedStrategies.mean_reversion(x, 20, 2),
        "动量策略": lambda x: AdvancedStrategies.momentum_strategy(x, 10, 20),
    }

    # 运行多策略比较
    multi_backtest = MultiStrategyBacktest(initial_capital=500000)
    multi_backtest.run_strategy_comparison(
        stock_data_dict=stock_data_dict,
        strategies_dict=strategies,
        start_date="20250101",
        end_date="20251231",
        capital_per_stock=20000,
    )

    # 生成比较报告
    comparison_df = multi_backtest.print_strategy_comparison()

    # 绘制比较图表
    # multi_backtest.plot_strategy_comparison()

    # 显示最佳策略的详细报告
    # best_strategy_name = comparison_df['portfolio_total_return'].idxmax()
    # print(f"\n📊 最佳策略 '{best_strategy_name}' 的详细报告:")
    # print("=" * 70)
    # multi_backtest.strategy_results[best_strategy_name].print_detailed_report()

    return multi_backtest


# 主函数更新
def main():
    """主函数"""
    # 运行高级策略测试
    run_advanced_strategy_test()


if __name__ == "__main__":
    main()
