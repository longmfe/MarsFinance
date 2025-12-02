class MultiStrategyBacktest:
    """多策略回测框架"""
    
    def __init__(self, initial_capital=1000000):
        self.initial_capital = initial_capital
        self.strategy_results = {}
        self.comparison_results = {}
        
    def run_strategy_comparison(self, stock_data_dict, strategies_dict, 
                              start_date=None, end_date=None,
                              capital_per_stock=100000):
        """
        运行多策略比较回测
        
        Parameters:
        stock_data_dict: 股票数据字典
        strategies_dict: 策略字典，{策略名称: 策略函数}
        """
        print("=" * 80)
        print("多策略回测比较")
        print("=" * 80)
        print(f"策略数量: {len(strategies_dict)}")
        print(f"股票数量: {len(stock_data_dict)}")
        print(f"时间范围: {start_date} 到 {end_date}")
        print("=" * 80)
        
        for strategy_name, strategy_func in strategies_dict.items():
            print(f"\n正在运行策略: {strategy_name}")
            
            portfolio_backtest = PortfolioBacktest(initial_capital=self.initial_capital)
            portfolio_backtest.run_stock_universe_backtest(
                stock_data_dict=stock_data_dict,
                strategy_function=strategy_func,
                start_date=start_date,
                end_date=end_date,
                capital_per_stock=capital_per_stock
            )
            
            # 保存策略结果
            self.strategy_results[strategy_name] = portfolio_backtest
            self.comparison_results[strategy_name] = portfolio_backtest.get_portfolio_metrics()
            
            # 打印策略简要结果
            metrics = self.comparison_results[strategy_name]
            print(f"{strategy_name} - 总收益: {metrics['portfolio_total_return']:.2%} | "
                  f"年化收益: {metrics['avg_annual_return']:.2%} | "
                  f"夏普比率: {metrics['avg_sharpe_ratio']:.2f}")
    
    def print_strategy_comparison(self):
        """打印策略比较报告"""
        if not self.comparison_results:
            print("没有可比较的结果")
            return
        
        print("\n" + "=" * 100)
        print("多策略比较报告")
        print("=" * 100)
        
        # 创建比较表格
        comparison_df = pd.DataFrame(self.comparison_results).T
        comparison_df = comparison_df.sort_values('portfolio_total_return', ascending=False)
        
        # 选择关键指标显示
        key_metrics = [
            'portfolio_total_return', 'avg_annual_return', 'avg_sharpe_ratio',
            'avg_max_drawdown', 'avg_win_rate', 'positive_return_ratio', 'total_trades'
        ]
        
        display_df = comparison_df[key_metrics].copy()
        display_df.columns = ['总收益率', '年化收益率', '夏普比率', '平均最大回撤', 
                            '平均胜率', '正收益比例', '总交易次数']
        
        # 格式化显示
        formatted_df = display_df.copy()
        formatted_df['总收益率'] = formatted_df['总收益率'].apply(lambda x: f"{x:.2%}")
        formatted_df['年化收益率'] = formatted_df['年化收益率'].apply(lambda x: f"{x:.2%}")
        formatted_df['平均最大回撤'] = formatted_df['平均最大回撤'].apply(lambda x: f"{x:.2%}")
        formatted_df['平均胜率'] = formatted_df['平均胜率'].apply(lambda x: f"{x:.2%}")
        formatted_df['正收益比例'] = formatted_df['正收益比例'].apply(lambda x: f"{x:.2%}")
        formatted_df['夏普比率'] = formatted_df['夏普比率'].apply(lambda x: f"{x:.2f}")
        
        print(formatted_df.to_string())
        print("=" * 100)
        
        # 找出最佳策略
        best_strategy = comparison_df['portfolio_total_return'].idxmax()
        best_return = comparison_df.loc[best_strategy, 'portfolio_total_return']
        
        print(f"\n🎯 最佳策略: {best_strategy} (总收益: {best_return:.2%})")
        
        return comparison_df
    
    def plot_strategy_comparison(self):
        """绘制策略比较图"""
        if not self.strategy_results:
            print("没有策略结果可比较")
            return
        
        plt.figure(figsize=(16, 12))
        
        # 1. 策略收益对比
        plt.subplot(2, 2, 1)
        strategy_returns = {name: result.get_portfolio_metrics()['portfolio_total_return'] 
                          for name, result in self.strategy_results.items()}
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(strategy_returns)))
        bars = plt.bar(strategy_returns.keys(), strategy_returns.values(), color=colors)
        plt.title('策略总收益对比')
        plt.ylabel('总收益率')
        plt.xticks(rotation=45)
        
        # 在柱状图上添加数值
        for bar, value in zip(bars, strategy_returns.values()):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{value:.2%}', ha='center', va='bottom')
        
        # 2. 夏普比率对比
        plt.subplot(2, 2, 2)
        strategy_sharpes = {name: result.get_portfolio_metrics()['avg_sharpe_ratio'] 
                          for name, result in self.strategy_results.items()}
        
        colors = plt.cm.Pastel1(np.linspace(0, 1, len(strategy_sharpes)))
        bars = plt.bar(strategy_sharpes.keys(), strategy_sharpes.values(), color=colors)
        plt.title('策略夏普比率对比')
        plt.ylabel('夏普比率')
        plt.xticks(rotation=45)
        
        # 3. 最大回撤对比
        plt.subplot(2, 2, 3)
        strategy_drawdowns = {name: result.get_portfolio_metrics()['avg_max_drawdown'] 
                            for name, result in self.strategy_results.items()}
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(strategy_drawdowns)))
        bars = plt.bar(strategy_drawdowns.keys(), strategy_drawdowns.values(), color=colors)
        plt.title('策略最大回撤对比')
        plt.ylabel('最大回撤')
        plt.xticks(rotation=45)
        
        # 4. 胜率对比
        plt.subplot(2, 2, 4)
        strategy_winrates = {name: result.get_portfolio_metrics()['avg_win_rate'] 
                           for name, result in self.strategy_results.items()}
        
        colors = plt.cm.Paired(np.linspace(0, 1, len(strategy_winrates)))
        bars = plt.bar(strategy_winrates.keys(), strategy_winrates.values(), color=colors)
        plt.title('策略胜率对比')
        plt.ylabel('胜率')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()


