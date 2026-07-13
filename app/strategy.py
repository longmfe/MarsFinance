import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

import datetime
import warnings

from scipy import stats
from xtquant import xtdata

warnings.filterwarnings("ignore")


class AdvancedStrategies:
    """高级策略库"""

    @staticmethod
    def dual_moving_average_cross(data, short_window=5, long_window=20):
        """双均线交叉策略"""
        if len(data) < long_window:
            return 0

        df = data.copy()
        df["short_ma"] = df["close"].rolling(window=short_window).mean()
        df["long_ma"] = df["close"].rolling(window=long_window).mean()

        current_short = df["short_ma"].iloc[-1]
        current_long = df["long_ma"].iloc[-1]

        if len(df) > 1:
            prev_short = df["short_ma"].iloc[-2]
            prev_long = df["long_ma"].iloc[-2]
        else:
            return 0

        # 金叉买入，死叉卖出
        if prev_short <= prev_long and current_short > current_long:
            return 1
        elif prev_short >= prev_long and current_short < current_long:
            return -1
        else:
            return 0

    @staticmethod
    def rsi_mean_reversion(data, period=14, oversold=30, overbought=70):
        """RSI均值回归策略"""
        if len(data) < period + 1:
            return 0

        df = data.copy()
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        current_rsi = rsi.iloc[-1]

        if current_rsi < oversold:
            return 1  # 超卖，买入
        elif current_rsi > overbought:
            return -1  # 超买，卖出
        else:
            return 0

    @staticmethod
    def bollinger_breakout(data, period=20, num_std=2):
        """布林带突破策略"""
        if len(data) < period:
            return 0

        df = data.copy()
        df["middle_band"] = df["close"].rolling(window=period).mean()
        df["std"] = df["close"].rolling(window=period).std()
        df["upper_band"] = df["middle_band"] + (df["std"] * num_std)
        df["lower_band"] = df["middle_band"] - (df["std"] * num_std)

        current_close = df["close"].iloc[-1]
        current_upper = df["upper_band"].iloc[-1]
        current_lower = df["lower_band"].iloc[-1]
        prev_close = df["close"].iloc[-2] if len(df) > 1 else current_close

        # 上突破买入，下突破卖出
        if prev_close <= current_upper and current_close > current_upper:
            return 1
        elif prev_close >= current_lower and current_close < current_lower:
            return -1
        else:
            return 0

    @staticmethod
    def macd_crossover(data, fast_period=12, slow_period=26, signal_period=9):
        """MACD交叉策略"""
        if len(data) < slow_period + signal_period:
            return 0

        df = data.copy()
        exp1 = df["close"].ewm(span=fast_period, adjust=False).mean()
        exp2 = df["close"].ewm(span=slow_period, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=signal_period, adjust=False).mean()
        histogram = macd - signal

        current_macd = macd.iloc[-1]
        current_signal = signal.iloc[-1]
        prev_macd = macd.iloc[-2] if len(macd) > 1 else current_macd
        prev_signal = signal.iloc[-2] if len(signal) > 1 else current_signal

        # MACD上穿信号线买入，下穿信号线卖出
        if prev_macd <= prev_signal and current_macd > current_signal:
            return 1
        elif prev_macd >= prev_signal and current_macd < current_signal:
            return -1
        else:
            return 0

    @staticmethod
    def volume_price_confirmation(data, volume_period=20):
        """量价确认策略"""
        if len(data) < volume_period:
            return 0

        df = data.copy()
        df["price_change"] = df["close"].pct_change()
        df["volume_ma"] = df["volume"].rolling(window=volume_period).mean()
        df["volume_ratio"] = df["volume"] / df["volume_ma"]

        current_price_change = df["price_change"].iloc[-1]
        current_volume_ratio = df["volume_ratio"].iloc[-1]
        prev_price_change = df["price_change"].iloc[-2] if len(df) > 1 else 0

        # 价涨量增买入，价跌量增卖出
        if current_price_change > 0 and current_volume_ratio > 1.2:
            return 1
        elif current_price_change < 0 and current_volume_ratio > 1.2:
            return -1
        else:
            return 0

    @staticmethod
    def mean_reversion(data, lookback=20, z_threshold=2):
        """均值回归策略（Z-score）"""
        if len(data) < lookback:
            return 0

        df = data.copy()
        returns = df["close"].pct_change().dropna()

        if len(returns) < lookback:
            return 0

        current_return = returns.iloc[-1]
        mean_return = returns.tail(lookback).mean()
        std_return = returns.tail(lookback).std()

        if std_return == 0:
            return 0

        z_score = (current_return - mean_return) / std_return

        # Z-score极端值回归
        if z_score < -z_threshold:
            return 1  # 超卖回归
        elif z_score > z_threshold:
            return -1  # 超买回归
        else:
            return 0

    @staticmethod
    def momentum_strategy(data, momentum_period=10, ma_period=20):
        """动量策略"""
        if len(data) < max(momentum_period, ma_period):
            return 0

        df = data.copy()
        df["momentum"] = df["close"] / df["close"].shift(momentum_period) - 1
        df["ma"] = df["close"].rolling(window=ma_period).mean()

        current_momentum = df["momentum"].iloc[-1]
        current_close = df["close"].iloc[-1]
        current_ma = df["ma"].iloc[-1]

        # 动量强劲且价格在均线上方买入
        if current_momentum > 0.02 and current_close > current_ma:
            return 1
        # 负动量且价格在均线下方卖出
        elif current_momentum < -0.02 and current_close < current_ma:
            return -1
        else:
            return 0


class MultiStrategyBacktest:
    """多策略回测框架"""

    def __init__(self, initial_capital=1000000):
        self.initial_capital = initial_capital
        self.strategy_results = {}
        self.comparison_results = {}

    def run_strategy_comparison(
        self,
        stock_data_dict,
        strategies_dict,
        start_date=None,
        end_date=None,
        capital_per_stock=100000,
    ):
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
                capital_per_stock=capital_per_stock,
            )

            # 保存策略结果
            self.strategy_results[strategy_name] = portfolio_backtest
            self.comparison_results[strategy_name] = (
                portfolio_backtest.get_portfolio_metrics()
            )

            # 打印策略简要结果
            metrics = self.comparison_results[strategy_name]
            print(
                f"{strategy_name} - 总收益: {metrics['portfolio_total_return']:.2%} | "
                f"年化收益: {metrics['avg_annual_return']:.2%} | "
                f"夏普比率: {metrics['avg_sharpe_ratio']:.2f}"
            )

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
        comparison_df = comparison_df.sort_values(
            "portfolio_total_return", ascending=False
        )

        # 选择关键指标显示
        key_metrics = [
            "portfolio_total_return",
            "avg_annual_return",
            "avg_sharpe_ratio",
            "avg_max_drawdown",
            "avg_win_rate",
            "positive_return_ratio",
            "total_trades",
        ]

        display_df = comparison_df[key_metrics].copy()
        display_df.columns = [
            "总收益率",
            "年化收益率",
            "夏普比率",
            "平均最大回撤",
            "平均胜率",
            "正收益比例",
            "总交易次数",
        ]

        # 格式化显示
        formatted_df = display_df.copy()
        formatted_df["总收益率"] = formatted_df["总收益率"].apply(lambda x: f"{x:.2%}")
        formatted_df["年化收益率"] = formatted_df["年化收益率"].apply(
            lambda x: f"{x:.2%}"
        )
        formatted_df["平均最大回撤"] = formatted_df["平均最大回撤"].apply(
            lambda x: f"{x:.2%}"
        )
        formatted_df["平均胜率"] = formatted_df["平均胜率"].apply(lambda x: f"{x:.2%}")
        formatted_df["正收益比例"] = formatted_df["正收益比例"].apply(
            lambda x: f"{x:.2%}"
        )
        formatted_df["夏普比率"] = formatted_df["夏普比率"].apply(lambda x: f"{x:.2f}")

        print(formatted_df.to_string())
        print("=" * 100)

        # 找出最佳策略
        best_strategy = comparison_df["portfolio_total_return"].idxmax()
        best_return = comparison_df.loc[best_strategy, "portfolio_total_return"]

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
        strategy_returns = {
            name: result.get_portfolio_metrics()["portfolio_total_return"]
            for name, result in self.strategy_results.items()
        }

        colors = plt.cm.Set3(np.linspace(0, 1, len(strategy_returns)))
        bars = plt.bar(strategy_returns.keys(), strategy_returns.values(), color=colors)
        plt.title("策略总收益对比")
        plt.ylabel("总收益率")
        plt.xticks(rotation=45)

        # 在柱状图上添加数值
        for bar, value in zip(bars, strategy_returns.values()):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001,
                f"{value:.2%}",
                ha="center",
                va="bottom",
            )

        # 2. 夏普比率对比
        plt.subplot(2, 2, 2)
        strategy_sharpes = {
            name: result.get_portfolio_metrics()["avg_sharpe_ratio"]
            for name, result in self.strategy_results.items()
        }

        colors = plt.cm.Pastel1(np.linspace(0, 1, len(strategy_sharpes)))
        bars = plt.bar(strategy_sharpes.keys(), strategy_sharpes.values(), color=colors)
        plt.title("策略夏普比率对比")
        plt.ylabel("夏普比率")
        plt.xticks(rotation=45)

        # 3. 最大回撤对比
        plt.subplot(2, 2, 3)
        strategy_drawdowns = {
            name: result.get_portfolio_metrics()["avg_max_drawdown"]
            for name, result in self.strategy_results.items()
        }

        colors = plt.cm.Set2(np.linspace(0, 1, len(strategy_drawdowns)))
        bars = plt.bar(
            strategy_drawdowns.keys(), strategy_drawdowns.values(), color=colors
        )
        plt.title("策略最大回撤对比")
        plt.ylabel("最大回撤")
        plt.xticks(rotation=45)

        # 4. 胜率对比
        plt.subplot(2, 2, 4)
        strategy_winrates = {
            name: result.get_portfolio_metrics()["avg_win_rate"]
            for name, result in self.strategy_results.items()
        }

        colors = plt.cm.Paired(np.linspace(0, 1, len(strategy_winrates)))
        bars = plt.bar(
            strategy_winrates.keys(), strategy_winrates.values(), color=colors
        )
        plt.title("策略胜率对比")
        plt.ylabel("胜率")
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.show()


# 原有的 StockBacktest 和 PortfolioBacktest 类保持不变
# 这里省略重复代码，只展示新增内容


def run_advanced_strategy_test():
    """运行高级策略测试"""
    print("🚀 开始高级策略回测比较")

    # 获取股票列表
    stock_codes = get_hs300_stock_list()[:20]  # 使用前20只股票进行快速测试
    print(f"使用 {len(stock_codes)} 只股票进行测试")

    # 获取数据
    stock_data_dict = generate_hs300_sample_data(
        stock_codes, start_date="20230101", end_date="20231231"
    )

    if not stock_data_dict:
        print("错误: 无法获取股票数据")
        return

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
        start_date="20230101",
        end_date="20231231",
        capital_per_stock=20000,
    )

    # 生成比较报告
    comparison_df = multi_backtest.print_strategy_comparison()

    # 绘制比较图表
    multi_backtest.plot_strategy_comparison()

    # 显示最佳策略的详细报告
    best_strategy_name = comparison_df["portfolio_total_return"].idxmax()
    print(f"\n📊 最佳策略 '{best_strategy_name}' 的详细报告:")
    print("=" * 70)
    multi_backtest.strategy_results[best_strategy_name].print_detailed_report()

    return multi_backtest


# 主函数更新
def main():
    """主函数"""
    # 运行高级策略测试
    run_advanced_strategy_test()


if __name__ == "__main__":
    main()
