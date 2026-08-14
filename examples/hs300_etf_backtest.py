# -*- coding: utf-8 -*-
"""沪深300ETF（510310）趋势策略回测示例。

数据源：akshare 新浪 ETF 日线（首次联网取数并落盘缓存，之后离线），
基准：腾讯源沪深300指数日线。回测引擎：``backtest.stock_backtest``
（双边 0.1% 佣金 + 0.1% 滑点，``enable_stop=False``，离场由策略负责）。

用法：
    python examples/hs300_etf_backtest.py            # 常规对比
    python examples/hs300_etf_backtest.py --force    # 强制重新取数
    python examples/hs300_etf_backtest.py --plot     # 存净值曲线图
"""

import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from backtest.stock_backtest import StockBacktest  # noqa: E402
from research.datafeed.akshare_source import fetch_etf_daily, fetch_index_daily  # noqa: E402
from strategies.etf_trend_strategy import (  # noqa: E402
    etf_donchian_atr,
    etf_ma_cross,
    etf_trend_regime,
)

INITIAL_CAPITAL = 1_000_000
ETF_CODE = "510310"
INDEX_CODE = "sh000300"


def buy_and_hold_metrics(close: pd.Series) -> dict:
    """从收盘价序列直接算持有到期的指标（与引擎口径一致）。"""
    total_return = close.iloc[-1] / close.iloc[0] - 1
    n_days = len(close)
    annual = (1 + total_return) ** (252 / n_days) - 1 if n_days > 0 else 0.0

    returns = close.pct_change().dropna()
    sharpe = (
        returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0.0
    )

    peak = close.iloc[0]
    max_dd = 0.0
    for value in close:
        peak = max(peak, value)
        max_dd = max(max_dd, (peak - value) / peak)

    return {
        "total_return": total_return,
        "annual_return": annual,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
    }


def run_one_backtest(data: pd.DataFrame, strategy_function, stock_code: str) -> dict:
    """跑一次单标的回测，返回引擎指标。"""
    backtest = StockBacktest(initial_capital=INITIAL_CAPITAL)
    return backtest.run_backtest(
        data, strategy_function, stock_code=stock_code, enable_stop=False
    )


def align_index_to_etf(index_data: pd.DataFrame, etf_data: pd.DataFrame) -> pd.DataFrame:
    """把指数基准截到 ETF 的日期窗口，保证持有对比口径一致。

    ``fetch_index_daily`` 的缓存里 date 列可能是字符串（缓存层归一化时
    datetime.date 会转 string），先统一转回 datetime64 再比较。
    """
    start, end = etf_data["date"].min(), etf_data["date"].max()
    index_data = index_data.copy()
    index_data["date"] = pd.to_datetime(index_data["date"])
    return index_data[(index_data["date"] >= start) & (index_data["date"] <= end)]


def print_comparison(rows: dict, start_date: str, end_date: str) -> None:
    """打印对比表：策略 + ETF 持有 + 指数持有。"""
    table = pd.DataFrame(rows).T
    table["total_return"] = table["total_return"].map("{:.2%}".format)
    table["annual_return"] = table["annual_return"].map("{:.2%}".format)
    table["sharpe_ratio"] = table["sharpe_ratio"].map("{:.2f}".format)
    table["max_drawdown"] = table["max_drawdown"].map("{:.2%}".format)
    table["total_trades"] = table["total_trades"].astype(int)
    table["win_rate"] = table["win_rate"].map("{:.2%}".format)

    print("\n" + "=" * 96)
    print(
        f"510310 策略对比 | 回测区间 {start_date} ~ {end_date} | "
        f"初始资金 {INITIAL_CAPITAL:,}"
    )
    print("=" * 96)
    print(table.to_string())
    print("=" * 96)


def parameter_sweep(etf_data: pd.DataFrame) -> None:
    """对主策略做小范围参数扫描，防止默认值只是运气。"""
    combos = [
        (fast, slow, regime)
        for fast in (10, 20)
        for slow in (50, 100)
        for regime in (150, 250)
    ]

    results = []
    for fast, slow, regime in combos:
        metrics = run_one_backtest(
            etf_data,
            lambda d, f=fast, s=slow, r=regime: etf_trend_regime(d, f, s, r),
            ETF_CODE,
        )
        results.append(
            {
                "fast": fast,
                "slow": slow,
                "regime": regime,
                "total_return": metrics["total_return"],
                "annual_return": metrics["annual_return"],
                "max_drawdown": metrics["max_drawdown"],
                "total_trades": metrics["total_trades"],
            }
        )

    sweep = pd.DataFrame(results).sort_values("total_return", ascending=False)
    returns = sweep["total_return"]

    print("\n参数扫描: etf_trend_regime（按总收益降序，前 5 名）")
    display = sweep.head(5).copy()
    display["total_return"] = display["total_return"].map("{:.2%}".format)
    display["annual_return"] = display["annual_return"].map("{:.2%}".format)
    display["max_drawdown"] = display["max_drawdown"].map("{:.2%}".format)
    print(display.to_string(index=False))
    print(
        f"共 {len(sweep)} 组: 中位数 {returns.median():.2%} | "
        f"均值 {returns.mean():.2%} | 最差 {returns.min():.2%}"
    )
    print(
        "结论：默认参数 (20, 60, 250) 是否落在头部决定了它是否可信，"
        "若最优参数与默认差距大，说明结果对参数敏感。"
    )


def plot_equity_curves(equity_curves: dict, path: str) -> None:
    """净值曲线存为 PNG（Agg 后端，无需 GUI）。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(14, 8))
    for name, series in equity_curves.items():
        plt.plot(series.index, series.values, label=name, linewidth=1.2)

    plt.title("510310 Strategy Equity Curves (normalized to initial capital)")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value (CNY)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"\n净值曲线已保存: {path}")


def main():
    parser = argparse.ArgumentParser(description="510310 趋势策略回测")
    parser.add_argument("--force", action="store_true", help="忽略缓存重新取数")
    parser.add_argument("--plot", action="store_true", help="保存净值曲线 PNG")
    args = parser.parse_args()

    print("加载 ETF 日线（首次联网，之后走缓存）...")
    etf_data = fetch_etf_daily(ETF_CODE, force=args.force)
    if etf_data.empty:
        print("未取到 510310 行情数据")
        return

    index_data = fetch_index_daily(INDEX_CODE, force=args.force)
    index_data = align_index_to_etf(index_data, etf_data)

    start_date = pd.Timestamp(etf_data["date"].iloc[0]).strftime("%Y-%m-%d")
    end_date = pd.Timestamp(etf_data["date"].iloc[-1]).strftime("%Y-%m-%d")
    print(f"ETF {ETF_CODE}: {len(etf_data)} 个交易日")
    print(f"指数 {INDEX_CODE}: {len(index_data)} 个交易日（已对齐）")

    # 1. 三类趋势策略 + 双基准
    rows = {}
    equity_curves = {}

    for name, func in [
        ("ma_cross(20,60)", etf_ma_cross),
        ("trend_regime(20,60,250)", etf_trend_regime),
        ("donchian_atr(55,20)", etf_donchian_atr),
    ]:
        backtest = StockBacktest(initial_capital=INITIAL_CAPITAL)
        metrics = backtest.run_backtest(
            etf_data, func, stock_code=ETF_CODE, enable_stop=False
        )
        rows[name] = metrics
        equity_curves[name] = pd.Series(
            backtest.portfolio_values, index=etf_data["date"].iloc[1:]
        )
        print(f"  {name}: 完成")

    bh = buy_and_hold_metrics(etf_data["close"])
    rows["ETF buy-and-hold"] = {
        "total_return": bh["total_return"],
        "annual_return": bh["annual_return"],
        "sharpe_ratio": bh["sharpe_ratio"],
        "max_drawdown": bh["max_drawdown"],
        "total_trades": 0,
        "win_rate": float("nan"),
    }
    equity_curves["ETF buy-and-hold"] = pd.Series(
        (etf_data["close"] / etf_data["close"].iloc[0] * INITIAL_CAPITAL).values,
        index=etf_data["date"],
    )

    if not index_data.empty:
        idx_bh = buy_and_hold_metrics(index_data["close"])
        rows[f"HS300 index buy-and-hold"] = {
            "total_return": idx_bh["total_return"],
            "annual_return": idx_bh["annual_return"],
            "sharpe_ratio": idx_bh["sharpe_ratio"],
            "max_drawdown": idx_bh["max_drawdown"],
            "total_trades": 0,
            "win_rate": float("nan"),
        }

    print_comparison(rows, start_date, end_date)

    # 2. 主策略参数扫描
    parameter_sweep(etf_data)

    # 3. 净值曲线
    if args.plot:
        plot_equity_curves(equity_curves, os.path.join("data", "hs300_etf_equity.png"))


if __name__ == "__main__":
    main()
