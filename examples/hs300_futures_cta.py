# -*- coding: utf-8 -*-
"""沪深300股指期货（IF）CTA 回测示例。

数据源：akshare 新浪期货日线，IF0 主力连续（已复权拼接，换月无跳空）。
回测引擎：``backtest.futures_backtest``——多空双向、按手数开仓、
12% 保证金、双边 0.23bp 佣金 + 0.05% 滑点。

用法：
    python examples/hs300_futures_cta.py            # 常规对比
    python examples/hs300_futures_cta.py --force    # 强制重新取数
    python examples/hs300_futures_cta.py --plot     # 存净值曲线图
"""

import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from backtest.futures_backtest import FuturesBacktest  # noqa: E402
from research.datafeed.akshare_source import fetch_futures_daily  # noqa: E402
from strategies.etf_trend_strategy import etf_ma_cross, etf_trend_regime  # noqa: E402
from strategies.futures_cta import futures_donchian, futures_ma_regime  # noqa: E402

INITIAL_CAPITAL = 2_000_000
SYMBOL = "IF0"
# 沪深300股指期货合约参数
MULTIPLIER = 300
MARGIN_RATE = 0.12
COMMISSION_RATE = 0.000023  # 成交金额的 0.23bp
SLIPPAGE = 0.0005
# CTA 风险参数：目标年化波动率（仓位按 ATR 倒推，而非满保证金加杠杆）
ANNUAL_VOL_TARGET = 0.15
ATR_PERIOD = 20


def buy_and_hold_metrics(close: pd.Series) -> dict:
    """从收盘价序列直接算持有到期的指标（无杠杆参考基准）。"""
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


def new_backtest():
    """按 IF 合约参数构造回测器（波动率目标仓位 + 保证金上限）。"""
    return FuturesBacktest(
        initial_capital=INITIAL_CAPITAL,
        multiplier=MULTIPLIER,
        margin_rate=MARGIN_RATE,
        commission_rate=COMMISSION_RATE,
        slippage=SLIPPAGE,
        annual_vol_target=ANNUAL_VOL_TARGET,
        atr_period=ATR_PERIOD,
    )


def print_comparison(rows: dict, start_date: str, end_date: str) -> None:
    """打印对比表：CTA 策略（多空/纯多）+ 期货价持有基准。"""
    table = pd.DataFrame(rows).T
    table["total_return"] = table["total_return"].map("{:.2%}".format)
    table["annual_return"] = table["annual_return"].map("{:.2%}".format)
    table["sharpe_ratio"] = table["sharpe_ratio"].map("{:.2f}".format)
    table["max_drawdown"] = table["max_drawdown"].map("{:.2%}".format)
    table["total_trades"] = table["total_trades"].astype(int)
    table["win_rate"] = table["win_rate"].map("{:.2%}".format)
    table["long_trades"] = table["long_trades"].astype(int)
    table["short_trades"] = table["short_trades"].astype(int)
    table["margin_calls"] = table["margin_calls"].astype(int)

    print("\n" + "=" * 100)
    print(
        f"IF0 CTA 对比 | 区间 {start_date} ~ {end_date} | "
        f"初始资金 {INITIAL_CAPITAL:,} | 每手 {MULTIPLIER}×指数点 | "
        f"目标年化波动 {ANNUAL_VOL_TARGET:.0%} (ATR {ATR_PERIOD} 日) | "
        f"保证金上限 {MARGIN_RATE:.0%}"
    )
    print("=" * 100)
    print(table.to_string())
    print("=" * 100)


def parameter_sweep(futures_data: pd.DataFrame) -> None:
    """对主策略做小范围参数扫描，防止默认值只是运气。"""
    combos = [
        (fast, slow, regime)
        for fast in (10, 20)
        for slow in (50, 100)
        for regime in (150, 250)
    ]

    results = []
    for fast, slow, regime in combos:
        backtest = new_backtest()
        metrics = backtest.run_backtest(
            futures_data,
            lambda d, f=fast, s=slow, r=regime: futures_ma_regime(d, f, s, r),
            symbol=SYMBOL,
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

    print("\n参数扫描: futures_ma_regime（多空，按总收益降序，前 5 名）")
    display = sweep.head(5).copy()
    display["total_return"] = display["total_return"].map("{:.2%}".format)
    display["annual_return"] = display["annual_return"].map("{:.2%}".format)
    display["max_drawdown"] = display["max_drawdown"].map("{:.2%}".format)
    print(display.to_string(index=False))
    print(
        f"共 {len(sweep)} 组: 中位数 {returns.median():.2%} | "
        f"均值 {returns.mean():.2%} | 最差 {returns.min():.2%}"
    )


def plot_equity_curves(equity_curves: dict, path: str) -> None:
    """净值曲线存为 PNG（Agg 后端，无需 GUI）。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(14, 8))
    for name, series in equity_curves.items():
        plt.plot(series.index, series.values, label=name, linewidth=1.2)

    plt.title("IF0 CTA Equity Curves")
    plt.xlabel("Date")
    plt.ylabel("Equity (CNY)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"\n净值曲线已保存: {path}")


def main():
    parser = argparse.ArgumentParser(description="IF0 股指期货 CTA 回测")
    parser.add_argument("--force", action="store_true", help="忽略缓存重新取数")
    parser.add_argument("--plot", action="store_true", help="保存净值曲线 PNG")
    args = parser.parse_args()

    print("加载 IF0 期货日线（首次联网，之后走缓存）...")
    futures_data = fetch_futures_daily(SYMBOL, force=args.force)
    if futures_data.empty:
        print("未取到 IF0 行情数据")
        return

    start_date = pd.Timestamp(futures_data["date"].iloc[0]).strftime("%Y-%m-%d")
    end_date = pd.Timestamp(futures_data["date"].iloc[-1]).strftime("%Y-%m-%d")
    print(f"IF0: {len(futures_data)} 个交易日，{start_date} ~ {end_date}")

    rows = {}
    equity_curves = {}

    # 1. CTA 策略：多空双向 + 主策略的纯多对照
    runs = [
        ("ma_cross(20,60) 多空", etf_ma_cross, {}),
        ("etf_trend_regime(20,60,250) 多空", etf_trend_regime, {}),
        ("futures_ma_regime(20,60,250) 多空", futures_ma_regime, {}),
        ("futures_ma_regime(20,60,250) 纯多", futures_ma_regime, {"allow_short": False}),
        ("futures_donchian(55,20,250) 多空", futures_donchian, {}),
    ]

    for name, func, kwargs in runs:
        backtest = new_backtest()
        metrics = backtest.run_backtest(futures_data, func, symbol=SYMBOL, **kwargs)
        rows[name] = metrics
        equity_curves[name] = pd.Series(
            backtest.portfolio_values, index=futures_data["date"].iloc[1:]
        )
        print(f"  {name}: 完成 ({metrics['total_trades']} 笔平仓)")

    # 2. 期货价持有基准（无杠杆，仅作参考）
    bh = buy_and_hold_metrics(futures_data["close"])
    rows["IF0 buy-and-hold (无杠杆)"] = {
        "total_return": bh["total_return"],
        "annual_return": bh["annual_return"],
        "sharpe_ratio": bh["sharpe_ratio"],
        "max_drawdown": bh["max_drawdown"],
        "total_trades": 0,
        "win_rate": float("nan"),
        "long_trades": 0,
        "short_trades": 0,
        "margin_calls": 0,
    }
    equity_curves["IF0 buy-and-hold"] = pd.Series(
        (futures_data["close"] / futures_data["close"].iloc[0] * INITIAL_CAPITAL).values,
        index=futures_data["date"],
    )

    print_comparison(rows, start_date, end_date)

    # 3. 主策略参数扫描
    parameter_sweep(futures_data)

    # 4. 净值曲线
    if args.plot:
        plot_equity_curves(equity_curves, os.path.join("data", "hs300_futures_cta.png"))


if __name__ == "__main__":
    main()
