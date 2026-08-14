# -*- coding: utf-8 -*-
"""宽基指数 ETF 趋势策略族。

为什么是趋势跟随：510310（沪深300ETF）这类宽基指数标的有两个特征——
长期带正漂移，但大级别回撤又深又长（2008/2015/2018 均超 40%）；单标的、
无截面分散，均值回归类策略在单边行情里会反复"接飞刀"。因此这里只做趋势
跟随：顺势入场、趋势破坏离场，并用长期均线（regime）过滤震荡市的假信号。

信号约定与仓库一致：1 = 买入，-1 = 卖出，0 = 不操作。

回测配合 ``backtest.stock_backtest.StockBacktest`` 时建议
``enable_stop=False``：引擎内置的 5% 固定止损对指数 ETF 太紧（宽基指数的
日常波动就会触发），离场逻辑由策略自身完成。

参数是手工调试的默认值，未做样本外校准——正式使用前先跑
``examples/hs300_etf_backtest.py`` 里的参数敏感性扫描。
"""

import pandas as pd


def _cross_up(fast_prev, slow_prev, fast_curr, slow_curr) -> bool:
    """快线自下而上穿越慢线（金叉）。"""
    return fast_prev <= slow_prev and fast_curr > slow_curr


def _cross_down(fast_prev, slow_prev, fast_curr, slow_curr) -> bool:
    """快线自上而下穿越慢线（死叉）。"""
    return fast_prev >= slow_prev and fast_curr < slow_curr


def etf_ma_cross(data: pd.DataFrame, fast: int = 20, slow: int = 60) -> int:
    """双均线交叉（基线版）。

    金叉买入、死叉卖出，不做任何过滤。作为对照基线：
    ``etf_trend_regime`` 在此之上加了长期趋势过滤。

    Args:
        data: 截至前一日的行情，至少需要 'close' 列
        fast: 快均线窗口
        slow: 慢均线窗口

    Returns:
        1(金叉买入) / -1(死叉卖出) / 0(不操作)
    """
    if len(data) < slow + 1:
        return 0

    close = data["close"]
    fast_ma = close.rolling(fast).mean()
    slow_ma = close.rolling(slow).mean()

    if pd.isna(fast_ma.iloc[-1]) or pd.isna(slow_ma.iloc[-1]):
        return 0

    if _cross_up(
        fast_ma.iloc[-2], slow_ma.iloc[-2], fast_ma.iloc[-1], slow_ma.iloc[-1]
    ):
        return 1
    if _cross_down(
        fast_ma.iloc[-2], slow_ma.iloc[-2], fast_ma.iloc[-1], slow_ma.iloc[-1]
    ):
        return -1
    return 0


def etf_trend_regime(
    data: pd.DataFrame, fast: int = 20, slow: int = 60, regime: int = 250
) -> int:
    """长期趋势过滤的双均线交叉（主策略）。

    入场：金叉 且 价格在 regime 均线之上（长期上升趋势）；
    离场：死叉（趋势破坏，无论价格在 regime 均线的哪一侧）。

    regime 过滤只约束入场、不约束离场：离场跟趋势本身走，避免在
    "长期均线之上但快速下跌" 的初期因为没到死叉而错过离场窗口。

    Args:
        data: 截至前一日的行情，至少需要 'close' 列
        fast: 快均线窗口
        slow: 慢均线窗口
        regime: 长期趋势过滤窗口（约一年交易日的均线）

    Returns:
        1(金叉且长期趋势向上) / -1(死叉) / 0(不操作)
    """
    if len(data) < max(slow, regime) + 1:
        return 0

    close = data["close"]
    fast_ma = close.rolling(fast).mean()
    slow_ma = close.rolling(slow).mean()
    regime_ma = close.rolling(regime).mean()

    if (
        pd.isna(fast_ma.iloc[-1])
        or pd.isna(slow_ma.iloc[-1])
        or pd.isna(regime_ma.iloc[-1])
    ):
        return 0

    if _cross_up(
        fast_ma.iloc[-2], slow_ma.iloc[-2], fast_ma.iloc[-1], slow_ma.iloc[-1]
    ):
        return 1 if close.iloc[-1] > regime_ma.iloc[-1] else 0
    if _cross_down(
        fast_ma.iloc[-2], slow_ma.iloc[-2], fast_ma.iloc[-1], slow_ma.iloc[-1]
    ):
        return -1
    return 0


def _true_range(df: pd.DataFrame) -> pd.Series:
    """经典真实波幅 TR = max(H-L, |H-prevC|, |L-prevC|)。"""
    prev_close = df["close"].shift(1)
    return pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def etf_donchian_atr(
    data: pd.DataFrame,
    entry: int = 55,
    exit_lookback: int = 20,
    atr_period: int = 14,
    atr_mult: float = 3.0,
    regime: int = 250,
) -> int:
    """唐奇安通道突破 + ATR 跟踪止损（海龟式变体）。

    入场：收盘价突破前 ``entry`` 日最高价，且价格在 regime 均线之上
    （长期趋势过滤，同 ``etf_trend_regime``）；
    离场：收盘价跌破前 ``exit_lookback`` 日最低价（通道下轨），或
    跌破 ATR 跟踪止损位。

    跟踪止损是**无状态近似**：引擎每次只把 ``data.iloc[:i]`` 切片传给
    策略、不传持仓状态，因此止损位取可见切片上前 ``entry`` 日的滚动收盘
    峰值回撤 ``atr_mult`` 倍 ATR，而非"入场价以来"的峰值。

    Args:
        data: 截至前一日的行情，至少需要 'high'/'low'/'close' 列
        entry: 突破通道窗口（前 N 日最高价）
        exit_lookback: 离场通道窗口（前 N 日最低价）
        atr_period: ATR 计算窗口
        atr_mult: 跟踪止损的 ATR 倍数
        regime: 长期趋势过滤窗口

    Returns:
        1(突破入场) / -1(跌破下轨或跟踪止损) / 0(不操作)
    """
    if len(data) < max(entry, regime, atr_period) + 2:
        return 0

    df = data.copy()
    close = df["close"]

    entry_high = df["high"].shift(1).rolling(entry).max()
    exit_low = df["low"].shift(1).rolling(exit_lookback).min()
    regime_ma = close.rolling(regime).mean()

    atr = _true_range(df).rolling(atr_period).mean()
    # 截至前一日的滚动收盘峰值，回撤 atr_mult 倍 ATR 即止损
    peak = close.rolling(entry).max().shift(1)
    atr_pct = atr / close
    trailing_stop = peak * (1 - atr_mult * atr_pct)

    if (
        pd.isna(entry_high.iloc[-1])
        or pd.isna(exit_low.iloc[-1])
        or pd.isna(regime_ma.iloc[-1])
        or pd.isna(trailing_stop.iloc[-1])
    ):
        return 0

    if close.iloc[-1] > entry_high.iloc[-1] and close.iloc[-1] > regime_ma.iloc[-1]:
        return 1
    if close.iloc[-1] < exit_low.iloc[-1] or close.iloc[-1] < trailing_stop.iloc[-1]:
        return -1
    return 0
