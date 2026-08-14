# -*- coding: utf-8 -*-
"""股指期货 CTA 策略：多空对称的趋势跟随。

与 ``etf_trend_strategy`` 的差别：信号 1/-1 在 ``FuturesBacktest`` 里
分别对应开多/开空（空仓时）或平仓反手（持仓时）。因此入场与离场条件
必须完全对称——多头入场条件不满足时开空，反之亦然——长期趋势过滤
同时约束多空两个方向，避免在牛市的回调里做空、熊市的反弹里做多。

信号约定与仓库一致：1 = 做多，-1 = 做空，0 = 不操作。
"""

import pandas as pd

from .etf_trend_strategy import _cross_down, _cross_up


def futures_ma_regime(
    data: pd.DataFrame, fast: int = 20, slow: int = 60, regime: int = 250
) -> int:
    """对称的均线交叉 + 长期趋势过滤（CTA 主策略）。

    - 1（开多/空翻多）：金叉 且 价格在 regime 均线之上
    - -1（开空/多翻空）：死叉 且 价格在 regime 均线之下
    - 交叉发生但价格仍在趋势过滤的错误一侧 → 0（不反转，保持原仓位）

    与 ``etf_trend_regime`` 的关键差别：离场同样被 regime 过滤。多头在
    温和回调中出现死叉、但价格仍在长期均线上方时不平仓——只有趋势真正
    翻空（死叉且跌破长期均线）才反转做空。代价是回调期间的浮亏更大，
    换来的是少被牛市里的假死叉甩下车。

    Args:
        data: 截至前一日的行情，至少需要 'close' 列
        fast: 快均线窗口
        slow: 慢均线窗口
        regime: 长期趋势过滤窗口

    Returns:
        1(金叉且趋势向上) / -1(死叉且趋势向下) / 0(不操作)
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
        return -1 if close.iloc[-1] < regime_ma.iloc[-1] else 0
    return 0


def futures_donchian(
    data: pd.DataFrame, entry: int = 55, exit_lookback: int = 20, regime: int = 250
) -> int:
    """对称唐奇安通道突破（海龟式，多空双向）。

    - 1（开多/空翻多）：收盘价突破前 ``entry`` 日最高价 且 价格在 regime
      均线之上
    - -1（开空/多翻空）：收盘价跌破前 ``exit_lookback`` 日最低价 且 价格
      在 regime 均线之下
    - 0：其余情形（含通道被突破但 regime 方向不配合）

    入场窗口（长）与离场窗口（短）刻意不对称：多头需要强突破才入场，
    离场则用更紧的通道保护浮盈——这是海龟体系的原设计。两个方向都被
    regime 过滤，保证做多只在长期多头市、做空只在长期空头市。

    Args:
        data: 截至前一日的行情，至少需要 'high'/'low'/'close' 列
        entry: 入场通道窗口（前 N 日最高价/最低价）
        exit_lookback: 离场通道窗口
        regime: 长期趋势过滤窗口

    Returns:
        1(向上突破且趋势向上) / -1(向下突破且趋势向下) / 0(不操作)
    """
    if len(data) < max(entry, regime) + 2:
        return 0

    close = data["close"]

    entry_high = data["high"].shift(1).rolling(entry).max()
    exit_low = data["low"].shift(1).rolling(exit_lookback).min()
    regime_ma = close.rolling(regime).mean()

    if (
        pd.isna(entry_high.iloc[-1])
        or pd.isna(exit_low.iloc[-1])
        or pd.isna(regime_ma.iloc[-1])
    ):
        return 0

    if close.iloc[-1] > entry_high.iloc[-1] and close.iloc[-1] > regime_ma.iloc[-1]:
        return 1
    if close.iloc[-1] < exit_low.iloc[-1] and close.iloc[-1] < regime_ma.iloc[-1]:
        return -1
    return 0
