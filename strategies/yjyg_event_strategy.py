# -*- coding: utf-8 -*-
"""业绩预告事件驱动策略。

信号约定：1 = 买入，-1 = 卖出，0 = 不操作。

依赖 ``data_loader.event_align.attach_yjyg_columns`` 附加的 ``yjyg_*`` 列。
整个持仓生命周期都编码在 ``yjyg_age``（距锚定 bar 的交易日数）里，
因此函数本身无状态，符合引擎 ``strategy_function(data) -> int`` 的契约。
"""

from typing import Tuple

import pandas as pd

# 涉及亏损的预告类型：基数为负，百分比变动在数学上无意义
# （实测 扭亏 最小值 +100.5 纯属算术假象，首亏 到 -22072，增亏 到 -38333），
# 故对这些类型不施加幅度阈值，仅按类型判别。
LOSS_LIKE_TYPES = frozenset({"扭亏", "首亏", "增亏", "减亏", "不确定"})

_REQUIRED_COLUMNS = ("yjyg_age", "yjyg_type", "yjyg_amp", "yjyg_fill_ok_next")


def amp_ok(forecast_type: str, amp, min_amp: float) -> bool:
    """判断业绩变动幅度是否达标（仅对盈利→盈利类型有效）。"""
    if forecast_type in LOSS_LIKE_TYPES:
        return True
    if amp is None or pd.isna(amp):
        return False
    return float(amp) >= min_amp


def yjyg_event_strategy(
    data: pd.DataFrame,
    holding_days: int = 10,
    entry_window: int = 3,
    min_amp: float = 50.0,
    allowed_types: Tuple[str, ...] = ("预增", "扭亏"),
) -> int:
    """业绩预告事件驱动策略。

    公告落地后的次日买入，持有 holding_days 个交易日后卖出。
    若买入当日封死一字涨停（买不进），在 entry_window 内逐日重试；
    超出窗口则放弃本次事件。

    Args:
        data: 截至前一日的行情，需含 event_align 附加的 yjyg_* 列
        holding_days: 持有交易日数
        entry_window: 允许延迟建仓的交易日数（应对一字板买不进）
        min_amp: 业绩变动幅度阈值（%），仅对盈利类预告生效
        allowed_types: 允许交易的预告类型

    Returns:
        交易信号: 1(买入), -1(卖出), 0(持有)
    """
    if len(data) == 0:
        return 0

    # 未附加事件列时安全退化为不操作，可与普通行情共用
    if any(col not in data.columns for col in _REQUIRED_COLUMNS):
        return 0

    row = data.iloc[-1]
    age = row["yjyg_age"]

    # 尚未发生任何事件：保持空仓
    if pd.isna(age):
        return -1

    qualifies = row["yjyg_type"] in allowed_types and amp_ok(
        row["yjyg_type"], row["yjyg_amp"], min_amp
    )

    # 建仓（含一字板延迟重试）。fill_ok_next 只做否决：它只能把 1 变成 0，
    # 永远不能把 0 变成 1，因此不可能凭空制造收益，只能剔除不可能的成交。
    if qualifies and 0 <= age <= entry_window and bool(row["yjyg_fill_ok_next"]):
        return 1

    # 持有窗口内
    if qualifies and age < holding_days:
        return 0

    # 窗口到期，或新事件不合格（如持仓期间落地一份预减修正）——离场。
    # 空仓时引擎会忽略 -1，因此这也是安全的默认返回。
    return -1
