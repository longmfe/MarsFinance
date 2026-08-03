# -*- coding: utf-8 -*-
"""报告期与交易日历。

本机没有可用的独立 A 股交易日历端点（中证指数接口失效），因此交易日历直接从
沪深300 指数自身的日期索引派生 —— 既精确又不额外消耗配额。
"""

from typing import List, Optional

import pandas as pd

# A 股法定披露截止日：年报+一季报 4/30，半年报 8/31，三季报 10/31。
# 用于把报告期映射到"最迟何时一定已公开"，仅作兜底，实际优先用公告日期。
_FILING_DEADLINES = {3: "0430", 6: "0831", 9: "1031", 12: "0430"}

_QUARTER_ENDS = ("0331", "0630", "0930", "1231")


def report_periods(
    start: str = "20091231", end: str = "20241231", annual_only: bool = False
) -> List[str]:
    """列出区间内的所有报告期（``YYYYMMDD``，升序）。

    Args:
        start: 起始报告期
        end: 结束报告期
        annual_only: 只要年报（``1231``）。Piotroski 定义在年度数据上，
            年度版复现用这个即可，也就免去了累计报表差分。

    Returns:
        list[str]: 报告期列表
    """
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    suffixes = ("1231",) if annual_only else _QUARTER_ENDS

    periods = []
    for year in range(start_ts.year, end_ts.year + 1):
        for suffix in suffixes:
            period = f"{year}{suffix}"
            if start_ts <= pd.Timestamp(period) <= end_ts:
                periods.append(period)

    return periods


def fiscal_year(period) -> int:
    """报告期所属财年。"""
    return pd.Timestamp(period).year


def quarter_of(period) -> int:
    """报告期所属季度（1-4）。"""
    return pd.Timestamp(period).quarter


def filing_deadline(period) -> pd.Timestamp:
    """报告期的法定披露截止日。

    年报（12/31）的截止日是**次年** 4/30，与一季报同期，这是 A 股的实际规则。
    仅用于没有公告日期时的保守兜底；有公告日期时一律以公告日期为准。
    """
    ts = pd.Timestamp(period)
    suffix = _FILING_DEADLINES[ts.quarter * 3 if ts.quarter < 4 else 12]
    year = ts.year + 1 if ts.quarter == 4 else ts.year
    return pd.Timestamp(f"{year}{suffix}")


def trading_days(
    start: Optional[str] = None,
    end: Optional[str] = None,
    index_symbol: str = "sh000300",
) -> pd.DatetimeIndex:
    """交易日历，由指数日线的日期索引派生。

    Args:
        start: 起始日 ``YYYYMMDD``，None 表示不设下界
        end: 结束日 ``YYYYMMDD``，None 表示不设上界
        index_symbol: 用作日历来源的指数

    Returns:
        pd.DatetimeIndex: 升序的交易日
    """
    from research.datafeed.akshare_source import fetch_index_daily

    frame = fetch_index_daily(symbol=index_symbol)
    days = pd.DatetimeIndex(pd.to_datetime(frame["date"])).sort_values()

    if start is not None:
        days = days[days >= pd.Timestamp(start)]
    if end is not None:
        days = days[days <= pd.Timestamp(end)]

    return days


def next_trading_day(date, calendar: pd.DatetimeIndex, offset: int = 0):
    """返回 >= date 的第 ``offset`` 个交易日。

    Args:
        date: 目标日期
        calendar: 升序交易日历
        offset: 0 表示当天或之后最近的交易日，1 表示再往后一个交易日

    Returns:
        pd.Timestamp | None: 越界时返回 None
    """
    target = pd.Timestamp(date)
    position = calendar.searchsorted(target, side="left") + offset

    if position >= len(calendar):
        return None
    return calendar[position]


def align_to_trading_days(dates, calendar: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """把一组日期对齐到各自之后最近的交易日，去重并升序。"""
    aligned = [next_trading_day(d, calendar) for d in dates]
    valid = [d for d in aligned if d is not None]
    return pd.DatetimeIndex(sorted(set(valid)))
