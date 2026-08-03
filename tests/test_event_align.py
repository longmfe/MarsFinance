# -*- coding: utf-8 -*-
"""事件对齐与成交可行性的正确性测试。

全部使用合成数据：不联网、不依赖 akshare、不依赖 xtquant。
"""

import datetime

import pandas as pd

from data_loader.event_align import (
    attach_yjyg_columns,
    attach_yjyg_to_universe,
    compute_tradability,
    limit_rate,
)

CODE = "600000.SH"


# --------------------------------------------------------------------------
# 锚定规则
# --------------------------------------------------------------------------


def test_anchor_is_last_bar_on_or_before_notice(make_prices, make_events):
    """公告发生在交易日当天时，锚点就是该 bar，yjyg_age 为 0。"""
    prices = make_prices(n=60)
    notice = prices.index[20]
    attached = attach_yjyg_columns(prices, make_events(notice), CODE)

    assert attached["yjyg_age"].iloc[20] == 0
    assert attached["yjyg_age"].iloc[21] == 1
    assert attached["yjyg_notice"].iloc[20] == notice


def test_notice_on_weekend_anchors_to_prior_bar(make_prices, make_events):
    """公告落在周末时锚定到上一个交易日，且该 bar 的日期早于公告日。"""
    prices = make_prices(n=60)
    friday_pos = next(
        i for i in range(20, 40) if pd.Timestamp(prices.index[i]).dayofweek == 4
    )
    saturday = (
        pd.Timestamp(prices.index[friday_pos]).date() + datetime.timedelta(days=1)
    ).strftime("%Y%m%d")
    attached = attach_yjyg_columns(prices, make_events(saturday), CODE)

    # 锚点是周五（日期早于公告日）——这不是泄漏，因为成交发生在下一根 bar
    assert attached["yjyg_age"].iloc[friday_pos] == 0
    assert attached.index[friday_pos] < saturday
    assert attached.index[friday_pos + 1] > saturday


def test_all_yjyg_columns_nan_before_first_event(make_prices, make_events):
    """首个事件之前，任何一列事件数据都不得有值。"""
    prices = make_prices(n=60)
    attached = attach_yjyg_columns(prices, make_events(prices.index[30]), CODE)

    before = attached.iloc[:30]
    for col in ("yjyg_age", "yjyg_type", "yjyg_amp", "yjyg_notice", "yjyg_period"):
        assert before[col].isna().all(), f"{col} 在首个事件前不应有值"


def test_event_before_price_window_is_dropped_not_clamped(make_prices, make_events):
    """公告早于行情起点的事件必须丢弃，绝不能夹到首根 bar 上。"""
    prices = make_prices(n=60, start="20240102")
    attached = attach_yjyg_columns(prices, make_events("20200101"), CODE)

    assert attached["yjyg_age"].isna().all()
    assert attached["yjyg_type"].isna().all()


def test_unsorted_index_raises(make_prices, make_events):
    """索引乱序会静默锚错 bar，必须直接报错而不是继续。"""
    prices = make_prices(n=60).iloc[::-1]
    try:
        attach_yjyg_columns(prices, make_events("20240201"), CODE)
    except ValueError as exc:
        assert "升序" in str(exc)
    else:
        raise AssertionError("乱序索引应当抛出 ValueError")


# --------------------------------------------------------------------------
# 涨跌停与成交可行性
# --------------------------------------------------------------------------


def test_chinext_limit_rate_switches_on_20200824():
    """创业板涨跌幅在 2020-08-24 由 10% 改为 20%。"""
    assert limit_rate("300750.SZ", "20200821") == 0.10
    assert limit_rate("300750.SZ", "20200824") == 0.20
    assert limit_rate("301001.SZ", "20240101") == 0.20


def test_limit_rate_by_board():
    """各板块涨跌幅。"""
    assert limit_rate("600000.SH", "20240101") == 0.10
    assert limit_rate("000001.SZ", "20240101") == 0.10
    assert limit_rate("688111.SH", "20240101") == 0.20
    assert limit_rate("920001.BJ", "20240101") == 0.30


def test_sealed_limit_up_is_detected(make_prices, seal_limit_up):
    """low 也钉在涨停价上 = 封死一字板，不可买入。"""
    prices = seal_limit_up(make_prices(n=10), 5)
    trad = compute_tradability(prices, CODE)

    assert bool(trad["locked_up"].iloc[5])
    assert not bool(trad["buy_ok"].iloc[5])
    assert bool(trad["buy_ok"].iloc[4])


def test_intraday_dip_below_limit_is_still_fillable(make_prices):
    """当日曾跌破涨停价，说明该价位真实成交过，按收盘价成交是可实现的。"""
    prices = make_prices(n=10)
    pos = 5
    pre = prices["preClose"].iloc[pos]
    for col, value in (
        ("close", pre * 1.10),
        ("high", pre * 1.10),
        ("low", pre * 1.05),  # 曾跌破涨停价
    ):
        prices.iloc[pos, prices.columns.get_loc(col)] = value

    trad = compute_tradability(prices, CODE)
    assert not bool(trad["locked_up"].iloc[pos])
    assert bool(trad["buy_ok"].iloc[pos])


def test_halted_bar_not_fillable(make_prices):
    """零成交量视为停牌，不可成交。"""
    prices = make_prices(n=10)
    prices.iloc[5, prices.columns.get_loc("volume")] = 0.0

    trad = compute_tradability(prices, CODE)
    assert bool(trad["halted"].iloc[5])
    assert not bool(trad["buy_ok"].iloc[5])
    assert not bool(trad["sell_ok"].iloc[5])


def test_fill_ok_next_is_shifted_and_last_bar_is_false(make_prices, seal_limit_up):
    """yjyg_fill_ok_next 描述的是下一根 bar；末根 bar 未知，保守取 False。"""
    prices = seal_limit_up(make_prices(n=10), 6)
    attached = attach_yjyg_columns(prices, None, CODE)

    assert not bool(attached["yjyg_fill_ok_next"].iloc[5])  # 下一根(6)封板
    assert bool(attached["yjyg_fill_ok_next"].iloc[4])
    assert not bool(attached["yjyg_fill_ok_next"].iloc[-1])


# --------------------------------------------------------------------------
# 全池附加与诊断
# --------------------------------------------------------------------------


def test_universe_diagnostics_count_out_of_universe_events(make_prices, make_events):
    """池外事件计数就是幸存者偏差的量化值。"""
    prices = make_prices(n=60)
    events = pd.concat(
        [
            make_events(prices.index[10], code="600000.SH"),
            make_events(prices.index[10], code="000002.SZ"),
        ],
        ignore_index=True,
    )

    attached, diag = attach_yjyg_to_universe({"600000.SH": prices}, events)

    assert diag["events_total"] == 2
    assert diag["events_in_universe"] == 1
    assert diag["events_not_in_universe"] == 1
    assert diag["events_anchored"] == 1
    assert "yjyg_age" in attached["600000.SH"].columns
