# -*- coding: utf-8 -*-
"""业绩预告策略的端到端测试——全部跑在真实的 StockBacktest 引擎上。

核心是 INVARIANT L：每一笔买入的**成交日**都必须严格晚于公告日。
"""

import pandas as pd

from backtest.stock_backtest import StockBacktest
from data_loader.event_align import attach_yjyg_columns
from strategies.yjyg_event_strategy import yjyg_event_strategy

CODE = "600000.SH"


def _run(attached, holding_days=10, entry_window=3, enable_stop=False):
    """在真实引擎上跑一遍，返回成交流水。"""
    engine = StockBacktest(initial_capital=100000)
    engine.run_backtest(
        attached,
        lambda x: yjyg_event_strategy(
            x, holding_days=holding_days, entry_window=entry_window
        ),
        CODE,
        enable_stop=enable_stop,
    )
    return engine.trades


def _buys(trades):
    return [t for t in trades if t["type"] == "BUY"]


def _sells(trades):
    return [t for t in trades if t["type"] == "SELL"]


# --------------------------------------------------------------------------
# 未来函数
# --------------------------------------------------------------------------


def test_fill_date_strictly_after_notice(make_prices, make_events):
    """INVARIANT L：每笔买入的成交日严格晚于公告日。本套测试的承重墙。"""
    prices = make_prices(n=60)
    notice = prices.index[20]
    attached = attach_yjyg_columns(prices, make_events(notice), CODE)

    trades = _run(attached)
    buys = _buys(trades)
    assert len(buys) == 1

    for trade in buys:
        i = attached.index.get_loc(trade["date"])
        assert attached.index[i] > attached["yjyg_notice"].iloc[i - 1]
        assert trade["date"] > notice


def test_perturbing_notice_shifts_fill_by_same_amount(make_prices, make_events):
    """把公告日推后 5 个交易日，成交日必须整整推后 5 根 bar，且绝不提前。

    这条比 INVARIANT L 更强：它证明了因果方向，能抓出单纯的差一错误。
    """
    prices = make_prices(n=60)
    base_pos = 20

    first = _buys(
        _run(attach_yjyg_columns(prices, make_events(prices.index[base_pos]), CODE))
    )
    later = _buys(
        _run(attach_yjyg_columns(prices, make_events(prices.index[base_pos + 5]), CODE))
    )

    assert len(first) == 1 and len(later) == 1
    i_first = prices.index.get_loc(first[0]["date"])
    i_later = prices.index.get_loc(later[0]["date"])
    assert i_later - i_first == 5


def test_no_trades_when_event_precedes_price_window(make_prices, make_events):
    """公告早于行情起点 -> 事件被丢弃 -> 不应有任何成交。"""
    prices = make_prices(n=60, start="20240102")
    attached = attach_yjyg_columns(prices, make_events("20200101"), CODE)
    assert _run(attached) == []


# --------------------------------------------------------------------------
# 成交可行性（一字板）
# --------------------------------------------------------------------------


def test_sealed_limit_up_defers_entry(make_prices, make_events, seal_limit_up):
    """本应成交的那根 bar 封死一字板时不成交，并在入场窗口内延迟重试。"""
    prices = make_prices(n=60)
    notice_pos = 20
    prices = seal_limit_up(prices, notice_pos + 1)  # 本应成交的 bar 封板

    attached = attach_yjyg_columns(prices, make_events(prices.index[notice_pos]), CODE)
    buys = _buys(_run(attached))

    assert len(buys) == 1
    sealed_date = prices.index[notice_pos + 1]
    assert buys[0]["date"] != sealed_date, "封死的一字板不应成交"
    assert buys[0]["date"] > sealed_date, "应在入场窗口内延迟重试"


def test_veto_only_forcing_fill_ok_false_yields_zero_trades(make_prices, make_events):
    """把 fill_ok_next 全部置 False 后必须零成交——证明该列只做否决。"""
    prices = make_prices(n=60)
    attached = attach_yjyg_columns(prices, make_events(prices.index[20]), CODE)
    attached["yjyg_fill_ok_next"] = False

    assert _run(attached) == []


def test_halted_fill_bar_defers_entry(make_prices, make_events):
    """停牌（零成交量）的 bar 不可成交。"""
    prices = make_prices(n=60)
    notice_pos = 20
    prices.iloc[notice_pos + 1, prices.columns.get_loc("volume")] = 0.0

    attached = attach_yjyg_columns(prices, make_events(prices.index[notice_pos]), CODE)
    buys = _buys(_run(attached))

    assert len(buys) == 1
    assert buys[0]["date"] != prices.index[notice_pos + 1]


# --------------------------------------------------------------------------
# 持仓生命周期与契约
# --------------------------------------------------------------------------


def test_holding_period_is_exactly_holding_days(make_prices, make_events):
    """买入与卖出相隔恰好 holding_days 根 bar。"""
    prices = make_prices(n=60)
    attached = attach_yjyg_columns(prices, make_events(prices.index[20]), CODE)

    trades = _run(attached, holding_days=10)
    buys, sells = _buys(trades), _sells(trades)
    assert len(buys) == 1 and len(sells) == 1

    i_buy = prices.index.get_loc(buys[0]["date"])
    i_sell = prices.index.get_loc(sells[0]["date"])
    assert i_sell - i_buy == 10


def test_non_qualifying_type_produces_no_trades(make_prices, make_events):
    """预减不在默认允许类型内，不应成交。"""
    prices = make_prices(n=60)
    events = make_events(prices.index[20], forecast_type="预减", amp=-70.0)
    attached = attach_yjyg_columns(prices, events, CODE)

    assert _run(attached) == []


def test_amp_below_threshold_produces_no_trades(make_prices, make_events):
    """盈利类预告的幅度低于阈值时不成交。"""
    prices = make_prices(n=60)
    events = make_events(prices.index[20], forecast_type="预增", amp=10.0)
    attached = attach_yjyg_columns(prices, events, CODE)

    assert _run(attached) == []


def test_loss_like_type_ignores_amp_threshold(make_prices, make_events):
    """扭亏基数为负，百分比无意义，不应被幅度阈值卡掉。"""
    prices = make_prices(n=60)
    events = make_events(prices.index[20], forecast_type="扭亏", amp=1.0)
    attached = attach_yjyg_columns(prices, events, CODE)

    assert len(_buys(_run(attached))) == 1


def test_returns_plain_int_and_zero_without_yjyg_columns(make_prices):
    """未附加事件列时安全退化为 0，且返回值必须是裸 int。"""
    prices = make_prices(n=60)
    signal = yjyg_event_strategy(prices)

    assert signal == 0
    assert type(signal) is int


def test_signal_is_always_valid_int(make_prices, make_events):
    """任何一根 bar 上的返回值都只能是 1 / -1 / 0。"""
    prices = make_prices(n=60)
    attached = attach_yjyg_columns(prices, make_events(prices.index[20]), CODE)

    for i in range(1, len(attached)):
        signal = yjyg_event_strategy(attached.iloc[:i])
        assert type(signal) is int
        assert signal in (1, -1, 0)


def test_disqualifying_revision_mid_hold_exits(make_prices, make_events):
    """持仓期间落地一份不合格的修正（预减），应当离场。"""
    prices = make_prices(n=60)
    events = pd.concat(
        [
            make_events(prices.index[20], forecast_type="预增", amp=100.0),
            make_events(prices.index[24], forecast_type="预减", amp=-70.0),
        ],
        ignore_index=True,
    )
    attached = attach_yjyg_columns(prices, events, CODE)

    trades = _run(attached, holding_days=10)
    buys, sells = _buys(trades), _sells(trades)
    assert len(buys) == 1 and len(sells) == 1

    # 修正公告锚定在 bar 24，故次日(25)离场，早于原定的 21+10=31
    assert prices.index.get_loc(sells[0]["date"]) == 25
