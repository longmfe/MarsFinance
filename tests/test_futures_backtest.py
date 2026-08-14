# -*- coding: utf-8 -*-
"""股指期货 CTA 引擎与期货数据加载器的单元测试（全离线，确定性数据）。"""

import datetime as dt

import pandas as pd
import pytest

from backtest.futures_backtest import FuturesBacktest
from research.datafeed.akshare_source import (
    _validate_futures_symbol,
    fetch_futures_daily,
)

# IF 合约参数
MULTIPLIER = 300
MARGIN_RATE = 0.12
COMMISSION_RATE = 0.000023
PRICE = 4000.0
NOTIONAL = PRICE * MULTIPLIER  # 每手名义价值 1,200,000
MARGIN_PER_LOT = NOTIONAL * MARGIN_RATE  # 每手保证金 144,000
CAPITAL = 2_000_000
EXPECTED_LOTS = CAPITAL // MARGIN_PER_LOT  # 13 手


def make_data(prices, dates=None):
    """构造带 date 列的日线 OHLC。"""
    if dates is None:
        dates = pd.bdate_range("2024-01-01", periods=len(prices))
    return pd.DataFrame(
        {
            "date": dates,
            "open": prices,
            "high": [p * 1.001 for p in prices],
            "low": [p * 0.999 for p in prices],
            "close": prices,
            "volume": [10000] * len(prices),
        }
    )


def signal_sequence(*values):
    """按前缀长度取信号的脚本化策略：前缀长度为 i+1 时返回 values[i]。

    引擎第 i 日调用策略时前缀长度为 i，因此 values[0] 是第 1 个交易日的信号。
    """

    def strategy(data):
        idx = len(data) - 1
        return values[idx] if idx < len(values) else 0

    return strategy


def make_backtest(**kwargs):
    params = dict(
        initial_capital=CAPITAL,
        multiplier=MULTIPLIER,
        margin_rate=MARGIN_RATE,
        commission_rate=COMMISSION_RATE,
        slippage=0.0,
        annual_vol_target=None,  # 关闭波动率目标，保证金上限直接决定手数
    )
    params.update(kwargs)
    return FuturesBacktest(**params)


class TestLongShortPnl:
    def test_long_profit(self):
        """无摩擦多单：4000 开 13 手，4100 平 → 盈利 13×100×300。"""
        prices = [PRICE] * 9 + [4100.0]
        data = make_data(prices)
        # 引擎第 9 日（最后一日）执行离场信号
        strat = signal_sequence(1, 0, 0, 0, 0, 0, 0, 0, -1)

        metrics = make_backtest(commission_rate=0.0).run_backtest(
            data, strat, symbol="IF0", reverse=False
        )

        assert metrics["final_value"] == pytest.approx(
            CAPITAL + EXPECTED_LOTS * 100 * MULTIPLIER
        )
        assert metrics["total_trades"] == 1
        assert metrics["win_rate"] == 1.0
        assert metrics["long_trades"] == 1
        assert metrics["short_trades"] == 0

    def test_short_profit(self):
        """无摩擦空单：4000 开空，3900 平（reverse=False 只平不反手）。"""
        prices = [PRICE] * 9 + [3900.0]
        data = make_data(prices)
        strat = signal_sequence(-1, 0, 0, 0, 0, 0, 0, 0, 1)

        metrics = make_backtest(commission_rate=0.0).run_backtest(
            data, strat, symbol="IF0", reverse=False
        )

        assert metrics["final_value"] == pytest.approx(
            CAPITAL + EXPECTED_LOTS * 100 * MULTIPLIER
        )
        assert metrics["total_trades"] == 1
        assert metrics["short_trades"] == 1

    def test_short_loss_reduces_capital(self):
        """空单亏损：4000 开空，4100 平 → 亏 13×100×300。"""
        prices = [PRICE] * 9 + [4100.0]
        data = make_data(prices)
        strat = signal_sequence(-1, 0, 0, 0, 0, 0, 0, 0, 1)

        metrics = make_backtest(commission_rate=0.0).run_backtest(
            data, strat, symbol="IF0", reverse=False
        )

        assert metrics["final_value"] == pytest.approx(
            CAPITAL - EXPECTED_LOTS * 100 * MULTIPLIER
        )
        assert metrics["win_rate"] == 0.0


class TestMarginAndFrictions:
    def test_contract_sizing_and_slippage(self):
        """保证金向下取整开手数；滑点使多头成交价高于市价。"""
        data = make_data([PRICE] * 5)
        strat = signal_sequence(1, 0, 0, 0, 0)

        backtest = make_backtest(slippage=0.001)
        backtest.run_backtest(data, strat, symbol="IF0")

        trade = backtest.trades[0]
        assert trade["contracts"] == EXPECTED_LOTS
        assert trade["price"] == pytest.approx(PRICE * 1.001)
        assert trade["side"] == "LONG"

        # 滑点后的名义价值仍满足保证金约束
        assert (
            trade["contracts"] * trade["price"] * MULTIPLIER * MARGIN_RATE
            <= CAPITAL
        )

    def test_commission_deducted_both_sides(self):
        """佣金双边收取：开仓即扣，平仓再扣。"""
        prices = [PRICE] * 9 + [4050.0]
        data = make_data(prices)
        strat = signal_sequence(1, 0, 0, 0, 0, 0, 0, 0, -1)

        backtest = make_backtest()
        metrics = backtest.run_backtest(data, strat, symbol="IF0", reverse=False)

        trade = backtest.trades[0]
        open_comm = EXPECTED_LOTS * PRICE * MULTIPLIER * COMMISSION_RATE
        close_comm = EXPECTED_LOTS * 4050.0 * MULTIPLIER * COMMISSION_RATE
        assert trade["commission"] == pytest.approx(open_comm)

        gross_pnl = EXPECTED_LOTS * 50.0 * MULTIPLIER
        assert trade["profit"] == pytest.approx(
            gross_pnl - open_comm - close_comm
        )
        assert metrics["final_value"] == pytest.approx(
            CAPITAL + gross_pnl - open_comm - close_comm
        )

    def test_insufficient_margin_no_trade(self):
        """资金不足以开一手时不交易。"""
        data = make_data([PRICE] * 5)
        strat = signal_sequence(1, 0, 0, 0, 0)

        backtest = make_backtest(initial_capital=10_000)
        metrics = backtest.run_backtest(data, strat, symbol="IF0")

        assert backtest.trades == []
        assert metrics["final_value"] == 10_000


class TestSignalHandling:
    def test_flip_long_to_short(self):
        """多空反手：平多后同日开空。"""
        prices = [PRICE] * 8
        data = make_data(prices)
        strat = signal_sequence(1, 0, 0, 0, -1, 0, 0)

        backtest = make_backtest()
        backtest.run_backtest(data, strat, symbol="IF0")

        assert len(backtest.trades) == 2
        assert backtest.trades[0]["side"] == "LONG"
        assert "exit_date" in backtest.trades[0]
        assert backtest.trades[1]["side"] == "SHORT"
        assert backtest.position == -EXPECTED_LOTS

    def test_reverse_false_closes_only(self):
        """reverse=False：反向信号只平仓不反手。"""
        prices = [PRICE] * 8
        data = make_data(prices)
        strat = signal_sequence(1, 0, 0, 0, -1, 0, 0)

        backtest = make_backtest()
        backtest.run_backtest(data, strat, symbol="IF0", reverse=False)

        assert len(backtest.trades) == 1
        assert "exit_date" in backtest.trades[0]
        assert backtest.position == 0

    def test_allow_short_false_never_shorts(self):
        """allow_short=False：-1 在空仓时不动作，持仓时只平多。"""
        prices = [PRICE] * 10
        data = make_data(prices)
        strat = signal_sequence(-1, 1, 0, 0, -1, -1, -1, 0, 0)

        backtest = make_backtest()
        backtest.run_backtest(data, strat, symbol="IF0", allow_short=False)

        assert all(t["side"] == "LONG" for t in backtest.trades)
        assert len(backtest.trades) == 1
        assert "exit_date" in backtest.trades[0]
        assert backtest.position == 0

    def test_hold_ignores_repeat_signal(self):
        """持仓中收到同向信号不重复开仓。"""
        prices = [PRICE] * 5
        data = make_data(prices)
        strat = signal_sequence(1, 1, 1, 1, 1)

        backtest = make_backtest()
        backtest.run_backtest(data, strat, symbol="IF0")

        assert len(backtest.trades) == 1
        assert backtest.position == EXPECTED_LOTS


class TestNoLookahead:
    def test_strategy_sees_only_prefix(self):
        """引擎必须把截至前一日的切片传给策略（结构性防未来函数）。"""
        seen_lengths = []

        def spy(data):
            seen_lengths.append(len(data))
            return 0

        data = make_data([PRICE] * 20)
        make_backtest().run_backtest(data, spy, symbol="IF0")

        assert seen_lengths == list(range(1, 20))

    def test_empty_run_returns_empty_metrics(self):
        data = make_data([PRICE])
        assert make_backtest().run_backtest(data, signal_sequence(1)) == {}


class TestVolTargetSizing:
    def test_contracts_scale_with_vol_target(self):
        """波动率目标手数：ATR=40 点时 15% 年化目标 → 1 手（保证金上限 13 手）。"""
        prices = [PRICE] * 30
        # high/low 固定 ±20 点 → TR = 40 点，ATR = 40
        data = make_data(prices)
        data["high"] = [p + 20 for p in prices]
        data["low"] = [p - 20 for p in prices]
        strat = signal_sequence(1, 0)

        backtest = make_backtest(annual_vol_target=0.15, atr_period=20)
        backtest.run_backtest(data, strat, symbol="IF0")

        expected = int(
            CAPITAL * 0.15 / (40.0 * MULTIPLIER * 252**0.5)
        )
        assert backtest.trades[0]["contracts"] == expected
        assert expected == 1
        assert backtest.position == 1

    def test_margin_cap_still_binds(self):
        """波动率目标给出的手数不得超过保证金上限。"""
        prices = [PRICE] * 30
        data = make_data(prices)
        data["high"] = [p + 1 for p in prices]  # 极小 ATR → 波动率目标手数很大
        data["low"] = [p - 1 for p in prices]
        strat = signal_sequence(1, 0)

        backtest = make_backtest(annual_vol_target=0.15, atr_period=20)
        backtest.run_backtest(data, strat, symbol="IF0")

        assert backtest.trades[0]["contracts"] == EXPECTED_LOTS

    def test_missing_high_low_falls_back_to_margin_sizing(self):
        """缺少 high/low 列时退回保证金上限。"""
        prices = [PRICE] * 30
        data = pd.DataFrame(
            {
                "date": pd.bdate_range("2024-01-01", periods=len(prices)),
                "close": prices,
            }
        )
        strat = signal_sequence(1, 0)

        backtest = make_backtest(annual_vol_target=0.15)
        backtest.run_backtest(data, strat, symbol="IF0")

        assert backtest.trades[0]["contracts"] == EXPECTED_LOTS


class TestMarginCall:
    def test_forced_liquidation_on_margin_shortfall(self):
        """权益跌破持仓保证金 → 当日收盘价强平，权益不为负。"""
        prices = [PRICE] * 9 + [3950.0]
        data = make_data(prices)
        strat = signal_sequence(1, 0, 0, 0, 0, 0, 0, 0, 0)

        backtest = make_backtest(commission_rate=0.0)
        metrics = backtest.run_backtest(data, strat, symbol="IF0", reverse=False)

        # 13 手多头，4000 → 3950：亏损 13×50×300 = 195,000
        assert metrics["margin_calls"] == 1
        assert backtest.position == 0
        assert metrics["final_value"] == pytest.approx(CAPITAL - 195_000)
        assert metrics["max_drawdown"] <= 1.0

    def test_no_margin_call_within_limits(self):
        """小波动不触发强平。"""
        prices = [PRICE] * 9 + [3990.0]
        data = make_data(prices)
        strat = signal_sequence(1, 0, 0, 0, 0, 0, 0, 0, 0)

        backtest = make_backtest(commission_rate=0.0)
        metrics = backtest.run_backtest(data, strat, symbol="IF0", reverse=False)

        assert metrics["margin_calls"] == 0
        assert backtest.position == EXPECTED_LOTS


class TestValidateFuturesSymbol:
    def test_normalizes_case(self):
        assert _validate_futures_symbol("if0") == "IF0"

    def test_accepts_contract_month(self):
        assert _validate_futures_symbol("IF2508") == "IF2508"

    def test_rejects_malformed(self):
        for bad in ("IF", "123", "IF!", "IF000000", ""):
            with pytest.raises(ValueError, match="无法解析期货代码"):
                _validate_futures_symbol(bad)


class TestFetchFuturesDaily:
    def _fake_sina_fetch(self, calls):
        """伪造新浪期货端点：中文列名、datetime.date、顺序倒序。"""
        import akshare

        def fake(symbol):
            calls.append(symbol)
            return pd.DataFrame(
                {
                    "日期": [dt.date(2024, 1, 3), dt.date(2024, 1, 2), dt.date(2024, 1, 1)],
                    "开盘价": [3.1, 2.1, 1.1],
                    "最高价": [3.2, 2.2, 1.2],
                    "最低价": [2.9, 1.9, 0.9],
                    "收盘价": [3.0, 2.0, 1.0],
                    "成交量": [300.0, 200.0, 100.0],
                    "持仓量": [30.0, 20.0, 10.0],
                    "动态结算价": [3.05, 2.05, 1.05],
                }
            )

        return fake

    def test_fetch_cached_renamed_and_sorted(self, tmp_cache, monkeypatch):
        import akshare

        calls = []
        monkeypatch.setattr(
            akshare, "futures_main_sina", self._fake_sina_fetch(calls)
        )

        first = fetch_futures_daily("IF0")
        second = fetch_futures_daily("if0")

        assert len(calls) == 1, "第二次（同代码不同大小写）必须命中缓存"
        assert calls[0] == "IF0"

        for out in (first, second):
            assert list(out.columns) == [
                "date",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "hold",
                "settle",
            ]
            assert pd.api.types.is_datetime64_any_dtype(out["date"])
            assert out["date"].is_monotonic_increasing
            assert out["close"].tolist() == [1.0, 2.0, 3.0]

    def test_invalid_symbol_raises_before_fetch(self, tmp_cache, monkeypatch):
        import akshare

        calls = []
        monkeypatch.setattr(
            akshare, "futures_main_sina", self._fake_sina_fetch(calls)
        )

        with pytest.raises(ValueError):
            fetch_futures_daily("IF")

        assert calls == [], "校验失败时不得发起网络请求"
