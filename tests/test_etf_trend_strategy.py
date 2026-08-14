# -*- coding: utf-8 -*-
"""ETF 趋势策略与 ETF 数据加载器的单元测试（全离线，确定性合成数据）。"""

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from research.datafeed.akshare_source import _to_sina_etf_symbol, fetch_etf_daily
from strategies.etf_trend_strategy import (
    etf_donchian_atr,
    etf_ma_cross,
    etf_trend_regime,
)


def make_ohlcv(close, high_ratio=1.005, low_ratio=0.995):
    """由收盘价序列构造确定性 OHLCV 行情。"""
    close = np.asarray(close, dtype=float)
    return pd.DataFrame(
        {
            "open": close,
            "high": close * high_ratio,
            "low": close * low_ratio,
            "close": close,
            "volume": np.full(len(close), 1_000_000.0),
        }
    )


def first_signal(strategy, data, start, end, target):
    """在 [start, end] 内逐日前缀调用策略，返回首个命中 target 信号的时点。"""
    for t in range(start, min(end, len(data))):
        if strategy(data.iloc[: t + 1]) == target:
            return t
    return None


class TestToSinaEtfSymbol:
    def test_sh_etf(self):
        assert _to_sina_etf_symbol("510310") == "sh510310"

    def test_sz_etf(self):
        assert _to_sina_etf_symbol("159915") == "sz159915"

    def test_accepts_prefixed_input(self):
        assert _to_sina_etf_symbol("sh510310") == "sh510310"
        assert _to_sina_etf_symbol("SZ159915") == "sz159915"

    def test_rejects_stock_code(self):
        with pytest.raises(ValueError, match="未知 ETF 交易所前缀"):
            _to_sina_etf_symbol("600519")

    def test_rejects_malformed(self):
        with pytest.raises(ValueError):
            _to_sina_etf_symbol("12345")
        with pytest.raises(ValueError):
            _to_sina_etf_symbol("abc")


class TestMaCross:
    def test_buy_then_hold_then_sell(self):
        """横盘 → 上涨 → 下跌：金叉买、持有不重复买、死叉卖。"""
        close = np.concatenate(
            [
                np.full(100, 100.0),
                100.0 + 0.5 * np.arange(200),
                200.0 - 0.5 * np.arange(100),
            ]
        )
        data = make_ohlcv(close)

        buy_at = first_signal(etf_ma_cross, data, 100, 300, 1)
        assert buy_at is not None, "上涨段应出现金叉买入"

        # 金叉之后、下跌之前：继续上涨只持有，不再买入
        holding_signals = [
            etf_ma_cross(data.iloc[: t + 1]) for t in range(buy_at + 1, 300)
        ]
        assert all(s == 0 for s in holding_signals)

        sell_at = first_signal(etf_ma_cross, data, 300, 400, -1)
        assert sell_at is not None, "下跌段应出现死叉卖出"

    def test_sideways_never_trades(self):
        data = make_ohlcv(np.full(300, 100.0))
        for t in (100, 200, 299):
            assert etf_ma_cross(data.iloc[: t + 1]) == 0

    def test_short_history_returns_zero(self):
        assert etf_ma_cross(make_ohlcv(np.arange(10.0))) == 0


class TestTrendRegime:
    def test_buy_requires_regime_up(self):
        """横盘 300 天后上涨：金叉时长期均线已可用且价格在其上 → 买入。"""
        close = np.concatenate(
            [np.full(300, 100.0), 100.0 + 0.5 * np.arange(200)]
        )
        data = make_ohlcv(close)

        buy_at = first_signal(etf_trend_regime, data, 300, 500, 1)
        assert buy_at is not None, "长期趋势向上时应买入"

    def test_golden_cross_gated_in_downtrend_regime(self):
        """长跌后反弹：金叉成立，但价格仍在长期均线之下 → 被过滤为 0。"""
        close = np.concatenate(
            [
                np.full(300, 300.0),  # 均线先定义且相等，且越过 regime 窗口
                300.0 - 0.5 * np.arange(400),  # 300 → 100
                100.0 + 0.5 * np.arange(60),  # 反弹
            ]
        )
        data = make_ohlcv(close)

        sell_at = first_signal(etf_trend_regime, data, 300, 700, -1)
        assert sell_at is not None, "下跌段应出现死叉卖出"

        found_golden_cross = False
        for t in range(700, 760):
            base = etf_ma_cross(data.iloc[: t + 1])
            filtered = etf_trend_regime(data.iloc[: t + 1])
            if base == 1:
                found_golden_cross = True
                assert filtered == 0, "长期趋势向下时金叉必须被过滤"

        assert found_golden_cross, "反弹段应出现金叉（对照组）"

    def test_sideways_never_trades(self):
        data = make_ohlcv(np.full(400, 100.0))
        for t in (300, 399):
            assert etf_trend_regime(data.iloc[: t + 1]) == 0

    def test_short_history_returns_zero(self):
        assert etf_trend_regime(make_ohlcv(np.arange(100.0))) == 0


class TestDonchianAtr:
    def test_breakout_buy_and_exit_on_decline(self):
        """长期上涨突破通道上轨 → 买入；随后下跌 → 离场。"""
        up = 100.0 * (1.012 ** np.arange(400))
        down = up[-1] * (0.988 ** np.arange(1, 61))
        data = make_ohlcv(np.concatenate([up, down]), high_ratio=1.002, low_ratio=0.998)

        buy_at = first_signal(etf_donchian_atr, data, 250, 400, 1)
        assert buy_at is not None, "突破前 55 日最高价且趋势向上应买入"

        sell_at = first_signal(etf_donchian_atr, data, 400, 460, -1)
        assert sell_at is not None, "下跌段应触发通道下轨或 ATR 跟踪止损"

    def test_sideways_never_trades(self):
        data = make_ohlcv(np.full(400, 100.0))
        for t in (300, 399):
            assert etf_donchian_atr(data.iloc[: t + 1]) == 0

    def test_short_history_returns_zero(self):
        assert etf_donchian_atr(make_ohlcv(np.arange(100.0))) == 0


class TestNoInputMutation:
    def test_strategies_do_not_mutate_input(self):
        close = np.concatenate(
            [np.full(300, 100.0), 100.0 + 0.5 * np.arange(100)]
        )
        data = make_ohlcv(close)

        for strategy in (etf_ma_cross, etf_trend_regime, etf_donchian_atr):
            before = data.copy(deep=True)
            strategy(data.iloc[:350])
            pd.testing.assert_frame_equal(data, before)


class TestFetchEtfDaily:
    def _fake_sina_fetch(self, calls):
        """伪造新浪 ETF 端点：date 为 datetime.date、顺序倒序。"""
        import akshare

        def fake(symbol):
            calls.append(symbol)
            return pd.DataFrame(
                {
                    "date": [
                        dt.date(2024, 1, 3),
                        dt.date(2024, 1, 2),
                        dt.date(2024, 1, 1),
                    ],
                    "open": [3.1, 2.1, 1.1],
                    "high": [3.2, 2.2, 1.2],
                    "low": [2.9, 1.9, 0.9],
                    "close": [3.0, 2.0, 1.0],
                    "volume": [300.0, 200.0, 100.0],
                }
            )

        return fake

    def test_fetch_cached_and_normalized(self, tmp_cache, monkeypatch):
        import akshare

        calls = []
        monkeypatch.setattr(
            akshare, "fund_etf_hist_sina", self._fake_sina_fetch(calls)
        )

        first = fetch_etf_daily("510310")
        second = fetch_etf_daily("510310")

        assert len(calls) == 1, "第二次必须命中缓存"
        assert calls[0] == "sh510310"

        for out in (first, second):
            assert pd.api.types.is_datetime64_any_dtype(out["date"])
            assert out["date"].is_monotonic_increasing, "必须按日期升序"
            assert out["close"].tolist() == [1.0, 2.0, 3.0]
            assert list(out.columns) == [
                "date",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]

    def test_different_code_fetches_separately(self, tmp_cache, monkeypatch):
        import akshare

        calls = []
        monkeypatch.setattr(
            akshare, "fund_etf_hist_sina", self._fake_sina_fetch(calls)
        )

        fetch_etf_daily("510310")
        fetch_etf_daily("159915")

        assert calls == ["sh510310", "sz159915"]

    def test_empty_result_passes_through_and_is_cached(self, tmp_cache, monkeypatch):
        import akshare

        calls = []

        def fake_empty(symbol):
            calls.append(symbol)
            return pd.DataFrame()

        monkeypatch.setattr(akshare, "fund_etf_hist_sina", fake_empty)

        first = fetch_etf_daily("510310")
        second = fetch_etf_daily("510310")

        assert first.empty and second.empty
        assert len(calls) == 1
