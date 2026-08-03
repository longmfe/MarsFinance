# -*- coding: utf-8 -*-
"""截面分组回测引擎的单元测试（全离线、构造性数据）。

这个引擎是全新代码，且它的 bug 最不容易被肉眼发现（错误的回测只会更好看），
所以这里用"答案已知"的构造数据来钉住行为。
"""

import numpy as np
import pandas as pd
import pytest

from research.ffscore.backtest import CrossSectionalBacktest, signal_efficacy


def make_world(
    n_days=120, codes=("A", "B", "C", "D"), drifts=(0.004, 0.002, 0.0, -0.002)
):
    """构造确定性行情：每只股票以固定日收益率增长。"""
    dates = pd.bdate_range("2020-01-01", periods=n_days)

    frames = []
    for code, drift in zip(codes, drifts):
        close = 100.0 * (1.0 + drift) ** np.arange(n_days)
        frames.append(
            pd.DataFrame(
                {
                    "close": close,
                    "volume": 1e6,
                    "outstanding_share": 1e8,
                },
                index=pd.MultiIndex.from_arrays(
                    [dates, [code] * n_days], names=["date", "code"]
                ),
            )
        )

    return pd.concat(frames).sort_index()


def make_factor(price_panel, values, dates=None):
    """在给定日期上构造截面因子。"""
    all_dates = price_panel.index.get_level_values("date").unique()
    dates = dates if dates is not None else [all_dates[0], all_dates[60]]

    records = {}
    for date in dates:
        for code, val in values.items():
            records[(pd.Timestamp(date), code)] = float(val)

    series = pd.Series(records)
    series.index = pd.MultiIndex.from_tuples(series.index, names=["date", "code"])
    return series.sort_index()


class TestAssignGroups:
    def test_score_value_buckets_by_integer(self):
        engine = CrossSectionalBacktest(group_by="score_value")
        factor = make_factor(make_world(), {"A": 9, "B": 9, "C": 2, "D": 0})
        groups = engine.assign_groups(factor)

        first = groups.xs(groups.index.get_level_values("date")[0], level="date")
        assert first["A"] == "9"
        assert first["B"] == "9"
        assert first["C"] == "2"
        assert first["D"] == "0"

    def test_score_value_keeps_ties_together(self):
        """整数因子的并列必须留在同一组 —— 这正是不用分位数的理由。"""
        engine = CrossSectionalBacktest(group_by="score_value")
        factor = make_factor(make_world(), {"A": 8, "B": 8, "C": 8, "D": 1})
        groups = engine.assign_groups(factor)

        first = groups.xs(groups.index.get_level_values("date")[0], level="date")
        assert first[["A", "B", "C"]].nunique() == 1

    def test_quantile_splits_into_n_groups(self):
        engine = CrossSectionalBacktest(group_by="quantile", n_groups=4)
        factor = make_factor(make_world(), {"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0})
        groups = engine.assign_groups(factor)

        first = groups.xs(groups.index.get_level_values("date")[0], level="date")
        assert first.nunique() == 4
        assert first["D"] == "G1"  # 最低分进第一组
        assert first["A"] == "G4"

    def test_rejects_bad_group_by(self):
        with pytest.raises(ValueError, match="group_by"):
            CrossSectionalBacktest(group_by="nonsense")

    def test_rejects_bad_weighting(self):
        with pytest.raises(ValueError, match="weighting"):
            CrossSectionalBacktest(weighting="nonsense")


class TestMonotonicity:
    def test_perfect_factor_orders_groups(self):
        """因子完全预测收益时，高分组必须跑赢低分组。"""
        prices = make_world()
        factor = make_factor(prices, {"A": 9, "B": 6, "C": 3, "D": 0})

        engine = CrossSectionalBacktest(commission=0.0, slippage=0.0, stamp_tax=0.0)
        result = engine.run(factor, prices)

        nav = result["group_nav"].iloc[-1]
        assert nav["9"] > nav["6"] > nav["3"] > nav["0"]

    def test_long_short_is_positive_for_good_factor(self):
        prices = make_world()
        factor = make_factor(prices, {"A": 9, "B": 6, "C": 3, "D": 0})

        engine = CrossSectionalBacktest(commission=0.0, slippage=0.0, stamp_tax=0.0)
        result = engine.run(factor, prices)

        assert result["long_short_nav"].iloc[-1] > 1.0

    def test_rank_ic_is_one_for_perfect_factor(self):
        prices = make_world()
        factor = make_factor(prices, {"A": 9, "B": 6, "C": 3, "D": 0})

        engine = CrossSectionalBacktest(commission=0.0, slippage=0.0, stamp_tax=0.0)
        result = engine.run(factor, prices)

        assert result["ic"].iloc[0] == pytest.approx(1.0)

    def test_inverted_factor_gives_negative_ic(self):
        prices = make_world()
        factor = make_factor(prices, {"A": 0, "B": 3, "C": 6, "D": 9})

        engine = CrossSectionalBacktest(commission=0.0, slippage=0.0, stamp_tax=0.0)
        result = engine.run(factor, prices)

        assert result["ic"].iloc[0] == pytest.approx(-1.0)


class TestCosts:
    def test_cost_difference_is_analytic(self):
        """零成本净值与含成本净值之差，必须精确等于换手 × 费率。"""
        prices = make_world()
        factor = make_factor(prices, {"A": 9, "B": 6, "C": 3, "D": 0})

        free = CrossSectionalBacktest(commission=0.0, slippage=0.0, stamp_tax=0.0)
        costed = CrossSectionalBacktest(
            commission=0.001, slippage=0.001, stamp_tax=0.001
        )

        free_result = free.run(factor, prices)
        costed_result = costed.run(factor, prices)

        turnover = costed_result["turnover"]
        group9 = turnover[turnover["group"] == "9"]

        # 每次调仓的成本 = 单边换手 × (2×(佣金+滑点) + 印花税)
        expected = group9["turnover"] * (2 * (0.001 + 0.001) + 0.001)
        assert group9["cost"].tolist() == pytest.approx(expected.tolist())

        # 含成本净值必然更低
        assert (
            costed_result["group_nav"]["9"].iloc[-1]
            < free_result["group_nav"]["9"].iloc[-1]
        )

    def test_initial_build_is_full_turnover(self):
        prices = make_world()
        factor = make_factor(prices, {"A": 9, "B": 6, "C": 3, "D": 0})

        result = CrossSectionalBacktest().run(factor, prices)
        first = result["turnover"].iloc[0]

        assert first["turnover"] == pytest.approx(1.0), "建仓是 100% 换手"

    def test_unchanged_holdings_have_low_turnover(self):
        """成分不变时，换手只来自价格漂移，应当远小于 1。"""
        prices = make_world()
        factor = make_factor(prices, {"A": 9, "B": 9, "C": 0, "D": 0})

        result = CrossSectionalBacktest().run(factor, prices)
        turnover = result["turnover"]

        # 按调仓日筛掉建仓那一次，而不是按行号 —— turnover 表里各组是交错排列的
        group9 = turnover[turnover["group"] == "9"]
        first_date = group9["date"].min()
        later = group9[group9["date"] > first_date]

        assert len(later) > 0
        assert (later["turnover"] < 0.2).all()


class TestSuspensionAndMissingData:
    def test_suspended_day_yields_zero_return_not_nan(self):
        """停牌日没有行情 → 当日收益记 0，不能污染整条净值。"""
        prices = make_world()

        # 制造停牌：抹掉 B 在中间一段的行情
        dates = prices.index.get_level_values("date").unique()
        drop = [(d, "B") for d in dates[20:25]]
        prices = prices.drop(index=drop)

        factor = make_factor(prices, {"A": 9, "B": 9, "C": 0, "D": 0})
        result = CrossSectionalBacktest().run(factor, prices)

        assert result["group_returns"].notna().all().all()
        assert np.isfinite(result["group_nav"].iloc[-1]).all()

    def test_code_absent_from_prices_is_skipped(self):
        prices = make_world(codes=("A", "B"), drifts=(0.004, -0.002))
        factor = make_factor(prices, {"A": 9, "B": 0, "ZZZZ": 5})

        result = CrossSectionalBacktest().run(factor, prices)
        assert np.isfinite(result["group_nav"].iloc[-1]).all()

    def test_empty_factor_raises(self):
        prices = make_world()
        empty = pd.Series(
            dtype=float,
            index=pd.MultiIndex.from_arrays([[], []], names=["date", "code"]),
        )
        with pytest.raises(ValueError, match="为空"):
            CrossSectionalBacktest().run(empty, prices)


class TestUniverseMask:
    def test_mask_excludes_stocks(self):
        prices = make_world()
        factor = make_factor(prices, {"A": 9, "B": 6, "C": 3, "D": 0})

        mask = pd.Series(True, index=factor.index)
        mask[mask.index.get_level_values("code") == "A"] = False

        result = CrossSectionalBacktest().run(factor, prices, universe_mask=mask)
        assert "9" not in result["group_nav"].columns


class TestExecutionLag:
    def test_lag_shifts_start_of_holding(self):
        """execution_lag=1 表示次日成交，与信号日隔开一天。"""
        prices = make_world()
        factor = make_factor(prices, {"A": 9, "B": 0})

        immediate = CrossSectionalBacktest(
            execution_lag=0, commission=0.0, slippage=0.0, stamp_tax=0.0
        ).run(factor, prices)
        delayed = CrossSectionalBacktest(
            execution_lag=1, commission=0.0, slippage=0.0, stamp_tax=0.0
        ).run(factor, prices)

        assert len(delayed["group_returns"]) < len(immediate["group_returns"])


class TestWeighting:
    def test_equal_weight_is_mean_of_constituents(self):
        """等权买入持有的净值 = 各成分累计净值的算术平均。"""
        prices = make_world(codes=("A", "B"), drifts=(0.01, 0.0))
        factor = make_factor(
            prices,
            {"A": 5, "B": 5},
            dates=[prices.index.get_level_values("date").unique()[0]],
        )

        result = CrossSectionalBacktest(
            commission=0.0, slippage=0.0, stamp_tax=0.0
        ).run(factor, prices)

        nav = result["group_nav"]["5"]
        n_days = len(nav)
        expected = (1.01**n_days + 1.0) / 2.0

        assert nav.iloc[-1] == pytest.approx(expected, rel=1e-6)

    def test_cap_weighting_runs(self):
        prices = make_world()
        factor = make_factor(prices, {"A": 9, "B": 6, "C": 3, "D": 0})

        result = CrossSectionalBacktest(weighting="cap").run(factor, prices)
        assert np.isfinite(result["group_nav"].iloc[-1]).all()


class TestSignalEfficacy:
    def test_perfect_signal_has_positive_spread(self):
        prices = make_world()
        dates = prices.index.get_level_values("date").unique()

        records = {}
        for date in [dates[0], dates[60]]:
            for code, flag in {"A": 1.0, "B": 1.0, "C": 0.0, "D": 0.0}.items():
                records[(date, code)] = flag

        signals = pd.DataFrame({"f1_roa_positive": pd.Series(records)})
        signals.index = pd.MultiIndex.from_tuples(signals.index, names=["date", "code"])

        table = signal_efficacy(signals, prices, ["f1_roa_positive"])

        assert table.loc["f1_roa_positive", "mean_spread"] > 0
        assert table.loc["f1_roa_positive", "n_periods"] >= 1
