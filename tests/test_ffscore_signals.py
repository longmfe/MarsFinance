# -*- coding: utf-8 -*-
"""九个 Piotroski 信号的逐项测试（全离线，答案手算）。

重点盯三件事：
1. 边界用严格不等号 —— 恰好为零必须判 0；
2. 缺失传播为 NaN，绝不静默当成 0；
3. 分母为零给 NaN 而不是 inf。
"""

import numpy as np
import pandas as pd
import pytest

from research.ffscore.config import default_config
from research.ffscore.score import compute_ffscore, compute_signals
from research.ffscore.signals import (
    REQUIRED_COLUMNS,
    SIGNAL_FUNCTIONS,
    SIGNAL_NAMES,
    f1_roa_positive,
    f2_cfo_positive,
    f3_delta_roa,
    f4_accrual,
    f5_delta_lever,
    f6_delta_liquid,
    f7_no_equity_offer,
    f8_delta_margin,
    f9_delta_turnover,
)

# 一只"样样都好"的股票 —— 九个信号应全部为 1
PERFECT = {
    "total_assets": 120.0,
    "total_assets_prev": 100.0,
    "total_assets_prev2": 90.0,
    "net_profit": 10.0,  # ROA = 10/100 = 0.10
    "net_profit_prev": 5.0,  # ROA_prev = 5/90 = 0.056 → 改善
    "cfo": 15.0,  # CFO/TA = 0.15 > ROA = 0.10 → 低应计
    "revenue": 200.0,  # 周转 200/100 = 2.00
    "revenue_prev": 150.0,  # 周转_prev 150/90 = 1.67 → 上升
    "operating_cost": 100.0,  # 毛利率 (200-100)/200 = 0.50
    "operating_cost_prev": 90.0,  # 毛利率_prev (150-90)/150 = 0.40 → 上升
    "total_liab": 30.0,  # 资产负债率 30/120 = 0.250
    "total_liab_prev": 40.0,  # _prev 40/100 = 0.400 → 下降
    "noncurrent_liab": 5.0,
    "noncurrent_liab_prev": 10.0,
    "current_assets": 60.0,  # 流动比率 60/20 = 3.0
    "current_liab": 20.0,
    "current_assets_prev": 50.0,  # _prev 50/25 = 2.0 → 上升
    "current_liab_prev": 25.0,
    "shares": 1000.0,  # 股本未增加
    "shares_prev": 1000.0,
}

# 一只"样样都差"的股票 —— 九个信号应全部为 0
TERRIBLE = {
    **PERFECT,
    "net_profit": -10.0,  # ROA = -0.10 < 0
    "net_profit_prev": 9.0,  # ROA_prev = 0.10 → 恶化
    "cfo": -15.0,  # CFO < 0，且 -0.15 < ROA
    "revenue": 200.0,
    "revenue_prev": 300.0,  # 周转 2.00 < 3.33 → 下降
    "operating_cost": 180.0,  # 毛利率 0.10
    "operating_cost_prev": 90.0,  # _prev 0.70 → 下降
    "total_liab": 60.0,  # 0.500
    "total_liab_prev": 30.0,  # 0.300 → 上升
    "noncurrent_liab": 20.0,
    "noncurrent_liab_prev": 5.0,
    "current_assets": 40.0,  # 2.0
    "current_liab": 20.0,
    "current_assets_prev": 60.0,  # 3.0 → 下降
    "current_liab_prev": 20.0,
    "shares": 1200.0,  # 增发
    "shares_prev": 1000.0,
}


def frame(**overrides):
    """构造单行报告帧。"""
    return pd.DataFrame([{**PERFECT, **overrides}])


def value(series):
    return series.iloc[0]


class TestGoldenRows:
    def test_perfect_stock_scores_nine(self):
        signals = compute_signals(frame())
        assert signals.iloc[0].tolist() == [1.0] * 9

    def test_terrible_stock_scores_zero(self):
        signals = compute_signals(pd.DataFrame([TERRIBLE]))
        assert signals.iloc[0].tolist() == [0.0] * 9

    def test_all_signals_present(self):
        assert set(compute_signals(frame()).columns) == set(SIGNAL_NAMES)
        assert len(SIGNAL_NAMES) == 9


class TestF1RoaPositive:
    @pytest.mark.parametrize(
        "net_profit,expected", [(10.0, 1.0), (-10.0, 0.0), (0.001, 1.0)]
    )
    def test_sign(self, net_profit, expected):
        assert value(f1_roa_positive(frame(net_profit=net_profit))) == expected

    def test_exactly_zero_is_zero(self):
        """严格不等号：ROA 恰好为 0 判 0。"""
        assert value(f1_roa_positive(frame(net_profit=0.0))) == 0.0

    def test_uses_beginning_assets(self):
        """分母是期初总资产（total_assets_prev），不是当期。"""
        base = f1_roa_positive(frame())
        changed = f1_roa_positive(frame(total_assets=999999.0))
        assert value(base) == value(changed)

    def test_average_basis_differs(self):
        """均值口径下分母变大，但符号不变 —— 用能翻转的数据来验证生效。"""
        beginning = f1_roa_positive(
            frame(total_assets=100.0, total_assets_prev=0.0), ta_basis="beginning"
        )
        average = f1_roa_positive(
            frame(total_assets=100.0, total_assets_prev=0.0), ta_basis="average"
        )
        assert np.isnan(value(beginning))  # 期初为 0 → NaN
        assert value(average) == 1.0  # 均值 50 → 可算

    def test_zero_denominator_is_nan_not_inf(self):
        out = value(f1_roa_positive(frame(total_assets_prev=0.0)))
        assert np.isnan(out)

    def test_missing_input_is_nan(self):
        assert np.isnan(value(f1_roa_positive(frame(net_profit=np.nan))))

    def test_missing_column_is_nan(self):
        df = frame().drop(columns=["net_profit"])
        assert np.isnan(value(f1_roa_positive(df)))


class TestF2CfoPositive:
    @pytest.mark.parametrize("cfo,expected", [(15.0, 1.0), (-1.0, 0.0), (0.0, 0.0)])
    def test_sign(self, cfo, expected):
        assert value(f2_cfo_positive(frame(cfo=cfo))) == expected

    def test_missing_is_nan(self):
        assert np.isnan(value(f2_cfo_positive(frame(cfo=np.nan))))


class TestF3DeltaRoa:
    def test_improvement(self):
        assert value(f3_delta_roa(frame())) == 1.0

    def test_deterioration(self):
        # ROA = 10/100 = 0.10；ROA_prev = 18/90 = 0.20 → 恶化
        assert value(f3_delta_roa(frame(net_profit_prev=18.0))) == 0.0

    def test_exactly_equal_is_zero(self):
        """ROA = 10/100 = 0.1，令 ROA_prev = 9/90 = 0.1 恰好相等 → 判 0。"""
        assert value(f3_delta_roa(frame(net_profit_prev=9.0))) == 0.0

    def test_uses_prior_period_own_denominator(self):
        """上期 ROA 的分母是 total_assets_prev2 —— 这就是需要 lag 2 的原因。"""
        assert np.isnan(value(f3_delta_roa(frame(total_assets_prev2=np.nan))))

    def test_needs_both_periods(self):
        assert np.isnan(value(f3_delta_roa(frame(net_profit_prev=np.nan))))


class TestF4Accrual:
    def test_cash_backed_earnings(self):
        assert value(f4_accrual(frame())) == 1.0

    def test_accrual_heavy_earnings(self):
        # CFO = 5 → 0.05 < ROA = 0.10
        assert value(f4_accrual(frame(cfo=5.0))) == 0.0

    def test_exactly_equal_is_zero(self):
        assert value(f4_accrual(frame(cfo=10.0))) == 0.0

    def test_works_when_both_negative(self):
        """亏损公司同样要能判：CFO = -5 优于净利 -10。"""
        assert value(f4_accrual(frame(net_profit=-10.0, cfo=-5.0))) == 1.0


class TestF5DeltaLever:
    def test_deleveraging_scores_one(self):
        assert value(f5_delta_lever(frame())) == 1.0

    def test_leveraging_up_scores_zero(self):
        assert value(f5_delta_lever(frame(total_liab=60.0))) == 0.0

    def test_exactly_equal_is_zero(self):
        # 用二进制可精确表示的值，否则浮点尾差会让"恰好相等"变成极小的负差
        # 资产负债率 = 30/120 = 0.25；_prev = 25/100 = 0.25
        out = f5_delta_lever(frame(total_liab=30.0, total_liab_prev=25.0))
        assert value(out) == 0.0

    def test_noncurrent_definition(self):
        """非流动负债口径更接近原文的长期负债定义。"""
        out = f5_delta_lever(frame(), definition="noncurrent")
        assert value(out) == 1.0

    def test_noncurrent_definition_detects_increase(self):
        out = f5_delta_lever(
            frame(noncurrent_liab=50.0, noncurrent_liab_prev=1.0),
            definition="noncurrent",
        )
        assert value(out) == 0.0

    def test_noncurrent_missing_is_nan(self):
        """批量数据没有非流动负债时该口径应给 NaN，而不是悄悄退化。"""
        df = frame().drop(columns=["noncurrent_liab"])
        assert np.isnan(value(f5_delta_lever(df, definition="noncurrent")))

    def test_unknown_definition_raises(self):
        with pytest.raises(ValueError, match="lever_definition"):
            f5_delta_lever(frame(), definition="nonsense")


class TestF6DeltaLiquid:
    def test_improving_current_ratio(self):
        assert value(f6_delta_liquid(frame())) == 1.0

    def test_deteriorating_current_ratio(self):
        assert value(f6_delta_liquid(frame(current_assets=30.0))) == 0.0

    def test_exactly_equal_is_zero(self):
        out = f6_delta_liquid(frame(current_assets=50.0, current_liab=25.0))
        assert value(out) == 0.0

    def test_bank_style_missing_data_is_nan(self):
        """银行报表没有流动/非流动分类，实测这两列为空 → 信号必须是 NaN。"""
        df = frame(current_assets=np.nan, current_liab=np.nan)
        assert np.isnan(value(f6_delta_liquid(df)))

    def test_zero_current_liab_is_nan_not_inf(self):
        assert np.isnan(value(f6_delta_liquid(frame(current_liab=0.0))))


class TestF7NoEquityOffer:
    def test_unchanged_shares_scores_one(self):
        """股本不变判 1 —— 这里边界是闭区间（<=）。"""
        assert value(f7_no_equity_offer(frame())) == 1.0

    def test_share_issuance_scores_zero(self):
        assert value(f7_no_equity_offer(frame(shares=1100.0))) == 0.0

    def test_buyback_scores_one(self):
        assert value(f7_no_equity_offer(frame(shares=900.0))) == 1.0

    def test_tolerance_absorbs_small_increase(self):
        """容忍度用于消化送转股 —— 本实现无法把它与增发区分开。"""
        df = frame(shares=1005.0)
        assert value(f7_no_equity_offer(df, tolerance=0.0)) == 0.0
        assert value(f7_no_equity_offer(df, tolerance=0.01)) == 1.0

    def test_missing_shares_is_nan(self):
        assert np.isnan(value(f7_no_equity_offer(frame(shares=np.nan))))


class TestF8DeltaMargin:
    def test_improving_margin(self):
        assert value(f8_delta_margin(frame())) == 1.0

    def test_deteriorating_margin(self):
        assert value(f8_delta_margin(frame(operating_cost=150.0))) == 0.0

    def test_exactly_equal_is_zero(self):
        # 毛利率 (200-120)/200 = 0.40 == (150-90)/150 = 0.40
        assert value(f8_delta_margin(frame(operating_cost=120.0))) == 0.0

    def test_zero_revenue_is_nan(self):
        assert np.isnan(value(f8_delta_margin(frame(revenue=0.0))))

    def test_missing_cost_is_nan(self):
        assert np.isnan(value(f8_delta_margin(frame(operating_cost=np.nan))))


class TestF9DeltaTurnover:
    def test_improving_turnover(self):
        assert value(f9_delta_turnover(frame())) == 1.0

    def test_deteriorating_turnover(self):
        assert value(f9_delta_turnover(frame(revenue_prev=300.0))) == 0.0

    def test_exactly_equal_is_zero(self):
        # 周转 200/100 = 2.0；令 _prev = 180/90 = 2.0
        assert value(f9_delta_turnover(frame(revenue_prev=180.0))) == 0.0

    def test_needs_lag_two_assets(self):
        assert np.isnan(value(f9_delta_turnover(frame(total_assets_prev2=np.nan))))


class TestSignalContracts:
    @pytest.mark.parametrize("name", SIGNAL_NAMES)
    def test_returns_only_zero_one_or_nan(self, name):
        for row in (PERFECT, TERRIBLE):
            out = SIGNAL_FUNCTIONS[name](pd.DataFrame([row]))
            assert set(out.dropna().unique()) <= {0.0, 1.0}

    @pytest.mark.parametrize("name", SIGNAL_NAMES)
    def test_never_returns_inf(self, name):
        """所有分母都置零，必须得到 NaN 而非 inf。"""
        zeros = {k: 0.0 for k in PERFECT}
        out = SIGNAL_FUNCTIONS[name](pd.DataFrame([zeros]))
        assert not np.isinf(out).any()

    @pytest.mark.parametrize("name", SIGNAL_NAMES)
    def test_empty_frame_yields_empty_series(self, name):
        empty = pd.DataFrame({k: pd.Series(dtype=float) for k in PERFECT})
        assert len(SIGNAL_FUNCTIONS[name](empty)) == 0

    @pytest.mark.parametrize("name", SIGNAL_NAMES)
    def test_required_columns_declared(self, name):
        """REQUIRED_COLUMNS 必须覆盖到该信号真正会用到的列。"""
        df = frame().drop(columns=list(REQUIRED_COLUMNS[name]), errors="ignore")
        assert np.isnan(value(SIGNAL_FUNCTIONS[name](df)))

    @pytest.mark.parametrize("name", SIGNAL_NAMES)
    def test_preserves_index(self, name):
        df = frame()
        df.index = pd.Index(["600519.SH"], name="code")
        assert SIGNAL_FUNCTIONS[name](df).index.equals(df.index)


class TestComputeFFScore:
    def test_score_equals_signal_sum(self):
        out = compute_ffscore(frame())
        assert out["f_score"].iloc[0] == pytest.approx(
            out[list(SIGNAL_NAMES)].iloc[0].sum()
        )

    def test_perfect_is_nine(self):
        assert compute_ffscore(frame())["f_score"].iloc[0] == 9.0

    def test_terrible_is_zero(self):
        assert compute_ffscore(pd.DataFrame([TERRIBLE]))["f_score"].iloc[0] == 0.0

    def test_n_available_counts_computable_signals(self):
        df = frame(current_assets=np.nan)  # 打掉 F6
        out = compute_ffscore(df)
        assert out["n_available"].iloc[0] == 8

    def test_min_signals_gate(self):
        """可算信号不足时该记录不参与，f_score 置 NaN。"""
        df = frame(current_assets=np.nan, shares=np.nan)  # 只剩 7 个
        out = compute_ffscore(df, default_config(min_signals=8))
        assert np.isnan(out["f_score"].iloc[0])
        assert out["n_available"].iloc[0] == 7

    def test_eight_signal_mode_allowed(self):
        df = frame(current_assets=np.nan)
        out = compute_ffscore(df, default_config(min_signals=8))
        assert out["f_score"].iloc[0] == 8.0

    def test_scaled_score_makes_modes_comparable(self):
        """8 信号模式下满分 8 折算回 9，才能与 9 信号模式比较。"""
        df = frame(current_assets=np.nan)
        out = compute_ffscore(df, default_config(min_signals=8))
        assert out["f_score_scaled"].iloc[0] == pytest.approx(9.0)

    def test_passes_through_metadata(self):
        df = frame()
        df["ann_date"] = pd.Timestamp("2023-04-28")
        df["name"] = "贵州茅台"
        out = compute_ffscore(df)

        assert out["ann_date"].iloc[0] == pd.Timestamp("2023-04-28")
        assert out["name"].iloc[0] == "贵州茅台"

    def test_lever_definition_flows_from_config(self):
        df = frame().drop(columns=["noncurrent_liab"])
        out = compute_ffscore(
            df, default_config(lever_definition="noncurrent", min_signals=0)
        )
        assert np.isnan(out["f5_delta_lever"].iloc[0])
        assert out["f_score"].iloc[0] == 8.0
