# -*- coding: utf-8 -*-
"""research/metrics.py 的单元测试（全离线）。"""

import numpy as np
import pandas as pd
import pytest

from research.metrics import (
    annualized_return,
    annualized_vol,
    calmar_ratio,
    drawdown_series,
    ic_summary,
    information_coefficient,
    max_drawdown,
    nav_from_returns,
    newey_west_t,
    sharpe_ratio,
    sortino_ratio,
    summarize_returns,
    turnover,
)


class TestNavAndAnnualization:
    def test_nav_accumulates(self):
        nav = nav_from_returns([0.1, -0.1])
        assert nav.iloc[0] == pytest.approx(1.1)
        assert nav.iloc[1] == pytest.approx(1.1 * 0.9)

    def test_annualized_return_identity(self):
        """恒定单期收益、恰好一年 → 年化 = (1+c)**periods - 1。"""
        c = 0.001
        r = [c] * 252
        assert annualized_return(r, periods=252) == pytest.approx(
            (1 + c) ** 252 - 1, rel=1e-9
        )

    def test_annualized_return_losing_series_is_finite(self):
        """回归测试：notebook 的 (cum-1)**(250/T)-1 会让亏损序列变 NaN。

        这里样本长度与年化期数不同，指数为分数；错误公式对负底数开分数次幂
        必得 NaN。正确实现必须返回有限负值。
        """
        r = [-0.001] * 500
        out = annualized_return(r, periods=252)

        assert np.isfinite(out), "亏损序列的年化收益不得为 NaN"
        assert out < 0

        final_nav = float(np.prod([1 + x for x in r]))
        assert out == pytest.approx(final_nav ** (252 / 500) - 1, rel=1e-9)

        # 与错误公式对照，证明本测试确实盯住了那个 bug。
        # 必须用 np.float64：notebook 里 cum.iloc[-1] 是 numpy 标量，负底数开分数
        # 次幂得 NaN；而纯 Python float 在同样运算下返回的是复数，不是 NaN。
        with np.errstate(invalid="ignore"):
            buggy = np.float64(final_nav - 1) ** (252 / 500) - 1
        assert np.isnan(buggy)

    def test_annualized_return_total_loss(self):
        assert annualized_return([-1.0, 0.5], periods=252) == -1.0

    def test_annualized_return_empty(self):
        assert annualized_return([], periods=252) == 0.0

    def test_annualized_vol(self):
        r = pd.Series([0.01, -0.01, 0.02, -0.02])
        assert annualized_vol(r, periods=252) == pytest.approx(
            r.std(ddof=1) * np.sqrt(252)
        )


class TestSharpeAndFriends:
    def test_sharpe_formula(self):
        rng = np.random.default_rng(0)
        r = pd.Series(rng.normal(0.0005, 0.01, 300))
        expected = r.mean() / r.std(ddof=1) * np.sqrt(252)
        assert sharpe_ratio(r, rf=0.0, periods=252) == pytest.approx(expected)

    def test_sharpe_zero_variance_returns_zero(self):
        """零波动是约定返回 0，不是 inf、不是 NaN。"""
        assert sharpe_ratio([0.001] * 50, periods=252) == 0.0

    def test_sharpe_too_short(self):
        assert sharpe_ratio([0.01], periods=252) == 0.0

    def test_sharpe_rf_reduces_result(self):
        r = pd.Series([0.001] * 100 + [-0.0005] * 100)
        assert sharpe_ratio(r, rf=0.04, periods=252) < sharpe_ratio(
            r, rf=0.0, periods=252
        )

    def test_sortino_only_penalizes_downside(self):
        """同均值下，下行波动更小的序列索提诺更高。"""
        a = pd.Series([0.02, -0.01] * 50)
        b = pd.Series([0.01, 0.0] * 50)
        assert sortino_ratio(b) > 0
        assert sortino_ratio(a) != sortino_ratio(b)

    def test_sortino_no_downside(self):
        assert sortino_ratio([0.01] * 10, rf=0.0) == np.inf


class TestDrawdown:
    def test_known_path(self):
        nav = [1.0, 2.0, 1.0, 4.0]
        assert max_drawdown(nav) == pytest.approx(0.5)

    def test_monotone_up_has_no_drawdown(self):
        assert max_drawdown([1.0, 1.1, 1.2]) == 0.0

    def test_max_equals_series_max(self):
        """两个函数必须口径一致，都返回正比例。"""
        rng = np.random.default_rng(7)
        nav = nav_from_returns(rng.normal(0, 0.01, 500))
        assert max_drawdown(nav) == pytest.approx(drawdown_series(nav).max())

    def test_drawdown_series_is_non_negative(self):
        rng = np.random.default_rng(11)
        nav = nav_from_returns(rng.normal(0, 0.02, 200))
        assert (drawdown_series(nav) >= -1e-12).all()

    def test_calmar(self):
        r = [0.01, -0.02, 0.03]
        expected = annualized_return(r) / max_drawdown(nav_from_returns(r))
        assert calmar_ratio(r) == pytest.approx(expected)

    def test_calmar_no_drawdown(self):
        assert calmar_ratio([0.01] * 10) == np.inf


class TestTurnover:
    def test_first_period_is_initial_build(self):
        w = pd.DataFrame([[0.5, 0.5]], columns=["a", "b"])
        assert turnover(w).iloc[0] == pytest.approx(0.5)

    def test_no_change_is_zero(self):
        w = pd.DataFrame([[0.5, 0.5], [0.5, 0.5]], columns=["a", "b"])
        assert turnover(w).iloc[1] == pytest.approx(0.0)

    def test_full_switch_is_one(self):
        w = pd.DataFrame([[1.0, 0.0], [0.0, 1.0]], columns=["a", "b"])
        assert turnover(w).iloc[1] == pytest.approx(1.0)


class TestInformationCoefficient:
    def test_perfect_rank_agreement(self):
        f = [1, 2, 3, 4, 5]
        r = [0.1, 0.2, 0.3, 0.4, 0.5]
        assert information_coefficient(f, r) == pytest.approx(1.0)

    def test_perfect_rank_inversion(self):
        f = [1, 2, 3, 4, 5]
        r = [0.5, 0.4, 0.3, 0.2, 0.1]
        assert information_coefficient(f, r) == pytest.approx(-1.0)

    def test_spearman_ignores_monotone_transform(self):
        """RankIC 对单调变换不变，这正是它比 pearson 稳健的地方。"""
        f = [1, 2, 3, 4, 5]
        r = [0.1, 0.2, 0.3, 0.4, 5.0]
        assert information_coefficient(f, r, "spearman") == pytest.approx(1.0)
        assert information_coefficient(f, r, "pearson") < 1.0

    def test_insufficient_sample_returns_nan(self):
        assert np.isnan(information_coefficient([1, 2], [0.1, 0.2]))

    def test_constant_factor_returns_nan(self):
        assert np.isnan(information_coefficient([1, 1, 1, 1], [0.1, 0.2, 0.3, 0.4]))

    def test_drops_nan_pairs(self):
        f = [1, 2, 3, 4, np.nan]
        r = [0.1, 0.2, 0.3, 0.4, 0.9]
        assert information_coefficient(f, r) == pytest.approx(1.0)


class TestNeweyWest:
    def test_lag_zero_matches_plain_t(self):
        x = pd.Series([0.1, -0.05, 0.2, 0.15, -0.1, 0.05])
        se = x.std(ddof=0) / np.sqrt(len(x))
        assert newey_west_t(x, lags=0) == pytest.approx(x.mean() / se)

    def test_positive_autocorrelation_shrinks_t(self):
        """正自相关会放大方差、压低 t 值 —— 这正是使用 NW 的理由。"""
        rng = np.random.default_rng(3)
        n = 400
        eps = rng.normal(0.02, 0.05, n)
        x = pd.Series(eps).ewm(alpha=0.2).mean()  # 引入强正自相关
        assert abs(newey_west_t(x, lags=10)) < abs(newey_west_t(x, lags=0))

    def test_zero_mean_series_has_small_t(self):
        rng = np.random.default_rng(5)
        x = pd.Series(rng.normal(0, 1, 1000))
        assert abs(newey_west_t(x)) < 3

    def test_too_short_returns_nan(self):
        assert np.isnan(newey_west_t([0.1, 0.2]))


class TestSummaries:
    def test_ic_summary_keys_and_values(self):
        ic = pd.Series([0.05, -0.02, 0.08, 0.01])
        out = ic_summary(ic)

        assert set(out) == {
            "ic_mean",
            "ic_std",
            "ic_ir",
            "ic_t",
            "positive_ratio",
            "n_periods",
        }
        assert out["ic_mean"] == pytest.approx(ic.mean())
        assert out["positive_ratio"] == pytest.approx(0.75)
        assert out["n_periods"] == 4

    def test_ic_summary_empty(self):
        out = ic_summary([])
        assert out["n_periods"] == 0
        assert np.isnan(out["ic_mean"])

    def test_summarize_returns(self):
        rng = np.random.default_rng(1)
        r = pd.Series(rng.normal(0.0004, 0.01, 756))
        out = summarize_returns(r)

        assert out["n_periods"] == 756
        assert out["sharpe"] == pytest.approx(sharpe_ratio(r))
        assert out["max_drawdown"] == pytest.approx(max_drawdown(nav_from_returns(r)))
        assert out["total_return"] == pytest.approx(float((1 + r).prod() - 1))
