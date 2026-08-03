# -*- coding: utf-8 -*-
"""时点（PIT）正确性测试 —— 本项目最重要的一组断言。

前视偏差不会让任何测试变红，只会让回测变好看，所以必须显式地钉住它。
"""

import numpy as np
import pandas as pd
import pytest

from research.datafeed.fundamentals import (
    add_available_date,
    add_lagged,
    as_of,
    build_pit_panel,
    finalize_report_panel,
    listing_dates,
    screen_universe,
)


def make_raw(rows):
    """由紧凑的 spec 构造原始报告帧（三张表的公告日期可分别指定）。"""
    frame = pd.DataFrame(rows)

    for col in ("ann_date_balance", "ann_date_income", "ann_date_cashflow"):
        if col not in frame.columns:
            frame[col] = frame["ann_date"]

    return frame.drop(columns=["ann_date"])


@pytest.fixture
def two_period_panel():
    """一只股票的两个年报：2021 年报 2022-04-20 公告，2022 年报 2023-04-28 公告。"""
    raw = make_raw(
        [
            {
                "code": "600519",
                "name": "贵州茅台",
                "period": "20211231",
                "ann_date": "2022-04-20",
                "total_assets": 100.0,
                "total_liab": 30.0,
                "total_equity": 70.0,
                "net_profit": 10.0,
                "revenue": 50.0,
                "operating_cost": 20.0,
                "cfo": 12.0,
            },
            {
                "code": "600519",
                "name": "贵州茅台",
                "period": "20221231",
                "ann_date": "2023-04-28",
                "total_assets": 120.0,
                "total_liab": 30.0,
                "total_equity": 90.0,
                "net_profit": 15.0,
                "revenue": 60.0,
                "operating_cost": 22.0,
                "cfo": 18.0,
            },
        ]
    )
    return add_available_date(finalize_report_panel(raw))


class TestAnnDateIsMaxOfThreeTables:
    def test_takes_latest_of_three(self):
        """一个信号要三张表齐备，最晚的那张才决定可用时点。"""
        raw = pd.DataFrame(
            [
                {
                    "code": "600519",
                    "period": "20221231",
                    "total_assets": 100.0,
                    "ann_date_balance": "2023-04-20",
                    "ann_date_income": "2023-04-28",
                    "ann_date_cashflow": "2023-04-25",
                }
            ]
        )
        panel = finalize_report_panel(raw)
        assert panel["ann_date"].iloc[0] == pd.Timestamp("2023-04-28")

    def test_derives_fiscal_year_and_quarter(self):
        raw = make_raw(
            [{"code": "600519", "period": "20220630", "ann_date": "2022-08-30"}]
        )
        panel = finalize_report_panel(raw)

        assert panel["fiscal_year"].iloc[0] == 2022
        assert panel["quarter"].iloc[0] == 2

    def test_debt_ratio_computed_not_trusted(self):
        """自己算资产负债率，避免接口单位（百分比 vs 比例）的不确定性。"""
        raw = make_raw(
            [
                {
                    "code": "600519",
                    "period": "20221231",
                    "ann_date": "2023-04-28",
                    "total_assets": 200.0,
                    "total_liab": 50.0,
                }
            ]
        )
        assert finalize_report_panel(raw)["debt_ratio"].iloc[0] == pytest.approx(0.25)

    def test_zero_assets_gives_nan_not_inf(self):
        raw = make_raw(
            [
                {
                    "code": "600519",
                    "period": "20221231",
                    "ann_date": "2023-04-28",
                    "total_assets": 0.0,
                    "total_liab": 50.0,
                }
            ]
        )
        assert np.isnan(finalize_report_panel(raw)["debt_ratio"].iloc[0])


class TestAsOfBoundary:
    def test_day_before_announcement_sees_prior_period(self, two_period_panel):
        cross = as_of(two_period_panel, "2023-04-27")
        assert cross.loc["600519.SH", "period"] == pd.Timestamp("20211231")

    def test_announcement_day_sees_new_period(self, two_period_panel):
        """公告当日即可用 —— 边界是闭区间。"""
        cross = as_of(two_period_panel, "2023-04-28")
        assert cross.loc["600519.SH", "period"] == pd.Timestamp("20221231")

    def test_before_any_announcement_is_empty(self, two_period_panel):
        assert len(as_of(two_period_panel, "2020-01-01")) == 0

    def test_returns_one_row_per_code(self, two_period_panel):
        cross = as_of(two_period_panel, "2024-01-01")
        assert cross.index.is_unique
        assert len(cross) == 1

    def test_requires_available_date(self):
        raw = make_raw(
            [{"code": "600519", "period": "20221231", "ann_date": "2023-04-28"}]
        )
        with pytest.raises(KeyError, match="available_date"):
            as_of(finalize_report_panel(raw), "2023-05-01")


class TestAvailableDateUsesLaggedAnnouncements:
    def test_late_prior_filing_delays_availability(self):
        """去年年报比今年年报还晚公告时，Δ 类信号必须等到那时才可用。

        追溯调整与延迟披露会让时序倒置，这正是取 max 而非直接用当期公告日期
        的理由。
        """
        raw = make_raw(
            [
                {
                    "code": "600519",
                    "period": "20211231",
                    "ann_date": "2023-06-01",  # 迟到的上期报告
                    "total_assets": 100.0,
                    "net_profit": 10.0,
                },
                {
                    "code": "600519",
                    "period": "20221231",
                    "ann_date": "2023-04-10",  # 当期反而更早
                    "total_assets": 120.0,
                    "net_profit": 15.0,
                },
            ]
        )
        panel = finalize_report_panel(raw)
        lagged = add_lagged(panel, ["total_assets", "net_profit"], lags=(1,))
        result = add_available_date(lagged)

        row = result.loc[("600519.SH", pd.Timestamp("20221231"))]

        assert row["ann_date"] == pd.Timestamp("2023-04-10")
        assert row["ann_date_prev"] == pd.Timestamp("2023-06-01")
        assert row["available_date"] == pd.Timestamp(
            "2023-06-01"
        ), "整条记录要等最晚的那份报告公告后才可用"

        # 直接用当期公告日期就会在 4/10 放行 —— 那是前视
        assert as_of(result, "2023-05-01").empty

    def test_lag_two_supported(self):
        """F9（ΔTURN）需要 TA_{t-2}。"""
        raw = make_raw(
            [
                {
                    "code": "600519",
                    "period": f"{y}1231",
                    "ann_date": f"{y + 1}-04-20",
                    "total_assets": float(v),
                }
                for y, v in [(2020, 100), (2021, 110), (2022, 120)]
            ]
        )
        panel = finalize_report_panel(raw)
        lagged = add_lagged(panel, ["total_assets"], lags=(1, 2))

        row = lagged.loc[("600519.SH", pd.Timestamp("20221231"))]
        assert row["total_assets"] == pytest.approx(120)
        assert row["total_assets_prev"] == pytest.approx(110)
        assert row["total_assets_prev2"] == pytest.approx(100)

    def test_gap_in_history_yields_nan_not_misalignment(self):
        """报告期有缺口时必须给 NaN，不能错位接上更早的年份。"""
        raw = make_raw(
            [
                {
                    "code": "600519",
                    "period": "20201231",
                    "ann_date": "2021-04-20",
                    "total_assets": 100.0,
                },
                {
                    "code": "600519",
                    "period": "20221231",
                    "ann_date": "2023-04-20",
                    "total_assets": 120.0,
                },
            ]
        )
        lagged = add_lagged(finalize_report_panel(raw), ["total_assets"], lags=(1,))
        row = lagged.loc[("600519.SH", pd.Timestamp("20221231"))]
        assert np.isnan(row["total_assets_prev"])


class TestBuildPitPanel:
    def test_no_lookahead_invariant(self, two_period_panel):
        """整面板不变式：任何一行的可用时点都不得晚于它所在的截面日期。"""
        dates = pd.to_datetime(["2022-06-30", "2023-04-27", "2023-04-28", "2023-12-31"])
        panel = build_pit_panel(two_period_panel, dates)

        assert (
            panel["available_date"] <= panel.index.get_level_values("date")
        ).all(), "出现了前视：某行使用了截面日尚未公开的数据"

    def test_picks_correct_period_per_date(self, two_period_panel):
        dates = pd.to_datetime(["2023-04-27", "2023-04-28"])
        panel = build_pit_panel(two_period_panel, dates)

        assert panel.loc[
            (pd.Timestamp("2023-04-27"), "600519.SH"), "period"
        ] == pd.Timestamp("20211231")
        assert panel.loc[
            (pd.Timestamp("2023-04-28"), "600519.SH"), "period"
        ] == pd.Timestamp("20221231")

    def test_poison_row_never_surfaces(self):
        """投毒：注入一条远期公告的荒谬记录，它绝不能出现在更早的截面里。"""
        raw = make_raw(
            [
                {
                    "code": "600519",
                    "period": "20221231",
                    "ann_date": "2023-04-28",
                    "total_assets": 120.0,
                    "net_profit": 15.0,
                },
                {
                    "code": "600519",
                    "period": "20231231",
                    "ann_date": "2024-04-28",  # 远期公告
                    "total_assets": 1e12,
                    "net_profit": 1e12,  # 荒谬到一眼可辨
                },
            ]
        )
        poisoned = add_available_date(finalize_report_panel(raw))

        dates = pd.to_datetime(["2023-06-30", "2024-01-31", "2024-04-28"])
        panel = build_pit_panel(poisoned, dates)

        early = panel[panel.index.get_level_values("date") < pd.Timestamp("2024-04-28")]
        assert (early["net_profit"] < 1e11).all(), "未来数据泄漏进了历史截面"

        assert (panel["available_date"] <= panel.index.get_level_values("date")).all()

        # 到了公告日当天，那条记录才应该出现
        assert panel.loc[
            (pd.Timestamp("2024-04-28"), "600519.SH"), "net_profit"
        ] == pytest.approx(1e12)

    def test_codes_without_visible_data_are_dropped(self, two_period_panel):
        panel = build_pit_panel(two_period_panel, pd.to_datetime(["2020-01-01"]))
        assert len(panel) == 0


class TestSurvivorshipScreens:
    def test_listing_dates_from_price_panel(self, price_panel):
        listed = listing_dates(price_panel)
        assert listed["600000.SH"] == price_panel.index.get_level_values("date").min()

    def test_not_yet_listed_is_excluded(self, price_panel):
        """核心的幸存者偏差修正：批量财报会把历史财报回填到尚未上市的年份。"""
        first_bar = price_panel.index.get_level_values("date").min()
        # 用 pd.Timedelta(n, "D") 而非 pd.Timedelta(days=n)：后者在
        # pandas 2.3 + numpy 2.5 下会触发 generic-unit 弃用警告
        date = first_bar + pd.Timedelta(30, "D")

        pit = pd.DataFrame(
            {"name": ["某公司"], "total_equity": [100.0]},
            index=pd.MultiIndex.from_tuples(
                [(date, "600000.SH")], names=["date", "code"]
            ),
        )
        mask = screen_universe(pit, price_panel, {"exclude_suspended": False})
        assert not mask.iloc[0], "上市不足 365 天的标的必须被剔除"

    def test_long_listed_passes(self, price_panel):
        first_bar = price_panel.index.get_level_values("date").min()
        date = first_bar + pd.Timedelta(400, "D")

        pit = pd.DataFrame(
            {"name": ["某公司"], "total_equity": [100.0]},
            index=pd.MultiIndex.from_tuples(
                [(date, "600000.SH")], names=["date", "code"]
            ),
        )
        mask = screen_universe(pit, price_panel, {"exclude_suspended": False})
        assert mask.iloc[0]

    @pytest.mark.parametrize(
        "name,kept",
        [
            ("贵州茅台", True),
            ("ST康美", False),
            ("*ST海航", False),
            ("退市锐电", False),
            ("招商银行", False),
            ("中信证券", False),
            ("中国平安", True),
            ("中国人寿保险", False),
        ],
    )
    def test_name_based_screens(self, name, kept):
        pit = pd.DataFrame(
            {"name": [name], "total_equity": [100.0]},
            index=pd.MultiIndex.from_tuples(
                [(pd.Timestamp("2023-01-03"), "600000.SH")], names=["date", "code"]
            ),
        )
        assert bool(screen_universe(pit, None).iloc[0]) is kept

    def test_negative_equity_excluded(self):
        pit = pd.DataFrame(
            {"name": ["某公司"], "total_equity": [-5.0]},
            index=pd.MultiIndex.from_tuples(
                [(pd.Timestamp("2023-01-03"), "600000.SH")], names=["date", "code"]
            ),
        )
        assert not screen_universe(pit, None).iloc[0]

    def test_mask_aligns_with_panel_index(self, price_panel):
        dates = price_panel.index.get_level_values("date").unique()[:3]
        codes = price_panel.index.get_level_values("code").unique()

        pit = pd.DataFrame(
            {"name": "某公司", "total_equity": 100.0},
            index=pd.MultiIndex.from_product([dates, codes], names=["date", "code"]),
        )
        mask = screen_universe(pit, price_panel)

        assert mask.index.equals(pit.index)
        assert mask.dtype == bool
