# -*- coding: utf-8 -*-
"""真实公告日期的接入与兜底 —— 针对批量端点"公告日期"实为更新日的修正。

实测：贵州茅台 FY2021 的真实公告日是 2022-03-31，批量端点给的是 2023-03-31
（即 UPDATE_DATE）。全市场报告期 20211231 有 4842 只的"公告日期"落在 2023 年。
"""

import numpy as np
import pandas as pd
import pytest

from research.datafeed.calendar import filing_deadline
from research.datafeed.fundamentals import (
    attach_announcement_dates,
    finalize_detail_panel,
    finalize_report_panel,
)


@pytest.fixture
def bulk_panel():
    """批量端点形态：只有 update_date（迟到约一年），没有真实公告日期。"""
    raw = pd.DataFrame(
        [
            {
                "code": "600519",
                "period": "20211231",
                "total_assets": 100.0,
                "update_date_balance": "2023-03-31",
                "update_date_income": "2023-03-31",
                "update_date_cashflow": "2023-03-31",
            },
            {
                "code": "600519",
                "period": "20221231",
                "total_assets": 120.0,
                "update_date_balance": "2024-04-03",
                "update_date_income": "2024-04-03",
                "update_date_cashflow": "2024-04-03",
            },
        ]
    )
    return finalize_report_panel(raw)


@pytest.fixture
def detail_panel():
    """逐股明细形态：NOTICE_DATE 是真实公告日期。"""
    raw = pd.DataFrame(
        [
            {
                "code": "600519.SH",
                "period": "2021-12-31",
                "ann_date": "2022-03-31",
                "current_assets": 220.0,
                "current_liab": 57.0,
                "noncurrent_liab": 1.0,
            },
            {
                "code": "600519.SH",
                "period": "2022-12-31",
                "ann_date": "2023-03-31",
                "current_assets": 216.0,
                "current_liab": 49.0,
                "noncurrent_liab": 1.0,
            },
        ]
    )
    return finalize_detail_panel(raw)


class TestBulkPanelKeepsUpdateDateSeparate:
    def test_update_date_is_captured(self, bulk_panel):
        assert bulk_panel["update_date"].iloc[0] == pd.Timestamp("2023-03-31")

    def test_ann_date_is_empty_before_attach(self, bulk_panel):
        """批量端点不提供真实公告日期，ann_date 必须留空而不是拿更新日充数。"""
        assert bulk_panel["ann_date"].isna().all()


class TestFinalizeDetailPanel:
    def test_index_and_columns(self, detail_panel):
        assert detail_panel.index.names == ["code", "period"]
        assert "current_assets" in detail_panel.columns

    def test_keeps_earliest_announcement_on_restatement(self):
        """同一报告期有多条时取最早公告日 —— 市场首次看到该数字的时点。"""
        raw = pd.DataFrame(
            [
                {
                    "code": "600519.SH",
                    "period": "2021-12-31",
                    "ann_date": "2022-03-31",
                    "current_assets": 220.0,
                },
                {
                    "code": "600519.SH",
                    "period": "2021-12-31",
                    "ann_date": "2023-05-10",
                    "current_assets": 221.0,
                },
            ]
        )
        out = finalize_detail_panel(raw)

        assert len(out) == 1
        assert out["ann_date"].iloc[0] == pd.Timestamp("2022-03-31")


class TestAttachAnnouncementDates:
    def test_uses_notice_date_not_update_date(self, bulk_panel, detail_panel):
        """核心断言：接入后必须是真实公告日期，比更新日早约一年。"""
        out = attach_announcement_dates(bulk_panel, detail_panel)

        assert out.loc[("600519.SH", pd.Timestamp("20211231")), "ann_date"] == (
            pd.Timestamp("2022-03-31")
        )
        assert out.loc[("600519.SH", pd.Timestamp("20221231")), "ann_date"] == (
            pd.Timestamp("2023-03-31")
        )

    def test_update_date_preserved_for_audit(self, bulk_panel, detail_panel):
        out = attach_announcement_dates(bulk_panel, detail_panel)
        assert out["update_date"].iloc[0] == pd.Timestamp("2023-03-31")

    def test_ann_date_is_earlier_than_update_date(self, bulk_panel, detail_panel):
        """这一年的差距正是本修正的全部意义。"""
        out = attach_announcement_dates(bulk_panel, detail_panel)
        assert (out["ann_date"] < out["update_date"]).all()

    def test_brings_current_assets_for_f6(self, bulk_panel, detail_panel):
        """F6（流动比率变化）依赖这两列，批量端点没有。"""
        out = attach_announcement_dates(bulk_panel, detail_panel)

        assert out.loc[
            ("600519.SH", pd.Timestamp("20211231")), "current_assets"
        ] == pytest.approx(220.0)
        assert out.loc[
            ("600519.SH", pd.Timestamp("20211231")), "current_liab"
        ] == pytest.approx(57.0)

    def test_deadline_fallback_when_no_detail(self, bulk_panel):
        """没有明细时退到法定披露截止日：年报 → 次年 4/30。"""
        out = attach_announcement_dates(bulk_panel, detail=None)

        assert out.loc[("600519.SH", pd.Timestamp("20211231")), "ann_date"] == (
            pd.Timestamp("2022-04-30")
        )

    def test_deadline_fallback_only_fills_gaps(self, bulk_panel):
        """明细覆盖到的记录保留真实公告日期，未覆盖的才用兜底。"""
        partial = finalize_detail_panel(
            pd.DataFrame(
                [
                    {
                        "code": "600519.SH",
                        "period": "2021-12-31",
                        "ann_date": "2022-03-31",
                        "current_assets": 220.0,
                    }
                ]
            )
        )
        out = attach_announcement_dates(bulk_panel, partial)

        assert out.loc[("600519.SH", pd.Timestamp("20211231")), "ann_date"] == (
            pd.Timestamp("2022-03-31")
        )
        assert out.loc[("600519.SH", pd.Timestamp("20221231")), "ann_date"] == (
            pd.Timestamp("2023-04-30")
        )

    def test_fallback_none_leaves_nat(self, bulk_panel):
        out = attach_announcement_dates(bulk_panel, detail=None, fallback="none")
        assert out["ann_date"].isna().all()

    def test_unknown_fallback_raises(self, bulk_panel):
        with pytest.raises(ValueError, match="未知的兜底方式"):
            attach_announcement_dates(bulk_panel, fallback="nonsense")

    def test_announcement_before_period_end_is_rejected(self, bulk_panel):
        """公告日期早于报告期期末是不可能的，必须被清掉而不是照单全收。"""
        bad = finalize_detail_panel(
            pd.DataFrame(
                [
                    {
                        "code": "600519.SH",
                        "period": "2021-12-31",
                        "ann_date": "2021-06-30",
                        "current_assets": 220.0,
                    }
                ]
            )
        )
        out = attach_announcement_dates(bulk_panel, bad, fallback="none")
        assert pd.isna(out.loc[("600519.SH", pd.Timestamp("20211231")), "ann_date"])


class TestFilingDeadline:
    @pytest.mark.parametrize(
        "period,expected",
        [
            ("20211231", "2022-04-30"),  # 年报：次年 4/30
            ("20220331", "2022-04-30"),  # 一季报
            ("20220630", "2022-08-31"),  # 半年报
            ("20220930", "2022-10-31"),  # 三季报
        ],
    )
    def test_deadlines(self, period, expected):
        assert filing_deadline(period) == pd.Timestamp(expected)

    def test_deadline_is_after_period_end(self):
        for period in ("20211231", "20220331", "20220630", "20220930"):
            assert filing_deadline(period) > pd.Timestamp(period)
