# -*- coding: utf-8 -*-
"""A 股累计报表还原（单季 / TTM）的单元测试。

A 股 Q2/Q3/Q4 的流量项是年初至今累计，不还原就直接跨期比较是经典错误。
"""

import numpy as np
import pandas as pd
import pytest

from research.datafeed.fundamentals import (
    finalize_report_panel,
    to_single_quarter,
    to_ttm,
)

# 两个财年的累计值：FY2021 单季 10/15/20/35，FY2022 单季 12/18/25/45
CUMULATIVE = {
    2021: [10.0, 25.0, 45.0, 80.0],
    2022: [12.0, 30.0, 55.0, 100.0],
}
SINGLE_QUARTER = {
    2021: [10.0, 15.0, 20.0, 35.0],
    2022: [12.0, 18.0, 25.0, 45.0],
}


def build_panel(cumulative=None, code="600519", drop_periods=()):
    """由累计值字典构造报告面板。"""
    cumulative = cumulative or CUMULATIVE
    rows = []

    for year, values in cumulative.items():
        for quarter, value in enumerate(values, start=1):
            period = f"{year}{['0331', '0630', '0930', '1231'][quarter - 1]}"
            if period in drop_periods:
                continue
            rows.append(
                {
                    "code": code,
                    "period": period,
                    "ann_date_balance": pd.Timestamp(period) + pd.Timedelta(60, "D"),
                    "net_profit": value,
                    "revenue": value * 4,
                    "total_assets": 1000.0 + year,  # 存量项：不得被差分
                }
            )

    return finalize_report_panel(pd.DataFrame(rows))


def series_for(panel, column, year):
    """取某财年四个季度的值（按季度升序）。"""
    frame = panel.reset_index()
    frame = frame[frame["fiscal_year"] == year].sort_values("quarter")
    return frame[column].tolist()


class TestToSingleQuarter:
    def test_differences_within_fiscal_year(self):
        out = to_single_quarter(build_panel())
        assert series_for(out, "net_profit", 2021) == pytest.approx(
            SINGLE_QUARTER[2021]
        )

    def test_q1_is_unchanged(self):
        out = to_single_quarter(build_panel())
        assert series_for(out, "net_profit", 2021)[0] == pytest.approx(10.0)

    def test_fiscal_year_boundary_resets(self):
        """FY2022 Q1 是 12，不能变成 100 - 80 = 20 那样跨年差分。"""
        out = to_single_quarter(build_panel())
        assert series_for(out, "net_profit", 2022)[0] == pytest.approx(12.0)

    def test_missing_quarter_yields_nan(self):
        """缺 Q2 时 Q3 无法还原，必须是 NaN 而不是错误的 45。"""
        panel = build_panel(drop_periods=("20210630",))
        out = to_single_quarter(panel)

        values = series_for(out, "net_profit", 2021)
        assert np.isnan(values[1]), "Q3 缺少上一季累计值时必须为 NaN"

    def test_stock_columns_untouched(self):
        """存量项是时点余额，绝不能差分。"""
        panel = build_panel()
        out = to_single_quarter(panel)

        pd.testing.assert_series_equal(
            out["total_assets"], panel["total_assets"], check_names=False
        )

    def test_all_flow_columns_converted(self):
        out = to_single_quarter(build_panel())
        assert series_for(out, "revenue", 2021) == pytest.approx(
            [v * 4 for v in SINGLE_QUARTER[2021]]
        )

    def test_preserves_index(self):
        panel = build_panel()
        assert to_single_quarter(panel).index.equals(panel.index)


class TestToTTM:
    def test_q4_equals_full_year(self):
        """年报 Q4 的累计值本身就是整年，TTM 应当不变。"""
        out = to_ttm(build_panel())
        assert series_for(out, "net_profit", 2022)[3] == pytest.approx(100.0)

    def test_annual_anchored_formula(self):
        """FY2022 Q2 的 TTM = 30 + 80 - 25 = 85。"""
        out = to_ttm(build_panel(), method="annual_anchored")
        assert series_for(out, "net_profit", 2022)[1] == pytest.approx(85.0)

    def test_two_methods_agree_on_complete_data(self):
        """annual_anchored 与 rolling4 在数据完整时必须一致 —— 交叉校验。"""
        panel = build_panel()
        anchored = to_ttm(panel, method="annual_anchored")
        rolling = to_ttm(panel, method="rolling4")

        a = series_for(anchored, "net_profit", 2022)
        r = series_for(rolling, "net_profit", 2022)

        assert a == pytest.approx(r), "两种 TTM 口径在完整数据上应当相等"

        # 手算核对（annual_anchored = 本期累计 + 上一完整财年 - 去年同期累计）：
        #   Q1: 12 + 80 - 10 = 82    Q2: 30 + 80 - 25 = 85
        #   Q3: 55 + 80 - 45 = 90    Q4: 100 + 80 - 80 = 100
        assert a == pytest.approx([82.0, 85.0, 90.0, 100.0])

    def test_first_year_has_no_ttm(self):
        """没有上一财年就算不出 TTM。"""
        out = to_ttm(build_panel())
        assert all(np.isnan(v) for v in series_for(out, "net_profit", 2021))

    def test_stock_columns_untouched(self):
        panel = build_panel()
        out = to_ttm(panel)
        pd.testing.assert_series_equal(
            out["total_assets"], panel["total_assets"], check_names=False
        )

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="未知的 TTM 方法"):
            to_ttm(build_panel(), method="nonsense")


class TestMultipleCodes:
    def test_codes_do_not_bleed_into_each_other(self):
        a = build_panel(code="600519")
        b = build_panel(
            cumulative={2021: [1.0, 2.0, 3.0, 4.0], 2022: [5.0, 6.0, 7.0, 8.0]},
            code="000001",
        )
        panel = pd.concat([a, b]).sort_index()

        out = to_single_quarter(panel)
        frame = out.reset_index()

        moutai = frame[
            (frame["code"] == "600519.SH") & (frame["fiscal_year"] == 2021)
        ].sort_values("quarter")["net_profit"]
        pingan = frame[
            (frame["code"] == "000001.SZ") & (frame["fiscal_year"] == 2021)
        ].sort_values("quarter")["net_profit"]

        assert moutai.tolist() == pytest.approx(SINGLE_QUARTER[2021])
        assert pingan.tolist() == pytest.approx([1.0, 1.0, 1.0, 1.0])
