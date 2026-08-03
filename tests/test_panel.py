# -*- coding: utf-8 -*-
"""代码归一与面板桥接的单元测试（全离线）。"""

import pandas as pd
import pytest

from research.datafeed.panel import (
    from_code_dict,
    normalize_code,
    to_code_dict,
    to_sina_symbol,
    to_wide,
)


class TestNormalizeCode:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("600519", "600519.SH"),
            ("sh600519", "600519.SH"),
            ("SH600519", "600519.SH"),
            ("600519.SH", "600519.SH"),
            ("600519.sh", "600519.SH"),
            ("688981", "688981.SH"),  # 科创板
            ("000001", "000001.SZ"),
            ("sz000001", "000001.SZ"),
            ("300750", "300750.SZ"),  # 创业板
            ("830799", "830799.BJ"),  # 北交所
            ("430047", "430047.BJ"),
            ("600519.XSHG", "600519.SH"),
            ("  600519  ", "600519.SH"),
        ],
    )
    def test_formats(self, raw, expected):
        assert normalize_code(raw) == expected

    def test_pads_short_numeric(self):
        """成分股接口偶尔返回去掉前导零的整数。"""
        assert normalize_code(1) == "000001.SZ"

    def test_idempotent(self):
        assert normalize_code(normalize_code("sh600519")) == "600519.SH"

    @pytest.mark.parametrize("bad", [None, "", "abcdef", "12345678", "999999"])
    def test_rejects_invalid(self, bad):
        with pytest.raises(ValueError):
            normalize_code(bad)


class TestSinaSymbol:
    def test_shanghai(self):
        assert to_sina_symbol("600519.SH") == "sh600519"

    def test_shenzhen(self):
        assert to_sina_symbol("000001") == "sz000001"

    def test_accepts_any_input_format(self):
        assert to_sina_symbol("sh600519") == "sh600519"

    def test_rejects_beijing(self):
        with pytest.raises(ValueError, match="北交所"):
            to_sina_symbol("830799.BJ")


class TestCodeDictBridge:
    def test_roundtrip_is_identity(self, price_panel):
        """to_code_dict ∘ from_code_dict 必须还原原面板。"""
        restored = from_code_dict(to_code_dict(price_panel))

        assert restored.index.names == ["date", "code"]
        pd.testing.assert_frame_equal(
            restored.sort_index(), price_panel.sort_index(), check_like=True
        )

    def test_code_dict_index_is_yyyymmdd_string(self, price_panel):
        """仓库既有约定：index 是 YYYYMMDD 字符串。"""
        out = to_code_dict(price_panel)
        first = out["600000.SH"]

        assert isinstance(first.index[0], str)
        assert len(first.index[0]) == 8
        assert first.index[0].isdigit()
        assert first.index.is_monotonic_increasing

    def test_code_dict_keys_are_normalized(self, price_panel):
        assert set(to_code_dict(price_panel)) == {
            "600000.SH",
            "000001.SZ",
            "300750.SZ",
        }

    def test_from_code_dict_accepts_date_column(self):
        data = {
            "sh600519": pd.DataFrame(
                {"date": ["20230103", "20230104"], "close": [1.0, 2.0]}
            )
        }
        panel = from_code_dict(data)

        assert panel.index.get_level_values("code").unique().tolist() == ["600519.SH"]
        assert panel["close"].tolist() == [1.0, 2.0]

    def test_from_code_dict_normalizes_keys(self):
        data = {"sh600519": pd.DataFrame({"close": [1.0]}, index=["20230103"])}
        assert from_code_dict(data).index.get_level_values("code")[0] == "600519.SH"

    def test_from_code_dict_empty(self):
        out = from_code_dict({})
        assert len(out) == 0
        assert out.index.names == ["date", "code"]


class TestToWide:
    def test_shape_and_orientation(self, price_panel):
        wide = to_wide(price_panel, "close")

        assert wide.shape == (60, 3)
        assert list(wide.columns) == ["000001.SZ", "300750.SZ", "600000.SH"]
        assert wide.index.is_monotonic_increasing

    def test_values_match_panel(self, price_panel):
        wide = to_wide(price_panel, "close")
        date = wide.index[5]
        assert wide.loc[date, "600000.SH"] == pytest.approx(
            price_panel.loc[(date, "600000.SH"), "close"]
        )

    def test_missing_field_raises(self, price_panel):
        with pytest.raises(KeyError, match="pb_ratio"):
            to_wide(price_panel, "pb_ratio")
