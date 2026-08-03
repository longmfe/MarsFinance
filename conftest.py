# -*- coding: utf-8 -*-
"""pytest 共享配置与合成数据构造器。

仓库根目录存在 __init__.py，pytest 默认的 rootdir 插入会越过本仓库，
导致 ``import data_loader`` 失败——故此处显式把仓库根加到 sys.path 首位。
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(__file__))


@pytest.fixture
def make_prices():
    """构造合成日线行情：交易日索引为 'YYYYMMDD'，价格恒定。"""

    def _make(n=60, start="20240102", close=10.0):
        dates = [d.strftime("%Y%m%d") for d in pd.bdate_range(start, periods=n)]
        closes = np.full(n, float(close))
        return pd.DataFrame(
            {
                "open": closes,
                "close": closes,
                "high": closes * 1.01,
                "low": closes * 0.99,
                "volume": np.full(n, 1000.0),
                "amount": closes * 1000.0,
                "preClose": closes,
            },
            index=dates,
        )

    return _make


@pytest.fixture
def make_events():
    """构造单只股票的事件表（列同 yjyg_loader.load_yjyg_events）。"""

    def _make(notice_date, forecast_type="预增", amp=100.0, code="600000.SH"):
        return pd.DataFrame(
            {
                "code": [code],
                "notice_date": [notice_date],
                "period": ["20231231"],
                "type": [forecast_type],
                "amp": [float(amp)],
                "name": ["测试股份"],
            }
        )

    return _make


@pytest.fixture
def seal_limit_up():
    """把某根 bar 改成封死的一字涨停（low 也钉在涨停价上）。"""

    def _seal(df, pos, rate=0.10):
        pre = df["preClose"].iloc[pos]
        limit_price = round(pre * (1.0 + rate), 2)
        for col in ("open", "close", "high", "low"):
            df.iloc[pos, df.columns.get_loc(col)] = limit_price
        return df

    return _seal
