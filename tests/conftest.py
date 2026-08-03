# -*- coding: utf-8 -*-
"""共享 fixtures。

设计原则：默认跑的测试**一次网络都不能发**。``tmp_cache`` 把缓存目录重定向到
临时目录，保证没有任何测试会碰真实的 ``data/akshare_cache``。
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def tmp_cache(tmp_path, monkeypatch):
    """把缓存根目录重定向到临时目录，返回该路径。"""
    monkeypatch.setenv("MARSFINANCE_CACHE_DIR", str(tmp_path / "akshare_cache"))
    return tmp_path / "akshare_cache"


@pytest.fixture
def counting_fetch():
    """返回 (fetch_fn, calls)：calls 是列表，每调用一次追加一项。"""
    calls = []

    def make(df=None):
        if df is None:
            df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})

        def fetch():
            calls.append(1)
            return df.copy()

        return fetch

    return make, calls


@pytest.fixture
def price_panel():
    """确定性合成行情面板：3 只股票 × 60 个交易日，MultiIndex (date, code)。"""
    dates = pd.bdate_range("2023-01-02", periods=60)
    codes = ["600000.SH", "000001.SZ", "300750.SZ"]

    rng = np.random.default_rng(42)
    frames = []

    for i, code in enumerate(codes):
        returns = rng.normal(0.0005 * (i + 1), 0.015, len(dates))
        close = 10.0 * (1 + i) * np.cumprod(1 + returns)
        frames.append(
            pd.DataFrame(
                {
                    "open": close * 0.995,
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "close": close,
                    "volume": rng.integers(1e6, 5e6, len(dates)).astype(float),
                    "amount": close * 1e6,
                    "outstanding_share": np.full(len(dates), 1e9 * (i + 1)),
                },
                index=pd.MultiIndex.from_arrays(
                    [dates, [code] * len(dates)], names=["date", "code"]
                ),
            )
        )

    return pd.concat(frames).sort_index()
