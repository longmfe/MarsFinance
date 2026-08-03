# -*- coding: utf-8 -*-
"""parquet 缓存的单元测试（全离线，缓存目录重定向到 tmp_path）。"""

import numpy as np
import pandas as pd
import pytest

from research.datafeed.cache import (
    cache_dir,
    cache_key,
    cache_stats,
    cached,
    is_cached,
    normalize_for_parquet,
)


class TestCacheKey:
    def test_stable_under_param_reordering(self):
        assert cache_key("ep", {"a": 1, "b": 2}) == cache_key("ep", {"b": 2, "a": 1})

    def test_differs_by_endpoint(self):
        assert cache_key("ep1", {"a": 1}) != cache_key("ep2", {"a": 1})

    def test_differs_by_params(self):
        assert cache_key("ep", {"a": 1}) != cache_key("ep", {"a": 2})

    def test_handles_non_serializable(self):
        assert isinstance(cache_key("ep", {"d": pd.Timestamp("2023-01-01")}), str)


class TestNormalizeForParquet:
    def test_mixed_sentinel_and_float_becomes_numeric(self):
        """eastmoney 的典型形态：'-' 哨兵混在浮点列里。"""
        df = pd.DataFrame({"x": [1.5, "-", 2.5, 3.5, "-"]})
        out = normalize_for_parquet(df)

        assert pd.api.types.is_numeric_dtype(out["x"])
        assert out["x"].isna().sum() == 2
        assert out["x"].iloc[0] == pytest.approx(1.5)

    def test_mostly_text_column_stays_text(self):
        df = pd.DataFrame({"name": ["贵州茅台", "平安银行", "1"]})
        out = normalize_for_parquet(df)

        assert not pd.api.types.is_numeric_dtype(out["name"])
        assert out["name"].iloc[0] == "贵州茅台"

    def test_all_sentinel_column(self):
        out = normalize_for_parquet(pd.DataFrame({"x": ["-", "-", "-"]}))
        assert out["x"].isna().all()

    def test_leaves_numeric_dtypes_alone(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [1.5, 2.5, 3.5]})
        pd.testing.assert_frame_equal(normalize_for_parquet(df), df)

    def test_result_is_parquet_writable(self, tmp_path):
        """归一化的意义就在于此：原始混合列 to_parquet 会抛错。"""
        df = pd.DataFrame({"x": [1.5, "-", 2.5], "name": ["a", "b", "c"]})
        normalize_for_parquet(df).to_parquet(tmp_path / "ok.parquet", index=False)

    def test_does_not_mutate_input(self):
        df = pd.DataFrame({"x": [1.5, "-"]})
        before = df["x"].tolist()
        normalize_for_parquet(df)
        assert df["x"].tolist() == before


class TestCached:
    def test_cache_dir_honours_env(self, tmp_cache):
        assert cache_dir() == tmp_cache

    def test_fetch_called_once_then_served_from_disk(self, tmp_cache, counting_fetch):
        make, calls = counting_fetch
        fetch = make()

        first = cached("demo", {"p": 1}, fetch)
        second = cached("demo", {"p": 1}, fetch)

        assert len(calls) == 1, "第二次必须命中缓存，不得再次取数"
        pd.testing.assert_frame_equal(first, second)

    def test_different_params_fetch_separately(self, tmp_cache, counting_fetch):
        make, calls = counting_fetch
        fetch = make()

        cached("demo", {"p": 1}, fetch)
        cached("demo", {"p": 2}, fetch)

        assert len(calls) == 2

    def test_force_refetches(self, tmp_cache, counting_fetch):
        make, calls = counting_fetch
        fetch = make()

        cached("demo", {"p": 1}, fetch)
        cached("demo", {"p": 1}, fetch, force=True)

        assert len(calls) == 2

    def test_empty_frame_is_cached_not_refetched(self, tmp_cache, counting_fetch):
        """空结果必须落盘，否则每次运行都会重打没有数据的报告期。"""
        make, calls = counting_fetch
        fetch = make(pd.DataFrame({"a": pd.Series(dtype=float)}))

        first = cached("demo", {"p": 1}, fetch)
        second = cached("demo", {"p": 1}, fetch)

        assert len(first) == 0
        assert len(second) == 0
        assert len(calls) == 1

    def test_exception_is_not_cached(self, tmp_cache):
        """网络异常不得被固化成一条空缓存。"""
        attempts = []

        def failing():
            attempts.append(1)
            raise ConnectionError("boom")

        with pytest.raises(RuntimeError, match="取数失败"):
            cached("demo", {"p": 9}, failing)

        assert not is_cached("demo", {"p": 9})
        assert len(attempts) == 3, "应重试 3 次"

    def test_corrupt_parquet_triggers_refetch(self, tmp_cache, counting_fetch):
        make, calls = counting_fetch
        fetch = make()

        cached("demo", {"p": 1}, fetch)

        path = next((tmp_cache / "demo").glob("*.parquet"))
        path.write_bytes(b"not a parquet file")

        out = cached("demo", {"p": 1}, fetch)

        assert len(calls) == 2
        assert len(out) == 3

    def test_normalizes_before_writing(self, tmp_cache, counting_fetch):
        make, _ = counting_fetch
        fetch = make(pd.DataFrame({"x": [1.5, "-", 2.5]}))

        out = cached("demo", {"p": 1}, fetch)

        assert pd.api.types.is_numeric_dtype(out["x"])
        assert np.isnan(out["x"].iloc[1])

    def test_writes_meta_sidecar(self, tmp_cache, counting_fetch):
        make, _ = counting_fetch
        cached("demo", {"p": 1}, make())

        metas = list((tmp_cache / "demo").glob("*.meta.json"))
        assert len(metas) == 1

        import json

        meta = json.loads(metas[0].read_text(encoding="utf-8"))
        assert meta["endpoint"] == "demo"
        assert meta["n_rows"] == 3
        assert "fetched_at" in meta

    def test_leaves_no_tmp_files(self, tmp_cache, counting_fetch):
        make, _ = counting_fetch
        cached("demo", {"p": 1}, make())
        assert list(tmp_cache.rglob("*.tmp")) == []


class TestIsCachedAndStats:
    def test_is_cached(self, tmp_cache, counting_fetch):
        make, _ = counting_fetch

        assert not is_cached("demo", {"p": 1})
        cached("demo", {"p": 1}, make())
        assert is_cached("demo", {"p": 1})

    def test_stats_on_missing_dir(self, tmp_cache):
        stats = cache_stats()
        assert stats["exists"] is False
        assert stats["n_entries"] == 0

    def test_stats_counts_entries(self, tmp_cache, counting_fetch):
        make, _ = counting_fetch
        fetch = make()

        cached("ep_a", {"p": 1}, fetch)
        cached("ep_a", {"p": 2}, fetch)
        cached("ep_b", {"p": 1}, fetch)

        stats = cache_stats()

        assert stats["n_entries"] == 3
        assert stats["by_endpoint"] == {"ep_a": 2, "ep_b": 1}
        assert stats["newest_fetch"] is not None
