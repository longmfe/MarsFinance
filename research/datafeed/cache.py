# -*- coding: utf-8 -*-
"""akshare 取数结果的 parquet 磁盘缓存。

一次性建库要跑数十分钟（60 个报告期 × 3 个批量端点，外加逐股行情），
缓存让后续所有运行都离线且秒级。缓存目录默认在 ``data/akshare_cache``，
而 ``.gitignore`` 里的 ``data/`` 已经覆盖它，不会进版本库。

四个关键实现细节：

1. **原子写**：先写 ``.tmp`` 再 ``os.replace``，长时间建库被 Ctrl-C 打断
   不会留下截断的 parquet。
2. **dtype 归一**：eastmoney 的列常把 ``'-'`` 哨兵和浮点混在一个 object 列里，
   直接 ``to_parquet`` 会抛 ``ArrowInvalid``。
3. **空结果也缓存**：否则每次运行都会重打那些确实没有数据的报告期。
4. **不缓存异常**：重试退避后照常抛出，避免把一次网络抖动固化成"空数据"。
"""

import hashlib
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Optional

import pandas as pd

# object 列里常见的"无数据"哨兵，归一时先替换成 NaN
_NULL_SENTINELS = {"-", "--", "", "None", "none", "null", "NULL", "nan", "NaN"}

# object 列中有多大比例能转成数字，才判定它是数值列
_NUMERIC_RATIO_THRESHOLD = 0.9

_RETRY_ATTEMPTS = 3
_RETRY_BASE_SECONDS = 2.0


def cache_dir() -> Path:
    """缓存根目录：``$MARSFINANCE_CACHE_DIR`` 或 ``<repo>/data/akshare_cache``。"""
    env = os.environ.get("MARSFINANCE_CACHE_DIR")
    if env:
        return Path(env)
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "data" / "akshare_cache"


def cache_key(endpoint: str, params: Dict) -> str:
    """由端点名与参数算出稳定的缓存键（对参数顺序不敏感）。"""
    payload = json.dumps(params, sort_keys=True, ensure_ascii=False, default=str)
    digest = hashlib.sha1(f"{endpoint}|{payload}".encode("utf-8")).hexdigest()
    return digest[:16]


def normalize_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    """把 object 列归一成数值或字符串，使其可写入 parquet。

    对每个 object 列：先把哨兵值换成 NaN，再尝试整列转数字；若能转成功的
    比例达到阈值就采纳数值列（``'-'`` 变 NaN），否则整列转字符串。

    Args:
        df: 原始 DataFrame

    Returns:
        pd.DataFrame: 副本，所有 object 列已归一
    """
    out = df.copy()

    for col in out.columns:
        if out[col].dtype != object:
            continue

        cleaned = out[col].apply(
            lambda v: (
                pd.NA if isinstance(v, str) and v.strip() in _NULL_SENTINELS else v
            )
        )

        non_null = cleaned.notna().sum()
        if non_null == 0:
            out[col] = cleaned.astype("string")
            continue

        numeric = pd.to_numeric(cleaned, errors="coerce")
        if numeric.notna().sum() >= _NUMERIC_RATIO_THRESHOLD * non_null:
            out[col] = numeric
        else:
            out[col] = cleaned.astype("string")

    return out


def _meta_path(parquet_path: Path) -> Path:
    return parquet_path.with_suffix(".meta.json")


def _write_atomic(df: pd.DataFrame, path: Path, endpoint: str, params: Dict) -> None:
    """原子地写入 parquet 与同名 meta。"""
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)

    try:
        import akshare

        akshare_version = getattr(akshare, "__version__", "unknown")
    except Exception:
        akshare_version = "unknown"

    meta = {
        "endpoint": endpoint,
        "params": {k: str(v) for k, v in params.items()},
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "akshare_version": akshare_version,
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
    }

    meta_tmp = _meta_path(path).with_suffix(".json.tmp")
    meta_tmp.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(meta_tmp, _meta_path(path))


def _fetch_with_retry(fetch_fn: Callable[[], pd.DataFrame], label: str) -> pd.DataFrame:
    """带指数退避的重试；耗尽后原样抛出，**绝不把异常写进缓存**。"""
    last_error = None

    for attempt in range(_RETRY_ATTEMPTS):
        try:
            return fetch_fn()
        except Exception as exc:  # noqa: BLE001 - 网络异常种类繁多，统一退避重试
            last_error = exc
            if attempt == _RETRY_ATTEMPTS - 1:
                break
            delay = _RETRY_BASE_SECONDS * (2**attempt) + random.uniform(0, 1)
            print(
                f"  ⚠️  {label} 第 {attempt + 1} 次失败({exc.__class__.__name__})，"
                f"{delay:.1f}s 后重试"
            )
            time.sleep(delay)

    raise RuntimeError(
        f"{label} 取数失败（已重试 {_RETRY_ATTEMPTS} 次）"
    ) from last_error


def cached(
    endpoint: str,
    params: Dict,
    fetch_fn: Callable[[], pd.DataFrame],
    force: bool = False,
) -> pd.DataFrame:
    """读缓存，未命中则调用 ``fetch_fn`` 取数并落盘。

    Args:
        endpoint: 端点名，同时是缓存子目录名
        params: 入参，参与缓存键计算
        fetch_fn: 无参可调用对象，返回 DataFrame
        force: 为 True 时忽略已有缓存，强制重新取数

    Returns:
        pd.DataFrame: 归一化后的数据（空结果同样会被缓存并返回空表）
    """
    key = cache_key(endpoint, params)
    path = cache_dir() / endpoint / f"{key}.parquet"

    if path.exists() and not force:
        try:
            return pd.read_parquet(path)
        except Exception as exc:  # noqa: BLE001 - 损坏的缓存不应让整个流程失败
            print(f"  ⚠️  缓存损坏，重新取数 ({path.name}): {exc.__class__.__name__}")

    label = f"{endpoint}{params}"
    raw = _fetch_with_retry(fetch_fn, label)

    if raw is None:
        raw = pd.DataFrame()

    normalized = normalize_for_parquet(raw)
    _write_atomic(normalized, path, endpoint, params)
    return normalized


def is_cached(endpoint: str, params: Dict) -> bool:
    """该请求是否已在缓存中（供可续跑的建库脚本跳过已完成项）。"""
    return (cache_dir() / endpoint / f"{cache_key(endpoint, params)}.parquet").exists()


def cache_stats() -> Dict:
    """缓存概览：条目数、总大小、按端点分布、最早/最晚抓取时间。"""
    root = cache_dir()
    if not root.exists():
        return {"root": str(root), "exists": False, "n_entries": 0, "total_mb": 0.0}

    by_endpoint: Dict[str, int] = {}
    total_bytes = 0
    timestamps = []

    for path in root.rglob("*.parquet"):
        by_endpoint[path.parent.name] = by_endpoint.get(path.parent.name, 0) + 1
        total_bytes += path.stat().st_size

        meta_path = _meta_path(path)
        if meta_path.exists():
            try:
                timestamps.append(
                    json.loads(meta_path.read_text(encoding="utf-8"))["fetched_at"]
                )
            except Exception:  # noqa: BLE001 - meta 损坏不影响统计
                pass

    return {
        "root": str(root),
        "exists": True,
        "n_entries": sum(by_endpoint.values()),
        "total_mb": round(total_bytes / 1024 / 1024, 2),
        "by_endpoint": dict(sorted(by_endpoint.items())),
        "oldest_fetch": min(timestamps) if timestamps else None,
        "newest_fetch": max(timestamps) if timestamps else None,
    }
