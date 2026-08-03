# -*- coding: utf-8 -*-
"""一次性建库脚本：把复现所需的全部数据抓进 parquet 缓存。

**可续跑**：已在缓存中的请求直接跳过，中途被打断后重跑即可接着走。
新浪与东方财富都有服务端限流，300 只的批量拉取中途被挡是常态。

用法::

    python research/scripts/build_cache.py --universe hs300 --start 2010 --end 2024
    python research/scripts/build_cache.py --stats          # 只看缓存概览

耗时（一次性，之后永久命中缓存）：

- 报告期批量三表：约 15s × 3 × 报告期数
- 逐股资产负债表明细：约 27s × 股票数（真实公告日期 + F6 所需流动资产/负债）
- 逐股日线：约 3.6s × 股票数
"""

import argparse
import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from research.datafeed.akshare_source import (  # noqa: E402
    fetch_balance_sheet,
    fetch_balance_sheet_detail,
    fetch_cashflow_statement,
    fetch_daily_bars,
    fetch_income_statement,
    fetch_index_constituents,
    fetch_index_daily,
)
from research.datafeed.cache import cache_stats, is_cached  # noqa: E402
from research.datafeed.calendar import report_periods  # noqa: E402
from research.datafeed.panel import normalize_code, to_em_symbol  # noqa: E402


def banner(text):
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


def get_universe(name):
    """取股票池代码列表。"""
    if name == "hs300":
        frame = fetch_index_constituents("000300")
        return [normalize_code(c) for c in frame["品种代码"]]
    if name == "zz500":
        frame = fetch_index_constituents("000905")
        return [normalize_code(c) for c in frame["品种代码"]]
    raise ValueError(f"未知股票池: {name!r}（支持 hs300 / zz500）")


def build_bulk(periods):
    """全市场三张批量财报，按报告期。"""
    banner(f"1/4 批量财报（{len(periods)} 个报告期 × 3 张表）")

    endpoints = [
        ("stock_zcfz_em", fetch_balance_sheet, "资产负债表"),
        ("stock_lrb_em", fetch_income_statement, "利润表"),
        ("stock_xjll_em", fetch_cashflow_statement, "现金流量表"),
    ]

    done = skipped = failed = 0
    started = time.time()

    for period in periods:
        for endpoint, fetch, label in endpoints:
            if is_cached(endpoint, {"date": period}):
                skipped += 1
                continue
            try:
                fetch(period)
                done += 1
                print(f"  ✅ {period} {label}")
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(f"  ❌ {period} {label}: {exc.__class__.__name__}")

    print(
        f"\n  新增 {done}，跳过 {skipped}，失败 {failed}，"
        f"耗时 {time.time() - started:.0f}s"
    )


def build_detail(codes):
    """逐股资产负债表明细：真实公告日期 + 流动资产/负债（F6）。"""
    banner(f"2/4 逐股资产负债表明细（{len(codes)} 只，约 27s/只）")

    done = skipped = failed = 0
    started = time.time()

    for i, code in enumerate(codes, 1):
        if is_cached(
            "stock_balance_sheet_by_report_em", {"symbol": to_em_symbol(code)}
        ):
            skipped += 1
            continue
        try:
            fetch_balance_sheet_detail(code)
            done += 1
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  ❌ {code}: {exc.__class__.__name__}")

        if i % 10 == 0:
            elapsed = time.time() - started
            rate = elapsed / max(done, 1)
            remaining = (len(codes) - i) * rate / 60
            print(
                f"  [{i}/{len(codes)}] 新增 {done} 跳过 {skipped} 失败 {failed}"
                f" | 预计剩余 {remaining:.0f} 分钟"
            )

    print(
        f"\n  新增 {done}，跳过 {skipped}，失败 {failed}，"
        f"耗时 {(time.time() - started) / 60:.1f} 分钟"
    )


def build_prices(codes, start, end):
    """逐股日线（后复权，含流通股本 —— F7 需要）。"""
    banner(f"3/4 逐股日线（{len(codes)} 只，约 3.6s/只）")

    done = skipped = failed = 0
    started = time.time()

    for i, code in enumerate(codes, 1):
        params = {"symbol": None, "start": start, "end": end, "adjust": "hfq"}
        try:
            from research.datafeed.panel import to_sina_symbol

            params["symbol"] = to_sina_symbol(code)
        except ValueError:
            skipped += 1
            continue

        if is_cached("stock_zh_a_daily", params):
            skipped += 1
            continue

        try:
            fetch_daily_bars(code, start_date=start, end_date=end)
            done += 1
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  ❌ {code}: {exc.__class__.__name__}")

        if i % 50 == 0:
            print(f"  [{i}/{len(codes)}] 新增 {done} 跳过 {skipped} 失败 {failed}")

    print(
        f"\n  新增 {done}，跳过 {skipped}，失败 {failed}，"
        f"耗时 {(time.time() - started) / 60:.1f} 分钟"
    )


def build_indices():
    """基准与交易日历所需的指数日线。"""
    banner("4/4 指数日线（基准 + 交易日历）")

    for symbol in ("sh000300", "sh000905", "sh000852", "sh000012", "sh000013"):
        try:
            frame = fetch_index_daily(symbol)
            print(f"  ✅ {symbol}: {len(frame)} 根")
        except Exception as exc:  # noqa: BLE001
            print(f"  ❌ {symbol}: {exc.__class__.__name__}")


def main():
    parser = argparse.ArgumentParser(description="一次性建库（可续跑）")
    parser.add_argument("--universe", default="hs300", help="股票池：hs300 / zz500")
    parser.add_argument("--start", type=int, default=2010, help="起始年份")
    parser.add_argument("--end", type=int, default=2024, help="结束年份")
    parser.add_argument(
        "--annual-only",
        action="store_true",
        help="只抓年报报告期（Piotroski 年度版足够）",
    )
    parser.add_argument(
        "--skip-detail",
        action="store_true",
        help="跳过逐股明细（会退化为 8 信号且公告日期用兜底）",
    )
    parser.add_argument("--stats", action="store_true", help="只打印缓存概览")
    args = parser.parse_args()

    if args.stats:
        for key, value in cache_stats().items():
            print(f"{key:15s}: {value}")
        return

    started = time.time()
    periods = report_periods(
        f"{args.start}1231", f"{args.end}1231", annual_only=args.annual_only
    )

    banner(
        f"建库开始：{args.universe}，{args.start}-{args.end}，"
        f"{len(periods)} 个报告期"
    )

    codes = get_universe(args.universe)
    print(f"股票池: {len(codes)} 只")

    build_bulk(periods)
    if not args.skip_detail:
        build_detail(codes)
    build_prices(codes, f"{args.start}0101", f"{args.end}1231")
    build_indices()

    banner(f"建库完成，总耗时 {(time.time() - started) / 60:.1f} 分钟")
    for key, value in cache_stats().items():
        print(f"{key:15s}: {value}")


if __name__ == "__main__":
    main()
