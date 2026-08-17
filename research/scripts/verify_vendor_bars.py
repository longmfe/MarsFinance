# -*- coding: utf-8 -*-
"""第三方日线与本地新浪缓存的逐日口径比对。

动机：接入任何新数据源（小石大数据等）之前，先确认它的复权日线能不能跟仓库
现有的新浪后复权口径对上。**价格绝对值不可比** —— 两家后复权的基准日不同，
整条序列本来就差一个常数倍 —— 所以真正的检验是两条：

1. **收益率逐日比对**（给结论）：复权基准无关。乘法复权下，前复权/后复权/
   不同基准日都只改变一个常数因子，收益率序列应当完全相同。对不上只有两种
   可能：分红送配的处理不同，或者对方用的是加法复权。
2. **复权比值恒定性**（给位置）：``本地 close / 对方 close`` 应当是常数。
   比值发生跳变的那些日期就是分歧的确切位置，拿去对当天的分红送配公告即可定位。

再加两条便宜的检查：交易日集合差异（暴露停牌与日历口径），以及成交量比值中位数
（暴露"股 vs 手"这类 100 倍单位坑）。

本脚本**不联网**：本地一侧直读 parquet 缓存，对方一侧读你导出的文件。

用法::

    # 对方文件含 code 列（按年份批量下载的形态）
    python research/scripts/verify_vendor_bars.py --vendor 2023.csv

    # 单只股票，文件里没有 code 列
    python research/scripts/verify_vendor_bars.py --vendor 600519.csv --code 600519

本地缓存里现有的是 ``--start 20100101 --end 20241231 --adjust hfq``（即默认值），
对方导出请取同一区间、后复权口径。
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from research.datafeed.cache import cache_dir, cache_key  # noqa: E402
from research.datafeed.panel import normalize_code, to_sina_symbol  # noqa: E402

#: 对方导出可能用的列名 → 规范列名。大小写不敏感。
COLUMN_ALIASES = {
    "date": ("date", "日期", "trade_date", "datetime", "time", "时间"),
    "open": ("open", "开盘", "开盘价"),
    "high": ("high", "最高", "最高价"),
    "low": ("low", "最低", "最低价"),
    "close": ("close", "收盘", "收盘价"),
    "volume": ("volume", "vol", "成交量"),
    "code": ("code", "symbol", "代码", "股票代码", "ts_code"),
}


def banner(text):
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


def load_vendor(path):
    """读对方导出的文件（csv / parquet），列名归一到规范形式。

    Args:
        path: 文件路径

    Returns:
        pd.DataFrame: 含 date/close 及可选的 open/high/low/volume/code

    Raises:
        ValueError: 缺少 date 或 close 列
    """
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_csv(path)

    lookup = {str(c).strip().lower(): c for c in frame.columns}
    rename = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in lookup:
                rename[lookup[alias]] = canonical
                break

    frame = frame.rename(columns=rename)

    missing = {"date", "close"} - set(frame.columns)
    if missing:
        raise ValueError(
            f"{path.name} 缺少必需列 {sorted(missing)}；"
            f"现有列: {list(frame.columns)}"
        )

    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    return frame


def load_local(code, start, end, adjust):
    """从 parquet 缓存读本地新浪日线（不联网）。

    Args:
        code: 任意格式的股票代码
        start: 起始日 ``YYYYMMDD``，须与建库时一致
        end: 结束日 ``YYYYMMDD``，须与建库时一致
        adjust: 复权口径，须与建库时一致

    Returns:
        pd.DataFrame: 以 date 为索引，含 close 及可选的 volume

    Raises:
        FileNotFoundError: 该请求不在缓存中
    """
    params = {
        "symbol": to_sina_symbol(code),
        "start": start,
        "end": end,
        "adjust": adjust,
    }
    key = cache_key("stock_zh_a_daily", params)
    path = cache_dir() / "stock_zh_a_daily" / f"{key}.parquet"

    if not path.exists():
        raise FileNotFoundError(f"缓存未命中 {params}")

    frame = pd.read_parquet(path)
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    return frame.drop_duplicates("date").set_index("date").sort_index()


def compare(local, vendor, ret_tol, ratio_tol):
    """比对两条日线，返回指标字典。

    Args:
        local: 本地日线，date 索引
        vendor: 对方日线，date 索引
        ret_tol: 收益率逐日偏差容差
        ratio_tol: 复权比值跳变容差

    Returns:
        dict: 见下方各键；``n_common`` 为 0 时只有覆盖度相关的键有意义
    """
    local_dates, vendor_dates = set(local.index), set(vendor.index)
    common = sorted(local_dates & vendor_dates)

    report = {
        "n_local": len(local_dates),
        "n_vendor": len(vendor_dates),
        "n_common": len(common),
        "only_local": sorted(local_dates - vendor_dates),
        "only_vendor": sorted(vendor_dates - local_dates),
    }
    if not common:
        return report

    local_close = pd.to_numeric(local.loc[common, "close"], errors="coerce")
    vendor_close = pd.to_numeric(vendor.loc[common, "close"], errors="coerce")

    # 收益率：复权基准无关，两家应当逐日相同
    ret_diff = (local_close.pct_change() - vendor_close.pct_change()).abs().dropna()
    report["ret_max_diff"] = float(ret_diff.max()) if len(ret_diff) else float("nan")
    report["ret_bad_days"] = ret_diff[ret_diff > ret_tol]

    # 复权比值：应当恒定，跳变处即分歧位置
    ratio = (local_close / vendor_close).replace([float("inf"), -float("inf")], pd.NA)
    ratio = pd.to_numeric(ratio, errors="coerce").dropna()
    if len(ratio) > 1:
        step = (ratio / ratio.shift(1) - 1.0).abs().dropna()
        report["ratio_breaks"] = step[step > ratio_tol]
        report["ratio_spread"] = float(ratio.max() / ratio.min() - 1.0)
    else:
        report["ratio_breaks"] = pd.Series(dtype=float)
        report["ratio_spread"] = float("nan")

    # 成交量单位：100 倍即"股 vs 手"
    if "volume" in local.columns and "volume" in vendor.columns:
        lv = pd.to_numeric(local.loc[common, "volume"], errors="coerce")
        vv = pd.to_numeric(vendor.loc[common, "volume"], errors="coerce")
        vol_ratio = (lv / vv.where(vv > 0)).dropna()
        report["vol_ratio"] = float(vol_ratio.median()) if len(vol_ratio) else None

    return report


def verdict(report, ret_tol):
    """把指标翻成一句结论。

    Returns:
        tuple: ``(是否通过, 结论文本)``
    """
    if report["n_common"] == 0:
        return False, "无重叠交易日，无法比对"

    n_bad = len(report["ret_bad_days"])
    n_breaks = len(report["ratio_breaks"])

    if n_bad == 0:
        if report["only_local"] or report["only_vendor"]:
            return True, "收益率完全一致，但交易日集合有差异（见下）"
        return True, "口径一致"

    if n_bad <= 5 and n_breaks <= 5:
        return False, f"{n_bad} 天收益率不一致 —— 疑似个别分红送配处理不同"
    if report["ratio_spread"] > 0.5:
        return False, f"{n_bad} 天收益率不一致，比值漂移 >50% —— 疑似对方未复权"
    return False, f"{n_bad} 天收益率不一致（>{ret_tol:g}），复权算法可能不同"


def print_report(code, report, ret_tol, max_show):
    """打印单只股票的比对详情。"""
    ok, text = verdict(report, ret_tol)
    print(f"\n{'✅' if ok else '❌'} {code}  {text}")
    print(
        f"   交易日 本地 {report['n_local']} / 对方 {report['n_vendor']}"
        f" / 重叠 {report['n_common']}"
    )

    if report["n_common"] == 0:
        return ok

    for label, dates in (("仅本地有", report["only_local"]),
                         ("仅对方有", report["only_vendor"])):
        if dates:
            shown = ", ".join(d.strftime("%Y-%m-%d") for d in dates[:max_show])
            more = f" …等 {len(dates)} 天" if len(dates) > max_show else ""
            print(f"   {label}: {shown}{more}")

    print(f"   收益率最大偏差 {report['ret_max_diff']:.2e}")

    if report["vol_ratio"] is not None:
        note = "（≈100，对方按手）" if 95 < report["vol_ratio"] < 105 else ""
        print(f"   成交量比值中位数 {report['vol_ratio']:.4g}{note}")

    breaks = report["ratio_breaks"]
    if len(breaks):
        print(f"   复权比值跳变 {len(breaks)} 处，最早的几处（对当天分红送配公告）：")
        for date, step in breaks.head(max_show).items():
            print(f"     {date:%Y-%m-%d}  跳变 {step:+.4%}")

    return ok


def main():
    parser = argparse.ArgumentParser(
        description="第三方日线与本地新浪缓存的逐日口径比对（不联网）"
    )
    parser.add_argument("--vendor", required=True, help="对方导出文件 csv/parquet")
    parser.add_argument("--code", help="股票代码；对方文件无 code 列时必填")
    parser.add_argument("--start", default="20100101", help="须与建库参数一致")
    parser.add_argument("--end", default="20241231", help="须与建库参数一致")
    parser.add_argument("--adjust", default="hfq", help="须与建库参数一致")
    parser.add_argument("--ret-tol", type=float, default=1e-4, help="收益率偏差容差")
    parser.add_argument("--ratio-tol", type=float, default=1e-4, help="比值跳变容差")
    parser.add_argument("--limit", type=int, help="最多比对几只")
    parser.add_argument("--max-show", type=int, default=8, help="每类明细最多列几行")
    args = parser.parse_args()

    vendor_all = load_vendor(args.vendor)

    if "code" in vendor_all.columns:
        groups = [(c, g) for c, g in vendor_all.groupby("code", sort=True)]
    elif args.code:
        groups = [(args.code, vendor_all)]
    else:
        parser.error("对方文件没有 code 列，请用 --code 指定股票代码")

    if args.limit:
        groups = groups[: args.limit]

    banner(f"口径比对：{Path(args.vendor).name} vs 本地新浪 {args.adjust}（{len(groups)} 只）")

    passed = failed = skipped = 0

    for raw_code, group in groups:
        try:
            code = normalize_code(raw_code)
        except ValueError as exc:
            print(f"\n⚠️  {raw_code} 跳过：{exc}")
            skipped += 1
            continue

        try:
            local = load_local(code, args.start, args.end, args.adjust)
        except FileNotFoundError:
            print(f"\n⚠️  {code} 跳过：本地缓存没有这只（先跑 build_cache.py）")
            skipped += 1
            continue
        except ValueError as exc:  # 北交所等新浪不支持的标的
            print(f"\n⚠️  {code} 跳过：{exc}")
            skipped += 1
            continue

        vendor = (
            group.drop_duplicates("date").set_index("date").sort_index()
        )
        report = compare(local, vendor, args.ret_tol, args.ratio_tol)

        if print_report(code, report, args.ret_tol, args.max_show):
            passed += 1
        else:
            failed += 1

    banner(f"通过 {passed}，不通过 {failed}，跳过 {skipped}")

    if failed:
        print(
            "\n不通过不等于对方错 —— 先拿上面列出的跳变日期对当天的分红送配公告，"
            "确认是谁的处理有问题，再决定接不接。"
        )

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
