# -*- coding: utf-8 -*-
"""FFScore 全量回测驱动：数据源 = mars-invest-os 的 PG。

先跑完 ``mars-invest-os/collectors/stock_fundamentals.py`` 全量采集（18 字段、
最早披露口径），本脚本再：

1. 读 ``stock_fundamentals`` → 报告面板（(code, period)，FY 年报 + 首次公告日）
2. ``add_lagged`` + ``add_available_date`` + ``compute_ffscore`` → 报告期 F-Score
3. ``build_pit_panel`` 把成品 PIT 对齐到每年 5 月首个交易日（年报 4/30 截止）
4. ``screen_universe`` 股票池筛选（金融股剔除等，见 config.screen_rules）
5. ``CrossSectionalBacktest`` 截面分组回测 + 单信号有效性

与默认 akshare 数据路径的差异（均因 PG 数据面）：

- BM 用 **PIT 总股本**（``stock_fundamentals.shares`` = ``cap_stk`` 实收资本，
  A 股面值多为 1 元故 ≈ 总股数）× 调仓日 close = 总市值。原版 Piotroski 亦常用
  总市值口径；无流通股本，故按 ``--bm-quantile`` 取 BM 最高分位（原版 0.2）。
- 无成交量 → ``exclude_suspended=False``。
- 行情从 2010-01-04 起，故调仓窗口取 2011-05 ~ 2025-05。

用法：
    python -m research.scripts.backtest_ffscore_pg [--out DIR] [--bm-quantile 0.2]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from research.datafeed.fundamentals import (
    add_available_date,
    add_lagged,
    build_pit_panel,
    screen_universe,
)
from research.ffscore.backtest import CrossSectionalBacktest, signal_efficacy
from research.ffscore.config import default_config
from research.ffscore.score import compute_ffscore
from research.ffscore.signals import LAG_COLUMNS, SIGNAL_NAMES
from research.ffscore.universe import select_high_bm

PG_DSN = "postgresql://postgres:mars@localhost:5432/mars_invest"

# PG 列名 → research 报告面板规范列名（其余同名直接透传）
COLUMN_MAP = {
    "tot_assets": "total_assets",
    "ocf": "cfo",
}

#: 调仓年份（每年 5 月首个交易日）。所需年报为上年 FY（及 lag 两年的历史）。
REBALANCE_YEARS = range(2011, 2026)
FY_FROM = 2008      # FY2010 的 lag2 = FY2008，足够覆盖首次调仓 2011-05
PRICE_FROM = "2010-01-01"  # stock_daily_close 实测自 2010-01-04


def load_fundamentals(engine, fy_from: int = FY_FROM) -> pd.DataFrame:
    """stock_fundamentals → (code, period) 报告面板，ann_date 为首次公告日。"""
    sql = text(
        """
        SELECT code, fy, ann_date, name,
               tot_assets, total_liab, total_equity,
               current_assets, current_liab, noncurrent_liab, shares,
               net_profit, revenue, operating_cost, ocf
        FROM stock_fundamentals
        WHERE fy >= :fy_from
        """
    )
    df = pd.read_sql(sql, engine, params={"fy_from": fy_from})
    if df.empty:
        raise RuntimeError("stock_fundamentals 为空，先跑 collectors.stock_fundamentals")

    df = df.rename(columns=COLUMN_MAP)
    df["period"] = pd.to_datetime(df["fy"].astype(str) + "-12-31")
    df["ann_date"] = pd.to_datetime(
        df["ann_date"].astype(str), format="%Y%m%d", errors="coerce"
    )
    df["fiscal_year"] = df["fy"]
    df["quarter"] = 4
    df = df.drop(columns=["fy"])

    panel = df.set_index(["code", "period"]).sort_index()
    n_valid_ann = panel["ann_date"].notna().sum()
    print(f"  fundamentals: {len(panel)} records, {panel.index.get_level_values('code').nunique()} codes"
          f", {int(n_valid_ann)} with ann_date")
    return panel


def load_price_panel(engine, date_from: str = PRICE_FROM) -> pd.DataFrame:
    """stock_daily_close → MultiIndex (date, code) 行情面板，仅 close。"""
    sql = text(
        """
        SELECT report_date AS date, stock_code AS code, close
        FROM stock_daily_close
        WHERE report_date >= :date_from
        """
    )
    df = pd.read_sql(sql, engine, params={"date_from": date_from})
    df["date"] = pd.to_datetime(df["date"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["close"])

    panel = df.set_index(["date", "code"]).sort_index()
    days = panel.index.get_level_values("date").unique()
    print(f"  price: {len(panel)} rows, {panel.index.get_level_values('code').nunique()} codes, "
          f"{days.min().date()}..{days.max().date()}")
    return panel


def load_benchmark(engine, name: str = "HS300") -> pd.Series:
    """benchmarks 表某指数的 close → 按日排序的 Series。"""
    df = pd.read_sql(
        text("SELECT report_date, value FROM benchmarks WHERE name=:n AND metric='close'"),
        engine,
        params={"n": name},
    )
    df["report_date"] = pd.to_datetime(df["report_date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    s = df.dropna(subset=["value"]).drop_duplicates(subset=["report_date"]).set_index(
        "report_date"
    )["value"].sort_index()
    return s


def annual_may_rebalance_dates(price_panel: pd.DataFrame, years) -> pd.DatetimeIndex:
    """每年 5 月首个交易日。调仓日即"信号日"，成交在次日（execution_lag=1）。"""
    days = pd.DatetimeIndex(sorted(price_panel.index.get_level_values("date").unique()))
    out = []
    for year in years:
        may = days[days >= pd.Timestamp(year=year, month=5, day=1)]
        if len(may):
            out.append(may[0])
    return pd.DatetimeIndex(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(Path(__file__).resolve().parent / ".." / "ffscore_pg_output"))
    parser.add_argument("--benchmark", default="HS300")
    parser.add_argument("--min-signals", type=int, default=9)
    parser.add_argument("--bm-quantile", type=float, default=0.2,
                        help="BM 最高分位（Piotroski 原版 0.2；1.0 = 不做价值筛选）")
    args = parser.parse_args()

    config = default_config(
        bm_quantile=args.bm_quantile,  # 原版先取 BM 最高五分位
        min_signals=args.min_signals,  # 9：信号完整才算分（数据面已齐）
        frequency="annual",
        rebalance="annual_may",
    )
    print("config:", config.to_dict())

    engine = create_engine(PG_DSN)
    print("loading data ...")
    report_panel = load_fundamentals(engine)
    price_panel = load_price_panel(engine)
    benchmark = load_benchmark(engine, args.benchmark)
    print(f"  benchmark {args.benchmark}: {benchmark.index.min().date()}..{benchmark.index.max().date()}")

    # 1. 报告期空间：滞后列 → 可用时点 → F-Score
    print("computing signals ...")
    lagged = add_lagged(report_panel, LAG_COLUMNS, lags=(1, 2))
    dated = add_available_date(lagged)
    scored = compute_ffscore(dated, config)

    # compute_ffscore 不透传 equity/shares，回填供 BM 与净资产筛选用
    scored = scored.join(dated[["total_equity", "shares"]], how="left")

    n_scored = scored["f_score"].notna().sum()
    print(f"  scored: {n_scored} records with valid f_score")
    dist = scored["f_score"].dropna().round().astype(int).value_counts().sort_index()
    print("  f_score distribution:")
    for k, v in dist.items():
        print(f"    {k}: {v}")

    # 2. PIT 对齐到调仓日 + 股票池筛选
    rebalance_dates = annual_may_rebalance_dates(price_panel, REBALANCE_YEARS)
    print(f"  rebalance dates: {list(rebalance_dates)}")

    fields = ["f_score", "name", "total_equity", "shares", "available_date", "period"] + list(SIGNAL_NAMES)
    pit = build_pit_panel(scored, rebalance_dates, fields=fields)

    rules = {**config.screen_rules(), "exclude_suspended": False}
    mask = screen_universe(pit, price_panel, rules)
    print(f"  pit: {len(pit)} rows, {int(mask.sum())} pass universe screen")

    # BM 价值筛选：BM = 净资产(PIT) / (PIT 总股本 × 调仓日 close)
    if config.bm_quantile < 1.0:
        close = pd.to_numeric(price_panel["close"].reindex(pit.index), errors="coerce")
        shares = pd.to_numeric(pit["shares"], errors="coerce")
        equity = pd.to_numeric(pit["total_equity"], errors="coerce")
        mcap = (close * shares).where((close * shares) > 0, np.nan)
        bm = equity.where(equity > 0, np.nan) / mcap
        bm_mask = select_high_bm(bm, quantile=config.bm_quantile, mask=mask)
        mask = mask & bm_mask
        print(f"  BM top {config.bm_quantile:.0%}: {int(bm_mask.sum())} pass value screen"
              f", median BM={bm[bm_mask].median():.4f}")

    # 每期各组持仓数（判断极端分值的样本量）
    factor_scored = (
        pit["f_score"].where(mask.reindex(pit.index).fillna(False)).dropna().round().astype(int)
    )
    group_sizes = factor_scored.groupby(level="date").value_counts().unstack(fill_value=0)
    print("\n  各组每期平均持仓数:")
    print(f"    {group_sizes.mean(axis=0).round(1).to_dict()}")

    # 3. 截面分组回测
    bt = CrossSectionalBacktest(
        group_by=config.group_by,
        n_groups=config.n_groups,
        weighting="equal",
        commission=config.commission,
        slippage=config.slippage,
        stamp_tax=config.stamp_tax,
        execution_lag=config.execution_lag,
    )
    result = bt.run(factor=pit["f_score"], price_panel=price_panel, universe_mask=mask)

    # 4. 输出
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = result["metrics_by_group"]
    table = pd.DataFrame(metrics).T.sort_index(key=lambda s: s.astype(int))
    cols = ["total_return", "annual_return", "annual_vol", "sharpe", "max_drawdown", "n_periods"]
    table = table[cols].round(3)
    print("\n=== 分组绩效（按 F-Score 分值，等权，2011-05 ~ 2025-05）===")
    print(table.to_string())

    print("\n=== 多空（最高分 - 最低分）===")
    ls = result["long_short_metrics"]
    if ls:
        print(pd.Series(ls).round(3).to_string())

    print("\n=== RankIC ===")
    print(pd.Series(result["ic_stats"]).round(3).to_string())

    print("\n=== 单信号有效性（价值池内，信号=1 与信号=0 组在下一调仓期平均收益差）===")
    efficacy = signal_efficacy(pit[mask], price_panel, SIGNAL_NAMES)
    eff = efficacy.sort_values("mean_spread", ascending=False)
    print(eff.round(4).to_string())

    # 基准对比：对齐到回测收益的日期范围
    group_returns = result["group_returns"]
    bm_ret = benchmark.pct_change(fill_method=None).reindex(group_returns.index).dropna()
    bm_nav = (1.0 + bm_ret).cumprod()
    print(f"\n  {args.benchmark} 同期净值 {bm_nav.iloc[-1]:.2f}（{bm_ret.index[0].date()}..{bm_ret.index[-1].date()}）")

    # 存盘
    result["group_nav"].to_csv(out_dir / "group_nav.csv")
    result["group_returns"].to_csv(out_dir / "group_returns.csv")
    pd.DataFrame(result["metrics_by_group"]).T.to_csv(out_dir / "metrics.csv")
    result["ic"].to_csv(out_dir / "ic.csv")
    result["turnover"].to_csv(out_dir / "turnover.csv")
    eff.to_csv(out_dir / "signal_efficacy.csv")
    pit[["f_score"]].to_csv(out_dir / "pit_fscore.csv")
    print(f"\n  outputs -> {out_dir}")


if __name__ == "__main__":
    main()
