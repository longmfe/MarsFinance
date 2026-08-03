# -*- coding: utf-8 -*-
"""事件级统计：把成交流水还原成「每次事件一行」并做汇总。

为什么不用 ``PortfolioBacktest.get_portfolio_metrics()``：
它把约 300 个彼此独立的账户做**不加权算术平均**，而事件策略下每个账户
除了事件后的十来根 bar 之外都空仓。平均 300 条几乎水平的净值曲线，量出来的是
「事件多久触发一次」，不是「事件到底赚不赚钱」。事件研究的分析单位是**事件**。

收益口径用 ``(revenue - cost) / cost``：cost 含买入侧佣金、revenue 已扣卖出侧
佣金，因此这是真实的净收益——比引擎里 ``stock_backtest.py`` 的单笔 ``profit``
更准（后者的 ``position_price`` 只含滑点不含买入佣金，略偏乐观）。
"""

from typing import List, Optional

import numpy as np
import pandas as pd


def summarize_events(
    all_trades: List[dict], events: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """把 BUY/SELL 流水配对成每次事件一行。

    ``self.trades`` 里 BUY 与 SELL 是两条独立记录且没有关联 id，
    因此按股票分组后依时间顺序配对。

    Args:
        all_trades: ``PortfolioBacktest.all_trades``
        events: 可选的事件表（``load_yjyg_events`` 输出），
            用于回填每笔交易对应的预告类型

    Returns:
        pd.DataFrame: 列为 stock / buy_date / sell_date / cost / revenue /
            ret / sub_type / yjyg_type
    """
    if not all_trades:
        return pd.DataFrame(
            columns=[
                "stock",
                "buy_date",
                "sell_date",
                "cost",
                "revenue",
                "ret",
                "sub_type",
                "yjyg_type",
            ]
        )

    by_stock = {}
    for trade in all_trades:
        by_stock.setdefault(trade.get("stock"), []).append(trade)

    rows = []
    for stock, trades in by_stock.items():
        open_buy = None
        for trade in trades:
            if trade.get("type") == "BUY":
                open_buy = trade
            elif trade.get("type") == "SELL" and open_buy is not None:
                cost = open_buy.get("cost", np.nan)
                revenue = trade.get("revenue", np.nan)
                rows.append(
                    {
                        "stock": stock,
                        "buy_date": open_buy.get("date"),
                        "sell_date": trade.get("date"),
                        "cost": cost,
                        "revenue": revenue,
                        "ret": (revenue - cost) / cost if cost else np.nan,
                        "sub_type": trade.get("sub_type"),
                    }
                )
                open_buy = None
        # 末尾未平仓的 BUY 直接丢弃：没有卖出价，收益无从计算

    df = pd.DataFrame(rows)
    if df.empty:
        df["yjyg_type"] = pd.Series(dtype="object")
        return df

    df = df.sort_values(["stock", "buy_date"], kind="mergesort").reset_index(drop=True)
    df["yjyg_type"] = _lookup_event_type(df, events)
    return df


def _lookup_event_type(trades_df: pd.DataFrame, events: Optional[pd.DataFrame]):
    """按 (股票, 买入日) 回填最近一次公告的预告类型。"""
    if events is None or len(events) == 0:
        return pd.Series([None] * len(trades_df), dtype="object")

    result = np.full(len(trades_df), None, dtype=object)
    by_code = {code: grp for code, grp in events.groupby("code")}

    for pos, (stock, buy_date) in enumerate(
        zip(trades_df["stock"], trades_df["buy_date"])
    ):
        grp = by_code.get(stock)
        if grp is None:
            continue
        grp = grp.sort_values("notice_date", kind="mergesort")
        notices = grp["notice_date"].to_numpy()
        # 公告必须早于买入日（严格小于——同日公告要次日才能成交）
        slot = np.searchsorted(notices, str(buy_date), side="left") - 1
        if slot >= 0:
            result[pos] = grp["type"].to_numpy(dtype=object)[slot]

    return pd.Series(result, dtype="object")


def event_summary_stats(events_df: pd.DataFrame) -> dict:
    """事件级汇总统计。

    收益分布严重右偏（预增的变动幅度实测最高到 +7708%），
    因此中位数与均值必须同时看。

    Args:
        events_df: ``summarize_events`` 的输出

    Returns:
        dict: 整体指标 + 分年度 / 分预告类型的明细表
    """
    if events_df is None or events_df.empty:
        return {"n_events": 0}

    returns = pd.to_numeric(events_df["ret"], errors="coerce").dropna()
    if returns.empty:
        return {"n_events": 0}

    n = len(returns)
    std = returns.std(ddof=1)
    t_stat = returns.mean() / (std / np.sqrt(n)) if std > 0 and n > 1 else np.nan

    # 注意口径：引擎在价格跌破成本 5% 时无条件打上 STOP_LOSS 标记
    # （stock_backtest.py:87-93），**即便策略当根 bar 本来就返回了 -1**。
    # 因此该比例是「离场时恰好浮亏超 5%」的占比，会高于止损真正改变了行为的次数。
    # 要看止损的真实影响，请对比 enable_stop=False / True 两次回测的收益。
    stop_loss_share = (
        (events_df["sub_type"] == "STOP_LOSS").mean()
        if "sub_type" in events_df.columns
        else np.nan
    )

    return {
        "n_events": n,
        "mean_return": returns.mean(),
        "median_return": returns.median(),
        "std_return": std,
        "hit_rate": (returns > 0).mean(),
        "t_stat": t_stat,
        "best": returns.max(),
        "worst": returns.min(),
        "stop_loss_share": stop_loss_share,
        "by_year": _group_stats(events_df, events_df["buy_date"].astype(str).str[:4]),
        "by_type": _group_stats(events_df, events_df["yjyg_type"]),
    }


def _group_stats(events_df: pd.DataFrame, keys) -> pd.DataFrame:
    """按给定键分组统计事件收益。"""
    df = events_df.assign(_key=keys)
    df = df[df["_key"].notna()]
    if df.empty:
        return pd.DataFrame(columns=["n", "mean", "median", "hit_rate"])

    grouped = df.groupby("_key")["ret"]
    out = pd.DataFrame(
        {
            "n": grouped.size(),
            "mean": grouped.mean(),
            "median": grouped.median(),
            "hit_rate": grouped.apply(lambda s: (s > 0).mean()),
        }
    )
    out.index.name = None
    return out


def print_event_report(stats: dict) -> None:
    """打印事件级统计报告。"""
    print("=" * 70)
    print("事件级统计（分析单位 = 每次事件，而非每只股票）")
    print("=" * 70)

    if stats.get("n_events", 0) == 0:
        print("没有完成的事件（无配对成交）")
        return

    print(f"事件数:       {stats['n_events']}")
    print(f"平均收益:     {stats['mean_return']:>8.2%}")
    print(f"中位数收益:   {stats['median_return']:>8.2%}   <- 分布右偏，以此为准")
    print(f"胜率:         {stats['hit_rate']:>8.2%}")
    print(f"t 统计量:     {stats['t_stat']:>8.2f}")
    print(f"最好 / 最差:  {stats['best']:>8.2%} / {stats['worst']:.2%}")
    if not pd.isna(stats.get("stop_loss_share", np.nan)):
        print(f"止损离场占比: {stats['stop_loss_share']:>8.2%}")

    for label, key in (("分年度", "by_year"), ("分预告类型", "by_type")):
        table = stats.get(key)
        if table is not None and not table.empty:
            print(f"\n{label}:")
            print(table.to_string())
