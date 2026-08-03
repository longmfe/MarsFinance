# -*- coding: utf-8 -*-
"""截面分组回测引擎。

仓库既有的 ``backtest/stock_backtest.py`` 做不了这件事：它是单标的、全仓
进出、没有换手也没有 IC。本模块是全新实现，专门服务于截面因子研究。

组合会计口径
------------
组内**等权买入、持有到下次调仓**。这意味着权重在持有期内随价格漂移，
而等权买入持有组合的净值恰好等于各成分累计净值的算术平均：

    NAV(t) = Σ_i w0_i · Π(1 + r_i)

因此不需要逐日循环调权重，一次 ``cumprod`` 就能得到精确结果。

停牌与退市
----------
停牌日没有行情 → 当日收益记 0（价格顺延），复牌后继续。彻底退市的标的其
累计净值就此冻结在最后一个价格上 —— 这是免费数据源下能做到的最诚实处理，
不假装能在退市当天以某个价格清仓。

分组的陷阱
----------
F-Score 是 0-9 的整数，并列极多。用 ``rank(pct=True)`` 分十档会把档位边界
切进并列块内部，同分股票被任意劈开，分组结果不可复现也无经济含义。默认的
``group_by="score_value"`` 直接按分值分桶，与 Piotroski 一致。
"""

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from research.metrics import (
    ic_summary,
    information_coefficient,
    max_drawdown,
    summarize_returns,
)


class CrossSectionalBacktest:
    """按因子分组的截面回测。

    Args:
        group_by: ``"score_value"`` 按因子取值分桶（整数因子用这个），
            或 ``"quantile"`` 按分位数分组
        n_groups: ``quantile`` 模式下的分组数
        weighting: ``"equal"`` 等权 或 ``"cap"`` 流通市值加权
        commission: 单边佣金
        slippage: 单边滑点
        stamp_tax: 印花税（仅卖出）。仓库既有引擎漏了这项
        execution_lag: 信号日到成交日的交易日间隔，1 表示次日成交
    """

    def __init__(
        self,
        group_by: str = "score_value",
        n_groups: int = 10,
        weighting: str = "equal",
        commission: float = 0.001,
        slippage: float = 0.001,
        stamp_tax: float = 0.001,
        execution_lag: int = 1,
    ):
        if group_by not in ("score_value", "quantile"):
            raise ValueError(
                f"group_by 只能是 score_value / quantile，得到 {group_by!r}"
            )
        if weighting not in ("equal", "cap"):
            raise ValueError(f"weighting 只能是 equal / cap，得到 {weighting!r}")

        self.group_by = group_by
        self.n_groups = n_groups
        self.weighting = weighting
        self.commission = commission
        self.slippage = slippage
        self.stamp_tax = stamp_tax
        self.execution_lag = execution_lag

    # --- 分组 -----------------------------------------------------------

    def assign_groups(self, factor: pd.Series) -> pd.Series:
        """逐截面把因子值映射到分组标签。"""
        values = factor.dropna()
        if values.empty:
            return pd.Series(dtype=object)

        if self.group_by == "score_value":
            return values.round().astype(int).astype(str)

        def bucket(cross_section):
            if len(cross_section) < self.n_groups:
                return pd.Series("G0", index=cross_section.index)
            ranks = cross_section.rank(ascending=True, method="first", pct=True)
            labels = np.minimum(
                (ranks * self.n_groups).apply(np.ceil).astype(int), self.n_groups
            )
            return labels.map(lambda i: f"G{i}")

        return values.groupby(level="date", group_keys=False).apply(bucket)

    # --- 权重 -----------------------------------------------------------

    def _target_weights(
        self, members: Sequence[str], date, price_panel: pd.DataFrame
    ) -> pd.Series:
        """组内目标权重（和为 1）。"""
        if self.weighting == "equal":
            return pd.Series(1.0 / len(members), index=list(members))

        caps = []
        for code in members:
            try:
                row = price_panel.loc[(date, code)]
                caps.append(float(row["close"]) * float(row["outstanding_share"]))
            except (KeyError, TypeError, ValueError):
                caps.append(np.nan)

        weights = pd.Series(caps, index=list(members))
        if weights.isna().all() or weights.sum() <= 0:
            return pd.Series(1.0 / len(members), index=list(members))

        weights = weights.fillna(weights.median())
        return weights / weights.sum()

    # --- 主流程 ---------------------------------------------------------

    def run(
        self,
        factor: pd.Series,
        price_panel: pd.DataFrame,
        universe_mask: Optional[pd.Series] = None,
    ) -> Dict:
        """执行分组回测。

        Args:
            factor: MultiIndex (date, code) 的因子值，index 的 date 即调仓日
            price_panel: MultiIndex (date, code) 行情，需含 close
            universe_mask: 可选布尔掩码，False 的样本不参与

        Returns:
            dict: group_nav / group_returns / long_short_nav / turnover /
                metrics_by_group / ic / ic_stats / n_holdings / rebalance_dates
        """
        if universe_mask is not None:
            factor = factor.where(universe_mask.reindex(factor.index).fillna(False))

        factor = factor.dropna()
        if factor.empty:
            raise ValueError("因子在筛选后为空，无法回测")

        close = price_panel["close"].unstack("code").sort_index()

        # 停牌处理必须显式：先把价格前向填充（停牌期间价格顺延），再算收益。
        # 这样停牌日收益为 0，而复牌当日的跳空会被完整计入 —— 那是真实收益。
        # 不能依赖 pct_change 的默认 fill_method（已弃用，且语义会变）。
        returns = close.ffill().pct_change(fill_method=None)
        trading_days = close.index

        groups = self.assign_groups(factor)
        rebalance_dates = sorted(groups.index.get_level_values("date").unique())

        # 调仓日 → 成交日（次日成交，避免用当日收盘价成交的隐性前视）
        execution_dates = []
        for date in rebalance_dates:
            position = trading_days.searchsorted(pd.Timestamp(date), side="left")
            position += self.execution_lag
            execution_dates.append(
                trading_days[position] if position < len(trading_days) else None
            )

        labels = sorted(set(groups.values))
        nav_by_group = {label: [] for label in labels}
        turnover_records = []
        holdings_count = []
        previous_weights: Dict[str, pd.Series] = {}

        for i, (rebalance_date, start) in enumerate(
            zip(rebalance_dates, execution_dates)
        ):
            if start is None:
                continue

            end = (
                execution_dates[i + 1]
                if i + 1 < len(execution_dates) and execution_dates[i + 1] is not None
                else trading_days[-1]
            )
            if end <= start:
                continue

            window = trading_days[(trading_days > start) & (trading_days <= end)]
            if len(window) == 0:
                continue

            cross_section = groups.xs(rebalance_date, level="date")
            holdings_count.append(
                {"date": rebalance_date, "n_total": len(cross_section)}
            )

            for label in labels:
                members = [
                    code
                    for code in cross_section[cross_section == label].index
                    if code in close.columns
                ]
                if not members:
                    nav_by_group[label].append(pd.Series(0.0, index=window, name=label))
                    continue

                weights = self._target_weights(members, start, price_panel)

                # 换手成本：卖出腿含印花税，买入腿不含
                previous = previous_weights.get(label)
                if previous is None:
                    one_way = 1.0
                else:
                    aligned = weights.reindex(
                        previous.index.union(weights.index)
                    ).fillna(0.0)
                    old = previous.reindex(aligned.index).fillna(0.0)
                    one_way = 0.5 * float((aligned - old).abs().sum())

                cost = one_way * (
                    2.0 * (self.commission + self.slippage) + self.stamp_tax
                )
                turnover_records.append(
                    {
                        "date": rebalance_date,
                        "group": label,
                        "turnover": one_way,
                        "cost": cost,
                    }
                )

                sub = returns.loc[window, members].fillna(0.0)
                cumulative = (1.0 + sub).cumprod()
                nav = (cumulative * weights.reindex(members).values).sum(axis=1)

                period_returns = nav.pct_change()
                period_returns.iloc[0] = nav.iloc[0] - 1.0
                period_returns.iloc[0] -= cost  # 调仓成本一次性计入首日

                nav_by_group[label].append(period_returns.rename(label))

                # 期末漂移后的权重，供下次调仓算换手
                drifted = cumulative.iloc[-1] * weights.reindex(members).values
                previous_weights[label] = drifted / drifted.sum()

        group_returns = pd.DataFrame(
            {
                label: pd.concat(parts).sort_index()
                for label, parts in nav_by_group.items()
                if parts
            }
        ).fillna(0.0)

        group_nav = (1.0 + group_returns).cumprod()

        metrics_by_group = {
            label: summarize_returns(group_returns[label])
            for label in group_returns.columns
        }

        long_short = self._long_short(group_returns, labels)
        ic_series = self._information_coefficient(factor, close, rebalance_dates)

        return {
            "group_returns": group_returns,
            "group_nav": group_nav,
            "long_short_returns": long_short,
            "long_short_nav": (1.0 + long_short).cumprod() if len(long_short) else None,
            "metrics_by_group": metrics_by_group,
            "long_short_metrics": (
                summarize_returns(long_short) if len(long_short) else {}
            ),
            "turnover": pd.DataFrame(turnover_records),
            "ic": ic_series,
            "ic_stats": ic_summary(ic_series),
            "n_holdings": pd.DataFrame(holdings_count),
            "rebalance_dates": rebalance_dates,
            "labels": list(group_returns.columns),
        }

    # --- 辅助 -----------------------------------------------------------

    @staticmethod
    def _sorted_labels(labels: Sequence[str]) -> list:
        """把分组标签按数值排序（"2" < "10"，"G2" < "G10"）。"""

        def key(label):
            digits = "".join(ch for ch in str(label) if ch.isdigit())
            return int(digits) if digits else 0

        return sorted(labels, key=key)

    def _long_short(self, group_returns: pd.DataFrame, labels) -> pd.Series:
        """最高分组减最低分组。"""
        available = self._sorted_labels(group_returns.columns)
        if len(available) < 2:
            return pd.Series(dtype=float)
        return group_returns[available[-1]] - group_returns[available[0]]

    @staticmethod
    def _information_coefficient(factor, close, rebalance_dates) -> pd.Series:
        """逐调仓期的 RankIC：因子值 vs 到下次调仓的收益。"""
        records = {}

        for i in range(len(rebalance_dates) - 1):
            start, end = rebalance_dates[i], rebalance_dates[i + 1]
            cross_section = factor.xs(start, level="date")

            codes = [c for c in cross_section.index if c in close.columns]
            if len(codes) < 3:
                continue

            window = close.loc[(close.index >= start) & (close.index <= end), codes]
            if len(window) < 2:
                continue

            forward = window.iloc[-1] / window.iloc[0] - 1.0
            records[start] = information_coefficient(
                cross_section.reindex(codes), forward, method="spearman"
            )

        return pd.Series(records, dtype=float).dropna()


def signal_efficacy(
    signals: pd.DataFrame,
    price_panel: pd.DataFrame,
    signal_names: Sequence[str],
) -> pd.DataFrame:
    """九个信号各自的单独有效性 —— 复现研报里的"单信号有效性"表。

    对每个信号，比较"信号=1"与"信号=0"两组在下一调仓期的平均收益。

    Args:
        signals: MultiIndex (date, code)，含各信号列（0/1/NaN）
        price_panel: 行情面板
        signal_names: 要评估的信号列名

    Returns:
        pd.DataFrame: 每行一个信号，含 mean_ret_1 / mean_ret_0 / spread /
            t_stat / n_periods
    """
    from scipy import stats

    close = price_panel["close"].unstack("code").sort_index()
    dates = sorted(signals.index.get_level_values("date").unique())

    rows = []
    for name in signal_names:
        spreads = []

        for i in range(len(dates) - 1):
            start, end = dates[i], dates[i + 1]
            cross_section = signals.xs(start, level="date")[name].dropna()

            codes = [c for c in cross_section.index if c in close.columns]
            if len(codes) < 4:
                continue

            window = close.loc[(close.index >= start) & (close.index <= end), codes]
            if len(window) < 2:
                continue

            forward = window.iloc[-1] / window.iloc[0] - 1.0
            flags = cross_section.reindex(codes)

            winners = forward[flags == 1].dropna()
            losers = forward[flags == 0].dropna()
            if len(winners) < 2 or len(losers) < 2:
                continue

            spreads.append(winners.mean() - losers.mean())

        if not spreads:
            rows.append({"signal": name, "n_periods": 0})
            continue

        series = pd.Series(spreads)
        t_stat, _ = (
            stats.ttest_1samp(series, 0.0) if len(series) > 1 else (np.nan, None)
        )

        rows.append(
            {
                "signal": name,
                "mean_spread": float(series.mean()),
                "t_stat": float(t_stat),
                "positive_ratio": float((series > 0).mean()),
                "n_periods": int(len(series)),
            }
        )

    return pd.DataFrame(rows).set_index("signal")
