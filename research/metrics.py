# -*- coding: utf-8 -*-
"""共享绩效指标：年化、夏普、回撤、换手、IC 及 Newey-West t 值。

三项复现（ffscore / cscv / allocation）共用本模块，``sharpe_ratio`` 也是
CSCV 默认注入的评价指标。

与 ``backtest/stock_backtest.py::StockBacktest.calculate_metrics`` 的公式
存在重复，这是有意为之：旧实现与单标的全仓进出的内部状态耦合，抽取它等于
重写一份可用的归档代码。参见 ``research/README.md``。

约定
----
- 所有 ``periods`` 参数指每年的期数（日频 252、月频 12）。
- ``rf`` 一律是**年化**无风险利率，内部按 ``(1+rf)**(1/periods)-1`` 折成单期。
- ``max_drawdown`` 与 ``drawdown_series`` 都返回**正的**回撤比例，
  满足 ``max_drawdown(nav) == drawdown_series(nav).max()``。
"""

from typing import Dict, Optional, Union

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _as_series(x: Union[pd.Series, np.ndarray, list]) -> pd.Series:
    """把输入统一成去掉 NaN 的 float Series。"""
    if not isinstance(x, pd.Series):
        x = pd.Series(x)
    return x.astype(float).dropna()


def nav_from_returns(returns, initial: float = 1.0) -> pd.Series:
    """由单期收益率序列累乘出净值曲线。

    Args:
        returns: 单期收益率序列（0.01 表示 1%）
        initial: 期初净值

    Returns:
        pd.Series: 净值曲线，长度与 returns 相同
    """
    r = _as_series(returns)
    return initial * (1.0 + r).cumprod()


def annualized_return(returns, periods: int = TRADING_DAYS) -> float:
    """几何年化收益率。

    注意：本函数对**亏损**序列同样返回有限值。仓库现有 notebook 中的
    ``(cum[-1] - 1) ** (250 / T) - 1`` 是错的 —— 它把**净**收益当底数，
    负数的分数次幂得 NaN，会让所有亏损策略静默变成 NaN。正确的底数是
    净值本身 ``cum[-1]``。

    Args:
        returns: 单期收益率序列
        periods: 每年期数

    Returns:
        float: 年化收益率；本金亏光（末期净值 <= 0）时返回 -1.0
    """
    r = _as_series(returns)
    if len(r) == 0:
        return 0.0

    final_nav = float((1.0 + r).prod())
    if final_nav <= 0:
        return -1.0

    return final_nav ** (periods / len(r)) - 1.0


def annualized_vol(returns, periods: int = TRADING_DAYS) -> float:
    """年化波动率（单期收益标准差 × sqrt(periods)，ddof=1）。"""
    r = _as_series(returns)
    if len(r) < 2:
        return 0.0
    return float(r.std(ddof=1) * np.sqrt(periods))


def _periodic_rf(rf: float, periods: int) -> float:
    """把年化无风险利率折算成单期利率。"""
    return (1.0 + rf) ** (1.0 / periods) - 1.0


def sharpe_ratio(returns, rf: float = 0.0, periods: int = TRADING_DAYS) -> float:
    """夏普比率（算术均值口径）。

    Args:
        returns: 单期收益率序列
        rf: 年化无风险利率
        periods: 每年期数

    Returns:
        float: 年化夏普；样本不足或零波动时返回 0.0
    """
    r = _as_series(returns)
    if len(r) < 2:
        return 0.0

    excess = r - _periodic_rf(rf, periods)
    sigma = excess.std(ddof=1)
    if sigma == 0 or not np.isfinite(sigma):
        return 0.0

    return float(excess.mean() / sigma * np.sqrt(periods))


def sortino_ratio(returns, rf: float = 0.0, periods: int = TRADING_DAYS) -> float:
    """索提诺比率：只用下行波动（低于无风险利率的部分）作分母。"""
    r = _as_series(returns)
    if len(r) < 2:
        return 0.0

    excess = r - _periodic_rf(rf, periods)
    downside = excess[excess < 0]
    if len(downside) == 0:
        return np.inf if excess.mean() > 0 else 0.0

    # 下行标准差以全样本长度为分母（Sortino 原始定义），不是只数下行样本
    dd = np.sqrt((downside**2).sum() / len(excess))
    if dd == 0:
        return 0.0

    return float(excess.mean() / dd * np.sqrt(periods))


def drawdown_series(nav) -> pd.Series:
    """回撤序列（正比例，0 表示处于历史高点）。"""
    v = _as_series(nav)
    if len(v) == 0:
        return pd.Series(dtype=float)

    peak = v.cummax()
    return (peak - v) / peak


def max_drawdown(nav) -> float:
    """最大回撤（正比例）。满足 ``max_drawdown(nav) == drawdown_series(nav).max()``。"""
    dd = drawdown_series(nav)
    return float(dd.max()) if len(dd) else 0.0


def calmar_ratio(returns, periods: int = TRADING_DAYS) -> float:
    """卡玛比率 = 年化收益 / 最大回撤。零回撤时返回 inf（收益为正）或 0。"""
    ann = annualized_return(returns, periods)
    mdd = max_drawdown(nav_from_returns(returns))
    if mdd == 0:
        return np.inf if ann > 0 else 0.0
    return float(ann / mdd)


def turnover(weights: pd.DataFrame) -> pd.Series:
    """单边换手率序列。

    ``0.5 * Σ|w_t - w_{t-1}|``，首期视上期权重为 0（建仓）。这是
    **目标权重到目标权重**的口径，不含两次调仓之间的价格漂移；截面回测
    引擎另有含漂移的精确换手，二者不要混用。

    Args:
        weights: 目标权重，index 为调仓日，columns 为标的

    Returns:
        pd.Series: 每个调仓日的单边换手率
    """
    w = weights.fillna(0.0)
    prev = w.shift(1).fillna(0.0)
    return 0.5 * (w - prev).abs().sum(axis=1)


def information_coefficient(factor, forward_returns, method: str = "spearman") -> float:
    """单期截面 IC。

    Args:
        factor: 截面因子值
        forward_returns: 对应的未来收益
        method: 'spearman'（RankIC，默认）或 'pearson'

    Returns:
        float: 相关系数；有效样本 < 3 时返回 NaN
    """
    f = pd.Series(factor).astype(float)
    r = pd.Series(forward_returns).astype(float)

    df = pd.concat([f, r], axis=1, keys=["f", "r"]).dropna()
    if len(df) < 3:
        return np.nan
    if df["f"].nunique() < 2 or df["r"].nunique() < 2:
        return np.nan

    return float(df["f"].corr(df["r"], method=method))


def newey_west_t(x, lags: Optional[int] = None) -> float:
    """检验序列均值是否显著为 0 的 Newey-West t 值（自相关稳健）。

    IC 序列通常存在自相关，普通 t 值会高估显著性。statsmodels 未列入依赖，
    这里直接实现 Bartlett 核的 HAC 标准误。

    Args:
        x: 待检验序列（如逐期 IC）
        lags: 截断滞后阶数，默认用经验规则 ``int(4 * (n/100) ** (2/9))``

    Returns:
        float: t 值；样本不足或零方差时返回 NaN
    """
    v = _as_series(x)
    n = len(v)
    if n < 3:
        return np.nan

    if lags is None:
        lags = int(4 * (n / 100.0) ** (2.0 / 9.0))
    lags = max(0, min(lags, n - 1))

    resid = (v - v.mean()).to_numpy()
    variance = float((resid**2).sum() / n)

    for j in range(1, lags + 1):
        gamma = float((resid[j:] * resid[:-j]).sum() / n)
        variance += 2.0 * (1.0 - j / (lags + 1.0)) * gamma

    if variance <= 0:
        return np.nan

    return float(v.mean() / np.sqrt(variance / n))


def ic_summary(ic_series, lags: Optional[int] = None) -> Dict[str, float]:
    """IC 序列的汇总统计。

    Returns:
        dict: ic_mean / ic_std / ic_ir / ic_t / positive_ratio / n_periods
    """
    ic = _as_series(ic_series)
    if len(ic) == 0:
        return {
            "ic_mean": np.nan,
            "ic_std": np.nan,
            "ic_ir": np.nan,
            "ic_t": np.nan,
            "positive_ratio": np.nan,
            "n_periods": 0,
        }

    std = ic.std(ddof=1) if len(ic) > 1 else np.nan
    return {
        "ic_mean": float(ic.mean()),
        "ic_std": float(std) if pd.notna(std) else np.nan,
        "ic_ir": float(ic.mean() / std) if pd.notna(std) and std != 0 else np.nan,
        "ic_t": newey_west_t(ic, lags=lags),
        "positive_ratio": float((ic > 0).mean()),
        "n_periods": int(len(ic)),
    }


def summarize_returns(
    returns, rf: float = 0.0, periods: int = TRADING_DAYS
) -> Dict[str, float]:
    """一次性算出常用绩效指标。

    Returns:
        dict: total_return / annual_return / annual_vol / sharpe / sortino /
              max_drawdown / calmar / n_periods
    """
    r = _as_series(returns)
    nav = nav_from_returns(r)

    return {
        "total_return": float(nav.iloc[-1] - 1.0) if len(nav) else 0.0,
        "annual_return": annualized_return(r, periods),
        "annual_vol": annualized_vol(r, periods),
        "sharpe": sharpe_ratio(r, rf=rf, periods=periods),
        "sortino": sortino_ratio(r, rf=rf, periods=periods),
        "max_drawdown": max_drawdown(nav),
        "calmar": calmar_ratio(r, periods),
        "n_periods": int(len(r)),
    }
