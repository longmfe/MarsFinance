# -*- coding: utf-8 -*-
"""Piotroski F-Score 的九个二元信号，每个都是可独立测试的纯函数。

统一契约
--------
入参是**报告期空间**的数据帧：同时带当期列与 ``*_prev`` / ``*_prev2`` 滞后列
（由 ``datafeed.fundamentals.add_lagged`` 生成）。返回 ``{0.0, 1.0}`` 的
``pd.Series``，未定义处为 ``NaN`` —— **绝不是 inf，也绝不静默当成 0**。
把"数据缺失"和"信号为 0"混为一谈会系统性地压低分数。

边界一律用严格不等号：恰好为零判 0。

三个代理口径
------------
免费数据源拿不到 Piotroski 原文的全部字段，以下三项是**明确标注的代理**，
不是等价实现：

- **F5** 原文用长期负债/均值总资产。批量接口只有总负债与资产负债率，
  默认退化为 Δ资产负债率；``lever_definition="noncurrent"`` 用非流动负债，
  需要逐股明细。
- **F7** 原文看是否增发普通股。这里用流通股本在相邻财年末的变化，
  **无法区分增发与送转股**，故留有 ``tolerance``。
- **F8** 原文用毛利率。这里用 ``(营业总收入 - 营业支出)/营业总收入``，
  东方财富该子项的口径尚未逐笔核验（见 research/README.md 的偏离清单）。
"""

from typing import Callable, Dict, Sequence, Tuple

import numpy as np
import pandas as pd

SIGNAL_NAMES: Tuple[str, ...] = (
    "f1_roa_positive",
    "f2_cfo_positive",
    "f3_delta_roa",
    "f4_accrual",
    "f5_delta_lever",
    "f6_delta_liquid",
    "f7_no_equity_offer",
    "f8_delta_margin",
    "f9_delta_turnover",
)

#: 每个信号的中文名，用于报告输出
SIGNAL_LABELS: Dict[str, str] = {
    "f1_roa_positive": "ROA 为正",
    "f2_cfo_positive": "经营现金流为正",
    "f3_delta_roa": "ROA 同比改善",
    "f4_accrual": "现金流优于净利润（低应计）",
    "f5_delta_lever": "杠杆下降",
    "f6_delta_liquid": "流动比率上升",
    "f7_no_equity_offer": "未增发股本",
    "f8_delta_margin": "毛利率上升",
    "f9_delta_turnover": "资产周转率上升",
}


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    """取列并转成 float；列不存在时返回全 NaN（信号随之为 NaN）。"""
    if name not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[name], errors="coerce")


def _divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """除法：分母为 0 或缺失时给 NaN，**绝不产生 inf**。"""
    return numerator / denominator.where(denominator != 0, np.nan)


def _binary(condition: pd.Series, *inputs: pd.Series) -> pd.Series:
    """把条件转成 {0.0, 1.0}，任一输入缺失则该行为 NaN。"""
    valid = pd.concat(list(inputs), axis=1).notna().all(axis=1)
    return condition.astype(float).where(valid, np.nan)


def _total_assets_denominator(
    df: pd.DataFrame, basis: str = "beginning", offset: int = 0
) -> pd.Series:
    """ROA / 周转率的资产分母。

    Args:
        df: 报告期数据帧
        basis: ``"beginning"`` 期初总资产（Piotroski 原文）或 ``"average"`` 均值
        offset: 0 表示当期的分母，1 表示上期的分母（供 Δ 类信号使用）
    """
    current = _col(df, "total_assets" if offset == 0 else "total_assets_prev")
    prior = _col(df, "total_assets_prev" if offset == 0 else "total_assets_prev2")

    if basis == "beginning":
        return prior
    if basis == "average":
        return (current + prior) / 2.0
    raise ValueError(f"未知的 ta_basis: {basis!r}")


# --- 盈利能力（4 项） ---------------------------------------------------


def f1_roa_positive(df: pd.DataFrame, ta_basis: str = "beginning") -> pd.Series:
    """ROA > 0。ROA = 净利润 / 期初总资产。"""
    roa = _divide(_col(df, "net_profit"), _total_assets_denominator(df, ta_basis, 0))
    return _binary(roa > 0, roa)


def f2_cfo_positive(df: pd.DataFrame, ta_basis: str = "beginning") -> pd.Series:
    """经营性现金流 > 0（按期初总资产标准化，符号与原始值一致）。"""
    cfo = _divide(_col(df, "cfo"), _total_assets_denominator(df, ta_basis, 0))
    return _binary(cfo > 0, cfo)


def f3_delta_roa(df: pd.DataFrame, ta_basis: str = "beginning") -> pd.Series:
    """ROA 同比改善。

    注意上期 ROA 用的是**上期的**期初总资产（即 ``total_assets_prev2``），
    这正是需要 lag 2 的原因。
    """
    roa = _divide(_col(df, "net_profit"), _total_assets_denominator(df, ta_basis, 0))
    roa_prev = _divide(
        _col(df, "net_profit_prev"), _total_assets_denominator(df, ta_basis, 1)
    )
    return _binary(roa - roa_prev > 0, roa, roa_prev)


def f4_accrual(df: pd.DataFrame, ta_basis: str = "beginning") -> pd.Series:
    """经营现金流优于净利润（应计项为负），即盈利有现金支撑。"""
    denominator = _total_assets_denominator(df, ta_basis, 0)
    cfo = _divide(_col(df, "cfo"), denominator)
    roa = _divide(_col(df, "net_profit"), denominator)
    return _binary(cfo > roa, cfo, roa)


# --- 杠杆 / 流动性 / 融资（3 项） ----------------------------------------


def f5_delta_lever(
    df: pd.DataFrame, definition: str = "debt_ratio", ta_basis: str = "beginning"
) -> pd.Series:
    """杠杆**下降**。

    Args:
        definition: ``"debt_ratio"`` 用 Δ资产负债率（代理口径，批量数据即可）；
            ``"noncurrent"`` 用 Δ(非流动负债/总资产)，更接近原文的长期负债口径，
            需要逐股明细提供 ``noncurrent_liab``
    """
    if definition == "debt_ratio":
        lever = _divide(_col(df, "total_liab"), _col(df, "total_assets"))
        lever_prev = _divide(_col(df, "total_liab_prev"), _col(df, "total_assets_prev"))
    elif definition == "noncurrent":
        lever = _divide(
            _col(df, "noncurrent_liab"), _total_assets_denominator(df, ta_basis, 0)
        )
        lever_prev = _divide(
            _col(df, "noncurrent_liab_prev"),
            _total_assets_denominator(df, ta_basis, 1),
        )
    else:
        raise ValueError(f"未知的 lever_definition: {definition!r}")

    return _binary(lever - lever_prev < 0, lever, lever_prev)


def f6_delta_liquid(df: pd.DataFrame) -> pd.Series:
    """流动比率**上升**。流动比率 = 流动资产 / 流动负债。

    只有逐股明细接口提供流动资产/负债的拆分。银行等金融机构的资产负债表没有
    流动/非流动分类，这两列为空 —— 信号随之为 NaN，与"排除金融股"一致。
    """
    ratio = _divide(_col(df, "current_assets"), _col(df, "current_liab"))
    ratio_prev = _divide(_col(df, "current_assets_prev"), _col(df, "current_liab_prev"))
    return _binary(ratio - ratio_prev > 0, ratio, ratio_prev)


def f7_no_equity_offer(df: pd.DataFrame, tolerance: float = 0.0) -> pd.Series:
    """未增发股本（股本未增加）。

    用相邻财年末的流通股本比较。**无法区分增发与送转股** —— 送转股同样会让
    股本增加，却不是"向市场要钱"。``tolerance`` 用来消化小幅变动。
    """
    shares = _col(df, "shares")
    shares_prev = _col(df, "shares_prev")
    return _binary(shares <= shares_prev * (1.0 + tolerance), shares, shares_prev)


# --- 运营效率（2 项） ---------------------------------------------------


def f8_delta_margin(df: pd.DataFrame) -> pd.Series:
    """毛利率**上升**。毛利率 = (营业总收入 - 营业支出) / 营业总收入。"""
    revenue = _col(df, "revenue")
    revenue_prev = _col(df, "revenue_prev")

    margin = _divide(revenue - _col(df, "operating_cost"), revenue)
    margin_prev = _divide(revenue_prev - _col(df, "operating_cost_prev"), revenue_prev)
    return _binary(margin - margin_prev > 0, margin, margin_prev)


def f9_delta_turnover(df: pd.DataFrame, ta_basis: str = "beginning") -> pd.Series:
    """资产周转率**上升**。周转率 = 营业总收入 / 期初总资产。"""
    turnover = _divide(_col(df, "revenue"), _total_assets_denominator(df, ta_basis, 0))
    turnover_prev = _divide(
        _col(df, "revenue_prev"), _total_assets_denominator(df, ta_basis, 1)
    )
    return _binary(turnover - turnover_prev > 0, turnover, turnover_prev)


#: 信号名 → 函数
SIGNAL_FUNCTIONS: Dict[str, Callable[..., pd.Series]] = {
    "f1_roa_positive": f1_roa_positive,
    "f2_cfo_positive": f2_cfo_positive,
    "f3_delta_roa": f3_delta_roa,
    "f4_accrual": f4_accrual,
    "f5_delta_lever": f5_delta_lever,
    "f6_delta_liquid": f6_delta_liquid,
    "f7_no_equity_offer": f7_no_equity_offer,
    "f8_delta_margin": f8_delta_margin,
    "f9_delta_turnover": f9_delta_turnover,
}

#: 每个信号需要的输入列（用于校验与可读报错）
REQUIRED_COLUMNS: Dict[str, Tuple[str, ...]] = {
    "f1_roa_positive": ("net_profit", "total_assets_prev"),
    "f2_cfo_positive": ("cfo", "total_assets_prev"),
    "f3_delta_roa": (
        "net_profit",
        "total_assets_prev",
        "net_profit_prev",
        "total_assets_prev2",
    ),
    "f4_accrual": ("cfo", "net_profit", "total_assets_prev"),
    "f5_delta_lever": (
        "total_liab",
        "total_assets",
        "total_liab_prev",
        "total_assets_prev",
    ),
    "f6_delta_liquid": (
        "current_assets",
        "current_liab",
        "current_assets_prev",
        "current_liab_prev",
    ),
    "f7_no_equity_offer": ("shares", "shares_prev"),
    "f8_delta_margin": (
        "revenue",
        "operating_cost",
        "revenue_prev",
        "operating_cost_prev",
    ),
    "f9_delta_turnover": (
        "revenue",
        "total_assets_prev",
        "revenue_prev",
        "total_assets_prev2",
    ),
}

#: 计算九个信号所需的、要接滞后的基础列
LAG_COLUMNS: Tuple[str, ...] = (
    "total_assets",
    "total_liab",
    "net_profit",
    "revenue",
    "operating_cost",
    "cfo",
    "current_assets",
    "current_liab",
    "noncurrent_liab",
    "shares",
)


def missing_columns(df: pd.DataFrame, signal: str) -> Sequence[str]:
    """列出某信号缺少的输入列（空表示可算）。"""
    return [c for c in REQUIRED_COLUMNS[signal] if c not in df.columns]
