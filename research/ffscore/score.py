# -*- coding: utf-8 -*-
"""把九个信号汇总成 F-Score。"""

from typing import Optional

import numpy as np
import pandas as pd

from research.ffscore.config import FFScoreConfig, default_config
from research.ffscore.signals import SIGNAL_FUNCTIONS, SIGNAL_NAMES


def compute_signals(
    report_df: pd.DataFrame, config: Optional[FFScoreConfig] = None
) -> pd.DataFrame:
    """逐个计算九个信号。

    Args:
        report_df: 报告期空间的数据帧，含当期列与 ``*_prev`` / ``*_prev2`` 滞后列
        config: 参数配置，None 用默认值

    Returns:
        pd.DataFrame: 与入参同索引，九列 ``{0.0, 1.0, NaN}``
    """
    config = config or default_config()

    kwargs = {
        "f1_roa_positive": {"ta_basis": config.ta_basis},
        "f2_cfo_positive": {"ta_basis": config.ta_basis},
        "f3_delta_roa": {"ta_basis": config.ta_basis},
        "f4_accrual": {"ta_basis": config.ta_basis},
        "f5_delta_lever": {
            "definition": config.lever_definition,
            "ta_basis": config.ta_basis,
        },
        "f6_delta_liquid": {},
        "f7_no_equity_offer": {"tolerance": config.eq_offer_tolerance},
        "f8_delta_margin": {},
        "f9_delta_turnover": {"ta_basis": config.ta_basis},
    }

    return pd.DataFrame(
        {
            name: SIGNAL_FUNCTIONS[name](report_df, **kwargs[name])
            for name in SIGNAL_NAMES
        },
        index=report_df.index,
    )


def compute_ffscore(
    report_df: pd.DataFrame, config: Optional[FFScoreConfig] = None
) -> pd.DataFrame:
    """计算 F-Score 及其配套列。

    缺失策略：可算信号数低于 ``config.min_signals`` 的记录，``f_score`` 置 NaN
    （该股当期不参与）。同时给出 ``f_score_scaled`` 把不足九项的分数折算回
    0-9 区间，使 8 信号模式与 9 信号模式可比。

    Args:
        report_df: 报告期空间的数据帧
        config: 参数配置

    Returns:
        pd.DataFrame: 九个信号列 + ``f_score`` / ``n_available`` /
            ``f_score_scaled``，并透传 ``ann_date`` / ``available_date``
    """
    config = config or default_config()

    signals = compute_signals(report_df, config)
    out = signals.copy()

    n_available = signals.notna().sum(axis=1)
    raw_score = signals.sum(axis=1, skipna=True)

    enough = n_available >= config.min_signals

    out["n_available"] = n_available
    out["f_score"] = raw_score.where(enough, np.nan)
    out["f_score_scaled"] = (
        raw_score * len(SIGNAL_NAMES) / n_available.replace(0, np.nan)
    ).where(enough, np.nan)

    for column in ("ann_date", "available_date", "period", "name", "fiscal_year"):
        if column in report_df.columns:
            out[column] = report_df[column]

    return out
