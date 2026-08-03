# -*- coding: utf-8 -*-
"""账面市值比（BM）与 Piotroski 的价值股筛选。

Piotroski 的做法是先取 BM 最高的一档（原文用五分位），再在其中按 F-Score 排序 ——
F-Score 是用来在便宜股里区分"便宜是因为被低估"还是"便宜是因为真的差"的。
"""

from typing import Optional

import numpy as np
import pandas as pd


def book_to_market(
    pit_panel: pd.DataFrame,
    price_panel: pd.DataFrame,
    equity_column: str = "total_equity",
) -> pd.Series:
    """账面市值比 = 股东权益(PIT) / 总市值。

    **已知偏差**：新浪日线的 ``outstanding_share`` 是**流通股本**而非总股本，
    对有限售股的公司会低估市值、高估 BM。见 research/README.md 的偏离清单。

    Args:
        pit_panel: MultiIndex (date, code)，含 ``equity_column``
        price_panel: MultiIndex (date, code)，含 close / outstanding_share
        equity_column: 净资产列名

    Returns:
        pd.Series: 与 pit_panel 同索引的 BM，不可算处为 NaN
    """
    equity = pd.to_numeric(pit_panel[equity_column], errors="coerce")

    close = pd.to_numeric(
        price_panel["close"].reindex(pit_panel.index), errors="coerce"
    )
    shares = pd.to_numeric(
        price_panel["outstanding_share"].reindex(pit_panel.index), errors="coerce"
    )

    market_cap = close * shares
    market_cap = market_cap.where(market_cap > 0, np.nan)

    bm = equity / market_cap
    return bm.where(equity > 0, np.nan)


def select_high_bm(
    bm: pd.Series, quantile: float = 0.2, mask: Optional[pd.Series] = None
) -> pd.Series:
    """逐截面选出 BM 最高的 ``quantile`` 比例。

    Args:
        bm: MultiIndex (date, code) 的 BM
        quantile: 取最高的比例，0.2 表示最高五分位；1.0 表示不筛选
        mask: 可选的前置掩码（如股票池筛选），只在通过的样本内取分位

    Returns:
        pd.Series: 布尔 Series，与 bm 同索引
    """
    values = bm.copy()
    if mask is not None:
        values = values.where(mask.reindex(values.index).fillna(False), np.nan)

    if quantile >= 1.0:
        return values.notna()

    # 逐截面按 BM 降序取百分位排名，rank 越小越贵
    pct = values.groupby(level="date").rank(ascending=False, pct=True, method="first")
    return (pct <= quantile) & values.notna()
