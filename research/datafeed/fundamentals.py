# -*- coding: utf-8 -*-
"""财务数据的装载、累计还原与时点（PIT）对齐 —— 整个复现的正确性核心。

核心设计决策
------------
**先在报告期空间算完信号，再把成品 PIT 对齐到调仓日。**

先对齐、后取滞后，是前视偏差最常见的滋生地：一条 2023Q1 的截面里混进了
"去年同期"的字段，而那个字段在截面日期未必已经公开。反过来做，则一条记录的
可用时点就是它用到的所有报告的公告日期的**最大值**，只算一次，之后
``merge_asof(direction="backward")`` 可证明安全。

取显式 ``max`` 而非直接用当期公告日期是必要的：延迟披露与追溯调整确实会让
时序倒置（去年年报比今年一季报还晚公告的情况真实存在）。

公告日期的陷阱（已实测确认）
----------------------------
**批量端点的"公告日期"不是真实公告日期，而是最后更新日。** 以贵州茅台为例：

=========  ======================  ==============================
报告期     NOTICE_DATE（真实公告）  批量端点"公告日期"（= UPDATE_DATE）
=========  ======================  ==============================
FY2021     2022-03-31              2023-03-31
FY2022     2023-03-31              2024-04-03
=========  ======================  ==============================

全市场同样如此：报告期 ``20211231`` 的"公告日期"里有 4842 只落在 2023 年。
原因是下一年的年报会把本期数字作为比较期重新披露，eastmoney 据此更新了该字段。

直接拿它当可用时点**不会造成前视**（方向是保守的），但会让策略永远在使用约
两年前的财报，严重低估 Piotroski 的效果。因此：

- 批量端点的那一列映射为 ``update_date``，仅供审计，**不作为可用时点**；
- 真实公告日期取自逐股接口的 ``NOTICE_DATE``（``load_detail_fundamentals``）；
- A 股三张报表在同一份定期报告中披露，所以资产负债表的 ``NOTICE_DATE``
  同样适用于利润表与现金流量表 —— 不必再逐股拉另外两张表；
- 缺明细时兜底用法定披露截止日（年报为次年 4/30）。

幸存者偏差（已实测）
--------------------
批量端点返回的**不是**时点快照，而是当前在册公司回填的历史财报：
``stock_zcfz_em(date="20131231")`` 里有 137 只 688 科创板股票，而科创板
2019 年 7 月才开板；2013 存在而 2023 消失的公司只有 4 只。

可修的一半（``screen_universe`` 强制执行）：用行情首个交易日推出上市日期，
剔除"当时尚未上市"的公司。不可修的一半：退市公司缺失，免费数据源无解，
必须在报告中声明。
"""

from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from research.datafeed.panel import normalize_code

# --- 列名映射：东方财富中文列 → 规范英文列 -------------------------------

BALANCE_COLUMNS = {
    "股票代码": "code",
    "股票简称": "name",
    "资产-总资产": "total_assets",
    "负债-总负债": "total_liab",
    "股东权益合计": "total_equity",
    "资产-货币资金": "cash",
    "资产-应收账款": "receivables",
    "资产-存货": "inventory",
    "公告日期": "update_date_balance",
}

INCOME_COLUMNS = {
    "股票代码": "code",
    "净利润": "net_profit",
    "营业总收入": "revenue",
    "营业总支出-营业支出": "operating_cost",
    "营业利润": "operating_profit",
    "公告日期": "update_date_income",
}

CASHFLOW_COLUMNS = {
    "股票代码": "code",
    "经营性现金流-现金流量净额": "cfo",
    "公告日期": "update_date_cashflow",
}

#: 逐股资产负债表明细：真实公告日期 + 流动资产/负债（F6、F5-noncurrent 所需）
DETAIL_COLUMNS = {
    "REPORT_DATE": "period",
    "NOTICE_DATE": "ann_date",
    "TOTAL_CURRENT_ASSETS": "current_assets",
    "TOTAL_CURRENT_LIAB": "current_liab",
    "TOTAL_NONCURRENT_LIAB": "noncurrent_liab",
}

#: 流量项（累计口径，需要差分还原单季）。存量项绝不能差分。
FLOW_COLUMNS = ("net_profit", "revenue", "operating_cost", "operating_profit", "cfo")

#: 存量项（时点余额，跨期直接可比）
STOCK_COLUMNS = (
    "total_assets",
    "total_liab",
    "total_equity",
    "cash",
    "receivables",
    "inventory",
)

_ST_MARKERS = ("ST", "退")
_FINANCIAL_KEYWORDS = ("银行", "证券", "保险", "信托", "租赁")


def _rename_and_index(frame: pd.DataFrame, mapping: Dict, period: str) -> pd.DataFrame:
    """取出需要的列、重命名、补上报告期，并按 code 去重。"""
    available = {src: dst for src, dst in mapping.items() if src in frame.columns}
    out = frame[list(available)].rename(columns=available).copy()

    if "code" not in out.columns:
        raise KeyError(f"数据帧缺少股票代码列，实际列: {list(frame.columns)[:10]}")

    out["code"] = out["code"].map(_safe_normalize)
    out = out[out["code"].notna()]
    out["period"] = pd.Timestamp(period)

    return out.drop_duplicates(subset=["code"], keep="first")


def _safe_normalize(raw):
    """代码归一，失败返回 NaN（批量表里偶有非股票行）。"""
    try:
        return normalize_code(raw)
    except (ValueError, TypeError):
        return np.nan


def load_bulk_fundamentals(
    periods: Sequence[str], verbose: bool = True
) -> pd.DataFrame:
    """装载多个报告期的三张批量财报并合成一个报告期面板。

    ``ann_date`` 取三张表公告日期的**最大值** —— 一个信号需要三张表齐备，
    最晚的那张才决定真正的可用时点。

    Args:
        periods: 报告期列表 ``YYYYMMDD``
        verbose: 打印进度

    Returns:
        pd.DataFrame: MultiIndex (code, period)，含 name/total_assets/
            total_liab/total_equity/net_profit/revenue/operating_cost/cfo/
            ann_date/fiscal_year/quarter/debt_ratio
    """
    from research.datafeed.akshare_source import (
        fetch_balance_sheet,
        fetch_cashflow_statement,
        fetch_income_statement,
    )

    frames = []

    for i, period in enumerate(periods, 1):
        if verbose:
            print(f"  [{i}/{len(periods)}] 装载报告期 {period}")

        balance = _rename_and_index(
            fetch_balance_sheet(period), BALANCE_COLUMNS, period
        )
        income = _rename_and_index(
            fetch_income_statement(period), INCOME_COLUMNS, period
        )
        cashflow = _rename_and_index(
            fetch_cashflow_statement(period), CASHFLOW_COLUMNS, period
        )

        merged = balance.merge(
            income.drop(columns=["period"]), on="code", how="outer"
        ).merge(cashflow.drop(columns=["period"]), on="code", how="outer")

        merged["period"] = pd.Timestamp(period)
        frames.append(merged)

    if not frames:
        return pd.DataFrame()

    panel = pd.concat(frames, ignore_index=True)
    return finalize_report_panel(panel)


def load_detail_fundamentals(
    codes: Sequence[str], verbose: bool = True
) -> pd.DataFrame:
    """逐股拉取资产负债表明细（约 27s/只，一次性缓存）。

    两个用途，一次抓取都拿到：

    1. **真实公告日期** ``NOTICE_DATE`` —— 批量端点那一列"公告日期"其实是
       ``UPDATE_DATE``（见模块文档字符串），不能当作可用时点。
    2. **流动资产 / 流动负债** —— F6（ΔLIQUID）与 F5 的 ``noncurrent`` 口径
       需要，批量端点没有这个拆分。

    同一报告期可能有多条记录（追溯调整），取**最早**的公告日期 —— 那才是
    市场首次看到这个报告期的时点。

    Args:
        codes: 股票代码列表
        verbose: 打印进度

    Returns:
        pd.DataFrame: MultiIndex (code, period)，含 ann_date/current_assets/
            current_liab/noncurrent_liab
    """
    from research.datafeed.akshare_source import fetch_balance_sheet_detail

    frames = []

    for i, code in enumerate(codes, 1):
        if verbose and (i % 20 == 0 or i == len(codes)):
            print(f"  [{i}/{len(codes)}] 逐股明细 {code}")

        try:
            raw = fetch_balance_sheet_detail(code)
        except Exception as exc:  # noqa: BLE001 - 个别标的失败不应中断整批
            print(f"  ⚠️  {code} 明细获取失败: {exc.__class__.__name__}")
            continue

        if raw is None or raw.empty:
            continue

        available = {s: d for s, d in DETAIL_COLUMNS.items() if s in raw.columns}
        frame = raw[list(available)].rename(columns=available).copy()
        frame["code"] = _safe_normalize(code)
        frames.append(frame)

    if not frames:
        return pd.DataFrame()

    panel = pd.concat(frames, ignore_index=True)
    return finalize_detail_panel(panel)


def finalize_detail_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """整理逐股明细：解析日期、按 (code, period) 取最早公告日期的那条。"""
    out = panel.copy()

    out["period"] = pd.to_datetime(out["period"], errors="coerce")
    out["ann_date"] = pd.to_datetime(out["ann_date"], errors="coerce")
    out = out.dropna(subset=["period", "code"])

    out = (
        out.sort_values("ann_date").groupby(["code", "period"], as_index=False).first()
    )
    return out.set_index(["code", "period"]).sort_index()


def attach_announcement_dates(
    panel: pd.DataFrame,
    detail: Optional[pd.DataFrame] = None,
    fallback: str = "deadline",
) -> pd.DataFrame:
    """给报告面板接上**真实**公告日期，并带上明细里的流动资产/负债。

    这是本项目最容易被忽略、后果却最严重的一步。批量端点自带的"公告日期"是
    最后更新日（下一年年报把本期数字作为比较期重新披露的日子），比真实公告
    晚约 12 个月。直接拿它当可用时点不会造成前视（方向是保守的），但会让策略
    永远在用两年前的财报，把 Piotroski 的效果严重低估。

    Args:
        panel: (code, period) 索引的报告面板
        detail: ``load_detail_fundamentals`` 的产物，None 表示只用兜底规则
        fallback: 明细缺失时的兜底。``"deadline"`` 用法定披露截止日
            （年报为次年 4/30），``"none"`` 则留空（该记录不会进入任何截面）

    Returns:
        pd.DataFrame: 面板副本，``ann_date`` 已被真实公告日期覆盖
    """
    from research.datafeed.calendar import filing_deadline

    out = panel.copy()

    if detail is not None and len(detail):
        extra = [
            c
            for c in ("current_assets", "current_liab", "noncurrent_liab")
            if c in detail.columns
        ]
        joined = out.drop(columns=["ann_date"], errors="ignore").join(
            detail[["ann_date"] + extra], how="left"
        )
        out = joined

    if "ann_date" not in out.columns:
        out["ann_date"] = pd.NaT
    out["ann_date"] = pd.to_datetime(out["ann_date"], errors="coerce")

    if fallback == "deadline":
        periods = out.index.get_level_values("period")
        deadline = pd.Series([filing_deadline(p) for p in periods], index=out.index)
        out["ann_date"] = out["ann_date"].fillna(deadline)
    elif fallback != "none":
        raise ValueError(f"未知的兜底方式: {fallback!r}")

    # 公告日期不可能早于报告期期末
    period_end = pd.Series(out.index.get_level_values("period"), index=out.index)
    invalid = out["ann_date"].notna() & (out["ann_date"] < period_end)
    if invalid.any():
        print(f"  ⚠️  {int(invalid.sum())} 条记录的公告日期早于报告期期末，已置空")
        out.loc[invalid, "ann_date"] = pd.NaT

    return out


def finalize_report_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """统一公告日期、补派生列、设成 (code, period) 索引。

    独立成函数是为了让测试可以直接喂手工构造的帧，不必联网。
    """
    out = panel.copy()

    # 代码归一放在这里而不是只放在批量装载路径上：本函数是构造报告面板的
    # 唯一入口，索引格式必须由它保证（normalize_code 幂等，重复调用无害）。
    out["code"] = out["code"].map(_safe_normalize)
    out = out[out["code"].notna()]

    # 三张表齐备才算可用 → 各自取最晚的日期
    for prefix, target in (("ann_date_", "ann_date"), ("update_date_", "update_date")):
        cols = [c for c in out.columns if c.startswith(prefix)]
        if cols:
            for col in cols:
                out[col] = pd.to_datetime(out[col], errors="coerce")
            out[target] = out[cols].max(axis=1)
            out = out.drop(columns=cols)

    if "ann_date" not in out.columns:
        out["ann_date"] = pd.NaT
    out["ann_date"] = pd.to_datetime(out["ann_date"], errors="coerce")

    out["period"] = pd.to_datetime(out["period"])
    out["fiscal_year"] = out["period"].dt.year
    out["quarter"] = out["period"].dt.quarter

    # 自己算资产负债率，不依赖接口给的那一列（其单位是百分比还是比例并不确定）
    if {"total_liab", "total_assets"}.issubset(out.columns):
        out["debt_ratio"] = _safe_divide(out["total_liab"], out["total_assets"])

    return out.set_index(["code", "period"]).sort_index()


def _safe_divide(numerator, denominator) -> pd.Series:
    """除法：分母为 0 或缺失时给 NaN，**绝不产生 inf**。"""
    num = pd.to_numeric(numerator, errors="coerce")
    den = pd.to_numeric(denominator, errors="coerce")

    den = den.where(den != 0, np.nan)
    return num / den


def add_lagged(
    panel: pd.DataFrame,
    columns: Iterable[str],
    lags: Sequence[int] = (1,),
) -> pd.DataFrame:
    """按"同季度、前 N 个财年"接上滞后列。

    用 ``(code, quarter, fiscal_year - lag)`` 显式 merge，而不是
    ``groupby.shift(4)`` —— 后者在报告期有缺口时会静默错配。

    公告日期同样接上滞后（``ann_date_prev`` 等），供 ``add_available_date``
    计算整条记录的可用时点。

    Args:
        panel: (code, period) 索引的报告面板
        columns: 需要滞后的列
        lags: 滞后的财年数，1 → 后缀 ``_prev``，2 → ``_prev2``

    Returns:
        pd.DataFrame: 原面板加上滞后列
    """
    base = panel.reset_index()
    out = base.copy()

    wanted = [c for c in columns if c in base.columns]

    for lag in lags:
        suffix = "_prev" if lag == 1 else f"_prev{lag}"

        # 去重：调用方若把 ann_date 也列进 columns，不能选到两次同名列
        right_cols = list(
            dict.fromkeys(["code", "fiscal_year", "quarter", "ann_date"] + wanted)
        )
        right = base[right_cols].copy()
        right["fiscal_year"] = right["fiscal_year"] + lag

        renamed = list(dict.fromkeys(wanted + ["ann_date"]))
        right = right.rename(columns={c: f"{c}{suffix}" for c in renamed})

        out = out.merge(right, on=["code", "fiscal_year", "quarter"], how="left")

    return out.set_index(["code", "period"]).sort_index()


def add_available_date(panel: pd.DataFrame) -> pd.DataFrame:
    """按行取所有 ``ann_date*`` 列的最大值作为 ``available_date``。

    这是整个 PIT 机制的锚点：一条用到了去年数据的记录，直到去年那张报表也
    公告之后才算可用。追溯调整与延迟披露会让时序倒置，所以必须取 max 而不是
    直接用当期公告日期。
    """
    out = panel.copy()
    ann_cols = [c for c in out.columns if c == "ann_date" or c.startswith("ann_date_")]

    if not ann_cols:
        raise KeyError("面板中没有任何 ann_date 列")

    out["available_date"] = out[ann_cols].max(axis=1)
    return out


def to_single_quarter(
    panel: pd.DataFrame, flow_columns: Iterable[str] = FLOW_COLUMNS
) -> pd.DataFrame:
    """把累计口径的流量项还原成单季值。

    A 股 Q2/Q3/Q4 的流量项是年初至今累计：Q1 原样，``Qn = cum_n - cum_{n-1}``，
    上一季缺失则为 NaN。**只对流量项施加，绝不碰存量项。**

    Args:
        panel: (code, period) 索引，含 fiscal_year / quarter
        flow_columns: 流量列名

    Returns:
        pd.DataFrame: 流量列被替换为单季值的副本
    """
    base = panel.reset_index()
    wanted = [c for c in flow_columns if c in base.columns]

    prior = base[["code", "fiscal_year", "quarter"] + wanted].copy()
    prior["quarter"] = prior["quarter"] + 1
    prior = prior.rename(columns={c: f"__prior_{c}" for c in wanted})

    merged = base.merge(prior, on=["code", "fiscal_year", "quarter"], how="left")

    for col in wanted:
        is_q1 = merged["quarter"] == 1
        merged[col] = merged[col].where(is_q1, merged[col] - merged[f"__prior_{col}"])

    merged = merged.drop(columns=[f"__prior_{c}" for c in wanted])
    return merged.set_index(["code", "period"]).sort_index()


def to_ttm(
    panel: pd.DataFrame,
    flow_columns: Iterable[str] = FLOW_COLUMNS,
    method: str = "annual_anchored",
) -> pd.DataFrame:
    """把累计口径的流量项换算成 TTM（滚动十二个月）。

    - ``annual_anchored``：``TTM = 本期累计 + 上一完整财年 - 去年同期累计``。
      对单个季度缺失更稳健，是默认。
    - ``rolling4``：先还原单季再滚动求和，作为交叉校验。

    Args:
        panel: (code, period) 索引的**累计口径**面板
        flow_columns: 流量列名
        method: ``"annual_anchored"`` 或 ``"rolling4"``

    Returns:
        pd.DataFrame: 流量列被替换为 TTM 值的副本
    """
    wanted = [c for c in flow_columns if c in panel.columns]

    if method == "rolling4":
        single = to_single_quarter(panel, wanted).reset_index()
        single = single.sort_values(["code", "period"])
        for col in wanted:
            single[col] = single.groupby("code")[col].transform(
                lambda s: s.rolling(4, min_periods=4).sum()
            )
        return single.set_index(["code", "period"]).sort_index()

    if method != "annual_anchored":
        raise ValueError(f"未知的 TTM 方法: {method!r}")

    base = panel.reset_index()

    # 去年同期累计
    same_q_last_year = base[["code", "fiscal_year", "quarter"] + wanted].copy()
    same_q_last_year["fiscal_year"] += 1
    same_q_last_year = same_q_last_year.rename(columns={c: f"__ly_{c}" for c in wanted})

    # 上一完整财年（去年 Q4 的累计值）
    last_full_year = base.loc[base["quarter"] == 4, ["code", "fiscal_year"] + wanted]
    last_full_year = last_full_year.copy()
    last_full_year["fiscal_year"] += 1
    last_full_year = last_full_year.rename(columns={c: f"__fy_{c}" for c in wanted})

    merged = base.merge(
        same_q_last_year, on=["code", "fiscal_year", "quarter"], how="left"
    ).merge(last_full_year, on=["code", "fiscal_year"], how="left")

    for col in wanted:
        merged[col] = merged[col] + merged[f"__fy_{col}"] - merged[f"__ly_{col}"]

    drop = [f"__ly_{c}" for c in wanted] + [f"__fy_{c}" for c in wanted]
    return merged.drop(columns=drop).set_index(["code", "period"]).sort_index()


def as_of(panel: pd.DataFrame, date, date_column: str = "available_date"):
    """取截止到 ``date`` 时每只股票**最新的已公开**记录。

    Args:
        panel: (code, period) 索引，含 ``available_date``
        date: 截面日期
        date_column: 可用时点列名

    Returns:
        pd.DataFrame: 以 code 为索引的单日截面
    """
    if date_column not in panel.columns:
        raise KeyError(f"面板缺少 {date_column!r}，请先调用 add_available_date")

    target = pd.Timestamp(date)
    visible = panel[panel[date_column].notna() & (panel[date_column] <= target)]

    if visible.empty:
        return visible.reset_index().set_index("code")

    latest = visible.reset_index().sort_values([date_column, "period"])
    return latest.groupby("code", sort=True).tail(1).set_index("code")


def build_pit_panel(
    panel: pd.DataFrame,
    dates: Sequence,
    fields: Optional[List[str]] = None,
    date_column: str = "available_date",
) -> pd.DataFrame:
    """把报告面板 PIT 对齐到一组截面日期，得到 (date, code) 因子面板。

    用 ``merge_asof(direction="backward")`` 实现 —— 只要 ``available_date``
    算对了，就不可能取到未公开的数据。

    Args:
        panel: (code, period) 索引，含 ``available_date``
        dates: 截面日期序列（如调仓日）
        fields: 需要的列，None 表示全部
        date_column: 可用时点列名

    Returns:
        pd.DataFrame: MultiIndex (date, code)，另含 ``available_date`` 与
            ``period`` 便于审计
    """
    if date_column not in panel.columns:
        raise KeyError(f"面板缺少 {date_column!r}，请先调用 add_available_date")

    right = panel.reset_index()
    right = right[right[date_column].notna()].sort_values(date_column)

    keep = (
        list(right.columns)
        if fields is None
        else (
            ["code", "period", date_column] + [f for f in fields if f in right.columns]
        )
    )
    right = right[list(dict.fromkeys(keep))]

    target_dates = pd.DatetimeIndex(sorted(pd.to_datetime(list(dates))))
    codes = right["code"].unique()

    left = pd.MultiIndex.from_product(
        [target_dates, codes], names=["date", "code"]
    ).to_frame(index=False)
    left = left.sort_values("date")

    merged = pd.merge_asof(
        left,
        right,
        left_on="date",
        right_on=date_column,
        by="code",
        direction="backward",
    )

    return merged.dropna(subset=[date_column]).set_index(["date", "code"]).sort_index()


def listing_dates(price_panel: pd.DataFrame) -> pd.Series:
    """每只股票的首个交易日 —— 上市日期的可靠代理。

    这是修正幸存者偏差中"未来才 IPO 的公司出现在历史截面"那一半的关键输入。
    """
    return price_panel.reset_index().groupby("code")["date"].min()


DEFAULT_SCREEN_RULES = {
    "exclude_st": True,
    "exclude_financials": True,
    "min_listing_days": 365,
    "require_positive_equity": True,
    "exclude_suspended": True,
}


def screen_universe(
    pit_panel: pd.DataFrame,
    price_panel: Optional[pd.DataFrame] = None,
    rules: Optional[Dict] = None,
) -> pd.Series:
    """股票池筛选，返回与 ``pit_panel`` 同索引的布尔掩码。

    ``min_listing_days`` 这条是**强制的时点正确性修正**，不是可选的美化：
    批量财报端点会把当前在册公司的历史财报回填到它们尚未上市的年份。

    Args:
        pit_panel: MultiIndex (date, code) 面板，含 name / total_equity
        price_panel: 行情面板，用于上市日期与停牌判断
        rules: 覆盖 ``DEFAULT_SCREEN_RULES`` 的规则

    Returns:
        pd.Series: 布尔掩码，True 表示保留
    """
    config = {**DEFAULT_SCREEN_RULES, **(rules or {})}
    mask = pd.Series(True, index=pit_panel.index)

    if config["exclude_st"] and "name" in pit_panel.columns:
        name = pit_panel["name"].fillna("").astype(str)
        mask &= ~name.str.contains("|".join(_ST_MARKERS), regex=True)

    if config["exclude_financials"] and "name" in pit_panel.columns:
        name = pit_panel["name"].fillna("").astype(str)
        mask &= ~name.str.contains("|".join(_FINANCIAL_KEYWORDS), regex=True)

    if config["require_positive_equity"] and "total_equity" in pit_panel.columns:
        mask &= pit_panel["total_equity"] > 0

    if price_panel is not None and len(price_panel):
        dates = pit_panel.index.get_level_values("date")
        codes = pit_panel.index.get_level_values("code")

        if config["min_listing_days"]:
            listed = listing_dates(price_panel)
            first_bar = pd.Series(codes.map(listed).values, index=pit_panel.index)
            age_days = (
                pd.Series(dates.values, index=pit_panel.index) - first_bar
            ).dt.days
            mask &= age_days >= config["min_listing_days"]

        if config["exclude_suspended"] and "volume" in price_panel.columns:
            volume = price_panel["volume"].reindex(pit_panel.index)
            mask &= volume.notna() & (volume > 0)

    return mask.fillna(False)
