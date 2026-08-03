# -*- coding: utf-8 -*-
"""事件对齐：把业绩预告事件按公告日期挂到日线行情上。

**锚定规则**：公告日期为 A 的事件，锚定在 **最后一个 date <= A 的 bar**。

回测引擎的契约是：第 i 日策略只能看到 ``data.iloc[:i]``（末行为 i-1），
成交发生在 ``data.iloc[i]['close']``——即策略的返回值总是**下一根 bar 成交**。
于是锚点落在 p（``index[p] <= A < index[p+1]``），策略在 p 出信号，引擎在 p+1
成交，而 p+1 的日期**严格晚于公告日 A**。由此得到一条可测试的不变量：

    对每一笔买入（引擎下标 i）：``data.index[i] > data['yjyg_notice'].iloc[i-1]``

公告落在周末时锚点是上一个交易日（日期早于 A），这不是泄漏：该 bar 上的标记
只会影响**下一根** bar 的成交，而下一根仍晚于 A。把不变量定义在**成交日**而非
标记日上，正是这一设计成立的原因。

本模块只依赖 pandas/numpy——不引入 akshare，也不引入 xtquant，
因此全部正确性逻辑都能在合成数据上离线单测。
"""

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

# 附加到行情上的事件列
YJYG_COLUMNS = (
    "yjyg_age",
    "yjyg_type",
    "yjyg_amp",
    "yjyg_notice",
    "yjyg_period",
    "yjyg_fill_ok_next",
    "yjyg_locked_up",
)

# 创业板涨跌幅由 10% 改为 20% 的生效日
CHINEXT_20PCT_DATE = "20200824"


def limit_rate(code: str, date_str: str) -> float:
    """按板块与日期返回当日涨跌停幅度。

    Args:
        code: 股票代码，QMT 格式如 '300750.SZ'，也接受裸六位
        date_str: 交易日 'YYYYMMDD'

    Returns:
        float: 涨跌停比例，如 0.10 / 0.20 / 0.30

    Note:
        未建模 ST/*ST 的 5% 限制（无可靠免费数据源；沪深300 按指数规则
        不含 ST，扩展到全市场时才需要处理），也未建模新股上市初期的无涨跌幅区间。
    """
    num = str(code).split(".")[0]
    suffix = str(code).split(".")[-1].upper() if "." in str(code) else ""

    if suffix == "BJ" or num.startswith(("43", "83", "87", "88", "920")):
        return 0.30  # 北交所
    if num.startswith(("688", "689")):
        return 0.20  # 科创板
    if num.startswith(("300", "301")):  # 创业板
        return 0.20 if str(date_str) >= CHINEXT_20PCT_DATE else 0.10
    return 0.10  # 主板


def compute_tradability(
    price_df: pd.DataFrame, code: str, tol: float = 0.005
) -> pd.DataFrame:
    """由 preClose/low/high/volume 计算一字板与停牌标记。

    判定用 ``low`` 而非 ``close``：引擎按收盘价成交，只要当日**曾经**跌破涨停价
    （low < 涨停价），说明该价位真实成交过，按收盘价成交是可实现的；只有 ``low``
    本身也钉在涨停价上，才是封死的一字板，买不进去。

    用比值而非 ``round(preClose*1.1, 2)`` 相等判断：``low`` 与 ``preClose`` 在复权
    下同比例缩放，比值对复权方式不变，因此不复权与后复权数据都适用。
    tol 用于吸收交易所报价的分位取整。

    Args:
        price_df: 行情 DataFrame，需含 preClose/low/high/volume，索引为 'YYYYMMDD'
        code: 股票代码，用于确定涨跌幅
        tol: 容差

    Returns:
        pd.DataFrame: 列为 locked_up / locked_down / halted / buy_ok / sell_ok
    """
    index = price_df.index
    pre = pd.to_numeric(price_df["preClose"], errors="coerce")
    low = pd.to_numeric(price_df["low"], errors="coerce")
    high = pd.to_numeric(price_df["high"], errors="coerce")
    volume = pd.to_numeric(price_df["volume"], errors="coerce")

    rates = np.array([limit_rate(code, d) for d in index], dtype=float)

    halted = (volume <= 0) | pre.isna() | (pre <= 0) | volume.isna()

    # preClose 非法时比值无意义，交给 halted 兜住
    safe_pre = pre.where(pre > 0)
    locked_up = ((low / safe_pre - 1.0) >= (rates - tol)).fillna(False)
    locked_down = ((high / safe_pre - 1.0) <= (-rates + tol)).fillna(False)

    halted = halted.fillna(True)

    return pd.DataFrame(
        {
            "locked_up": locked_up.astype(bool),
            "locked_down": locked_down.astype(bool),
            "halted": halted.astype(bool),
            "buy_ok": (~locked_up & ~halted).astype(bool),
            "sell_ok": (~locked_down & ~halted).astype(bool),
        },
        index=index,
    )


def anchor_positions(index_values: np.ndarray, notice_dates: np.ndarray) -> np.ndarray:
    """把公告日期锚定到 bar 序号；-1 表示公告早于行情起点。

    ``side='right'`` 给出等值元素**之后**的插入点，减 1 即落在最后一个
    ``date <= notice`` 的 bar 上。
    """
    if len(notice_dates) == 0:
        return np.array([], dtype=int)
    return np.searchsorted(index_values, notice_dates, side="right") - 1


def attach_yjyg_columns(
    price_df: pd.DataFrame,
    events: Optional[pd.DataFrame],
    code: str,
) -> pd.DataFrame:
    """把单只股票的业绩预告事件附加为 ``yjyg_*`` 列。

    Args:
        price_df: 该股票的日线行情，索引为 'YYYYMMDD' 升序字符串
        events: 该股票的事件（列同 ``yjyg_loader.load_yjyg_events``），可为 None
        code: 股票代码

    Returns:
        pd.DataFrame: 原行情列 + YJYG_COLUMNS，索引不变
    """
    out = price_df.copy()
    n = len(out)

    index_values = out.index.to_numpy()
    if n > 1 and not np.all(index_values[:-1] <= index_values[1:]):
        # 索引乱序会让 searchsorted 静默锚错 bar，直接制造未来函数
        raise ValueError(f"{code}: 行情索引必须按日期升序排列")

    trad = compute_tradability(out, code)
    out["yjyg_locked_up"] = trad["locked_up"].to_numpy()
    # 下一根 bar 是否可买。末根 bar 未知，按 False 处理（保守）。
    out["yjyg_fill_ok_next"] = (
        trad["buy_ok"].shift(-1, fill_value=False).astype(bool).to_numpy()
    )

    age = np.full(n, np.nan)
    type_arr = np.full(n, None, dtype=object)
    amp_arr = np.full(n, np.nan)
    notice_arr = np.full(n, None, dtype=object)
    period_arr = np.full(n, None, dtype=object)

    if events is not None and len(events) > 0 and n > 0:
        ev = events.sort_values(["notice_date", "period"], kind="mergesort")
        ev = ev.reset_index(drop=True)
        pos = anchor_positions(index_values, ev["notice_date"].to_numpy())

        # pos < 0：公告早于行情起点。丢弃，绝不夹到 0——夹到 0 会在首根 bar
        # 上凭空造出一个事件。代价是窗口开头的事件覆盖不全，
        # 因此调用方应比分析区间多加载约一个季度的行情。
        keep = pos >= 0
        pos = pos[keep]
        ev = ev.loc[keep].reset_index(drop=True)

        if len(ev) > 0:
            # 事件已按公告日排序，同一 bar 上的多个事件后写覆盖先写（确定性）
            ev_slot = np.full(n, -1, dtype=int)
            ev_slot[pos] = np.arange(len(ev))
            is_anchor = ev_slot >= 0

            bar_no = np.arange(n, dtype=float)
            anchor_no = (
                pd.Series(np.where(is_anchor, bar_no, np.nan)).ffill().to_numpy()
            )
            age = bar_no - anchor_no

            ev_id = (
                pd.Series(np.where(is_anchor, ev_slot.astype(float), np.nan))
                .ffill()
                .to_numpy()
            )
            filled = ~np.isnan(ev_id)
            sel = ev_id[filled].astype(int)

            type_arr[filled] = ev["type"].to_numpy(dtype=object)[sel]
            amp_arr[filled] = pd.to_numeric(ev["amp"], errors="coerce").to_numpy()[sel]
            notice_arr[filled] = ev["notice_date"].to_numpy(dtype=object)[sel]
            period_arr[filled] = ev["period"].to_numpy(dtype=object)[sel]

    out["yjyg_age"] = age
    out["yjyg_type"] = type_arr
    out["yjyg_amp"] = amp_arr
    out["yjyg_notice"] = notice_arr
    out["yjyg_period"] = period_arr

    return out


def attach_yjyg_to_universe(
    stock_data_dict: Dict[str, pd.DataFrame],
    events: pd.DataFrame,
) -> Tuple[Dict[str, pd.DataFrame], dict]:
    """全池附加事件列，并返回覆盖率诊断。

    诊断里的 ``events_not_in_universe`` 就是**幸存者偏差的量化值**：事件表本身
    没有幸存者偏差（它是每个报告期的全市场截面），有偏的是价格股票池——
    ``get_hs300_stock_list()`` 返回的是**今天**的成分股。本函数不假装修复它，
    只把它变成一个打印出来的数字。调用方若持有时点成分股名单，
    直接传入对应的 ``stock_data_dict`` 即可，无需改动本模块。

    Args:
        stock_data_dict: {股票代码: 行情 DataFrame}
        events: 事件表，列同 ``yjyg_loader.load_yjyg_events``

    Returns:
        tuple: (附加后的字典, 诊断 dict)
    """
    if events is None:
        events = pd.DataFrame(columns=["code", "notice_date", "period", "type", "amp"])

    universe = set(stock_data_dict.keys())
    in_universe = events[events["code"].isin(universe)] if len(events) else events
    n_total = len(events)
    n_in_universe = len(in_universe)

    by_code = (
        {code: grp for code, grp in in_universe.groupby("code")}
        if n_in_universe
        else {}
    )

    attached = {}
    n_anchored = 0
    n_before_window = 0
    n_stocks_with_events = 0

    for code, price_df in stock_data_dict.items():
        ev = by_code.get(code)
        attached[code] = attach_yjyg_columns(price_df, ev, code)

        if ev is not None and len(ev) > 0:
            n_stocks_with_events += 1
            pos = anchor_positions(
                price_df.index.to_numpy(),
                ev.sort_values(["notice_date", "period"], kind="mergesort")[
                    "notice_date"
                ].to_numpy(),
            )
            n_before_window += int((pos < 0).sum())
            n_anchored += int((pos >= 0).sum())

    diagnostics = {
        "events_total": n_total,
        "events_in_universe": n_in_universe,
        "events_not_in_universe": n_total - n_in_universe,
        "events_anchored": n_anchored,
        "events_before_price_window": n_before_window,
        "stocks_total": len(stock_data_dict),
        "stocks_with_events": n_stocks_with_events,
    }
    return attached, diagnostics


def print_coverage(diagnostics: dict) -> None:
    """打印事件覆盖率诊断。"""
    d = diagnostics
    print("=" * 70)
    print("业绩预告事件覆盖率")
    print("=" * 70)
    print(f"事件总数:                 {d['events_total']}")
    print(
        f"落在股票池内:             {d['events_in_universe']}  "
        f"(池外丢弃 {d['events_not_in_universe']} —— 幸存者偏差量化值)"
    )
    print(f"成功锚定到行情:           {d['events_anchored']}")
    print(f"公告早于行情起点被丢弃:   {d['events_before_price_window']}")
    print(
        f"股票池:                   {d['stocks_with_events']}/{d['stocks_total']} 只含事件"
    )
