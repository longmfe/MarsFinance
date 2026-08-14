# -*- coding: utf-8 -*-
"""akshare 端点的薄封装：延迟导入 + 直连 + 磁盘缓存。

本层刻意返回**原始**的 akshare 数据帧（东方财富的列名是中文），列名重命名
交给 ``fundamentals.py``，这样接口 schema 漂移只会在一个地方暴露出来。

本机实测的可用性（决定了整个数据层的形状）：

============================================  ==================  ======
端点                                          主机                状态
============================================  ==================  ======
stock_zcfz_em / stock_lrb_em / stock_xjll_em  datacenter-web.em   可用
stock_balance_sheet_by_report_em              datacenter-web.em   可用
stock_zh_a_daily                              新浪                可用
fund_etf_hist_sina                            新浪                可用
futures_main_sina                             新浪                可用
index_stock_cons                              新浪                可用
stock_zh_index_daily                          腾讯                可用
stock_zh_a_hist / stock_zh_a_spot_em          push2.eastmoney     **不可用**
index_stock_cons_csindex                      中证指数            **不可用**
============================================  ==================  ======

失效的两组端点不要再用：``stock_zh_a_hist`` 走 ``80.push2.eastmoney.com``，
本机连接被拒；中证指数的接口返回的不是 Excel 而是错误页。
``fund_etf_hist_sina`` 与 ``stock_zh_a_daily`` 同属新浪 realstock 主机，
同样可用；若某天失效，备选是东方财富的 ``fund_etf_hist_em``（push2 系，
本机未验证）。
"""

import re
from typing import Optional

import pandas as pd

from research.datafeed.cache import cached
from research.datafeed.panel import to_em_symbol, to_sina_symbol
from research.datafeed.proxy import ensure_direct_connection


def _ak():
    """延迟导入 akshare 并确保直连，未安装时给出可操作的报错。"""
    try:
        import akshare
    except ImportError as exc:
        raise ImportError(
            "缺少 akshare：pip install -r requirements-research.txt"
        ) from exc

    ensure_direct_connection()
    return akshare


def fetch_balance_sheet(period: str, force: bool = False) -> pd.DataFrame:
    """全市场资产负债表（按报告期批量，一次约 16s / 5000+ 行）。

    Args:
        period: 报告期 ``YYYYMMDD``，如 ``"20231231"``
        force: 忽略缓存重新取数

    Returns:
        pd.DataFrame: 含 股票代码/资产-总资产/负债-总负债/资产负债率/
            股东权益合计/公告日期 等列
    """
    return cached(
        "stock_zcfz_em",
        {"date": period},
        lambda: _ak().stock_zcfz_em(date=period),
        force=force,
    )


def fetch_income_statement(period: str, force: bool = False) -> pd.DataFrame:
    """全市场利润表（按报告期批量，一次约 12s）。

    Returns:
        pd.DataFrame: 含 净利润/营业总收入/营业总支出-营业支出/公告日期 等列
    """
    return cached(
        "stock_lrb_em",
        {"date": period},
        lambda: _ak().stock_lrb_em(date=period),
        force=force,
    )


def fetch_cashflow_statement(period: str, force: bool = False) -> pd.DataFrame:
    """全市场现金流量表（按报告期批量，一次约 16s）。

    Returns:
        pd.DataFrame: 含 经营性现金流-现金流量净额/公告日期 等列
    """
    return cached(
        "stock_xjll_em",
        {"date": period},
        lambda: _ak().stock_xjll_em(date=period),
        force=force,
    )


def fetch_balance_sheet_detail(code: str, force: bool = False) -> pd.DataFrame:
    """单只股票的完整资产负债表（逐股，约 27s，含 102 个报告期 × 319 列）。

    只有 Piotroski 的 F6（流动比率变化）和 F5 的 ``noncurrent`` 口径需要它 ——
    批量端点没有流动资产/流动负债的拆分。代价是每只 27s，故仅对目标股票池拉取。

    Args:
        code: 任意格式的股票代码，内部转成 ``"SH600519"``

    Returns:
        pd.DataFrame: 含 REPORT_DATE/NOTICE_DATE/TOTAL_ASSETS/
            TOTAL_CURRENT_ASSETS/TOTAL_CURRENT_LIAB/TOTAL_EQUITY 等列
    """
    symbol = to_em_symbol(code)
    return cached(
        "stock_balance_sheet_by_report_em",
        {"symbol": symbol},
        lambda: _ak().stock_balance_sheet_by_report_em(symbol=symbol),
        force=force,
    )


def fetch_daily_bars(
    code: str,
    start_date: str = "20050101",
    end_date: Optional[str] = None,
    adjust: str = "hfq",
    force: bool = False,
) -> pd.DataFrame:
    """单只股票的日线（新浪源，约 3.6s）。

    默认后复权：后复权不改写历史价格，是时点安全的；前复权用未来分红重写历史，
    本身带轻微未来函数。

    Args:
        code: 任意格式的股票代码
        start_date: 起始日 ``YYYYMMDD``
        end_date: 结束日 ``YYYYMMDD``，None 表示至今
        adjust: ``"hfq"`` 后复权 / ``"qfq"`` 前复权 / ``""`` 不复权

    Returns:
        pd.DataFrame: 含 date/open/high/low/close/volume/amount/
            outstanding_share/turnover
    """
    symbol = to_sina_symbol(code)
    end = end_date or pd.Timestamp.today().strftime("%Y%m%d")

    return cached(
        "stock_zh_a_daily",
        {"symbol": symbol, "start": start_date, "end": end, "adjust": adjust},
        lambda: _ak().stock_zh_a_daily(
            symbol=symbol, start_date=start_date, end_date=end, adjust=adjust
        ),
        force=force,
    )


def fetch_index_constituents(symbol: str = "000300", force: bool = False):
    """指数成分股（新浪源）。

    **注意：这是当前成分，不是时点成分。** 中证指数的历史成分接口在本机不可用，
    因此股票池的时点正确性只能靠上市日期过滤来保证（见 ``fundamentals.py``
    的 ``screen_universe``）。

    Returns:
        pd.DataFrame: 含 品种代码/品种名称/纳入日期
    """
    return cached(
        "index_stock_cons",
        {"symbol": symbol},
        lambda: _ak().index_stock_cons(symbol=symbol),
        force=force,
    )


def fetch_index_daily(symbol: str = "sh000300", force: bool = False) -> pd.DataFrame:
    """指数日线（腾讯源）。用作基准，也是交易日历的来源。

    Args:
        symbol: 带市场前缀的指数代码，如 ``"sh000300"``

    Returns:
        pd.DataFrame: 含 date/open/high/low/close/volume
    """
    return cached(
        "stock_zh_index_daily",
        {"symbol": symbol},
        lambda: _ak().stock_zh_index_daily(symbol=symbol),
        force=force,
    )


def _to_sina_etf_symbol(code: str) -> str:
    """ETF 代码 → 新浪接口格式：``"510310"`` → ``"sh510310"``。

    ETF 的代码约定与股票不同：沪市 ETF 以 5 开头（51/56/58/588），
    深市 ETF 以 1 开头（15/16）。``panel.normalize_code`` 只认股票前缀，
    故这里单独处理，不改动共享的股票代码约定。

    Args:
        code: ETF 代码，如 ``"510310"`` / ``"159915"``

    Returns:
        str: 带市场前缀的新浪格式代码

    Raises:
        ValueError: 不是 6 位数字，或前缀不属于已知 ETF 市场
    """
    text = str(code).strip().lower()
    if text.startswith("sh") or text.startswith("sz"):
        text = text[2:]
    if not text.isdigit() or len(text) != 6:
        raise ValueError(f"无法解析 ETF 代码: {code!r}")

    if text.startswith("5"):
        return f"sh{text}"
    if text.startswith("1"):
        return f"sz{text}"
    raise ValueError(f"未知 ETF 交易所前缀: {code!r} (沪市 5 开头，深市 1 开头)")


def fetch_etf_daily(code: str = "510310", force: bool = False) -> pd.DataFrame:
    """单只 ETF 的日线（新浪源，全历史，约 1~3s）。

    与 ``fetch_daily_bars`` 的区别：ETF 走 ``fund_etf_hist_sina`` 端点
    （``stock_zh_a_daily`` 只覆盖股票）。首次调用联网取数并落盘缓存，
    之后离线且秒级。

    Args:
        code: ETF 代码，如 ``"510310"``（沪深300ETF）、``"159915"``
        force: 忽略缓存重新取数

    Returns:
        pd.DataFrame: 含 date/open/high/low/close/volume，按日期升序；
        date 已归一为 ``datetime64``（新浪原始返回 ``datetime.date``，
        直接落盘会被 ``normalize_for_parquet`` 误判为 object 列而破坏）
    """
    symbol = _to_sina_etf_symbol(code)

    raw = cached(
        "fund_etf_hist_sina",
        {"symbol": symbol},
        lambda: _ak().fund_etf_hist_sina(symbol=symbol),
        force=force,
    )

    if raw.empty:
        return raw

    out = raw.copy()
    out["date"] = pd.to_datetime(out["date"])
    return out.sort_values("date").reset_index(drop=True)


# 新浪期货日线端点返回的中文列名 → 仓库统一英文列名
_FUTURES_COLUMN_MAP = {
    "日期": "date",
    "开盘价": "open",
    "最高价": "high",
    "最低价": "low",
    "收盘价": "close",
    "成交量": "volume",
    "持仓量": "hold",
    "动态结算价": "settle",
}


def _validate_futures_symbol(symbol: str) -> str:
    """期货代码校验并归一为大写：``"if0"`` → ``"IF0"``。

    新浪格式：品种字母（1~3 位）+ 数字。``IF0`` 是主力连续（已复权拼接，
    换月无跳空，适合直接回测），``IF2508`` 是具体合约。

    Raises:
        ValueError: 不是合法的新浪期货代码格式
    """
    text = str(symbol).strip().upper()
    if not re.fullmatch(r"[A-Z]{1,3}\d{1,4}", text):
        raise ValueError(f"无法解析期货代码: {symbol!r} (新浪格式如 'IF0' / 'IF2508')")
    return text


def fetch_futures_daily(symbol: str = "IF0", force: bool = False) -> pd.DataFrame:
    """单品种期货日线（新浪源，主力连续为已复权拼接）。

    首次调用联网取数并落盘缓存，之后离线且秒级。``IF0`` 的可用历史
    约从 2017 年起（新浪只保留这么多）。

    Args:
        symbol: 期货代码，``"IF0"`` 主力连续 / ``"IF2508"`` 具体合约
        force: 忽略缓存重新取数

    Returns:
        pd.DataFrame: 含 date/open/high/low/close/volume/hold/settle，
        按日期升序；date 已归一为 ``datetime64``
    """
    symbol = _validate_futures_symbol(symbol)

    raw = cached(
        "futures_main_sina",
        {"symbol": symbol},
        lambda: _ak().futures_main_sina(symbol=symbol),
        force=force,
    )

    if raw.empty:
        return raw

    out = raw.rename(columns=_FUTURES_COLUMN_MAP).copy()
    out["date"] = pd.to_datetime(out["date"])
    return out.sort_values("date").reset_index(drop=True)
