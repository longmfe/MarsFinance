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
index_stock_cons                              新浪                可用
stock_zh_index_daily                          腾讯                可用
stock_zh_a_hist / stock_zh_a_spot_em          push2.eastmoney     **不可用**
index_stock_cons_csindex                      中证指数            **不可用**
============================================  ==================  ======

失效的两组端点不要再用：``stock_zh_a_hist`` 走 ``80.push2.eastmoney.com``，
本机连接被拒；中证指数的接口返回的不是 Excel 而是错误页。
"""

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
