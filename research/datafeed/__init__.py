# -*- coding: utf-8 -*-
"""datafeed —— akshare 数据源、磁盘缓存与时点（PIT）对齐。

子模块刻意不在此重导入 ``akshare_source``：akshare 是可选重依赖，且导入即
可能触发网络相关的初始化，应在真正取数时才引入。
"""

from research.datafeed.panel import (
    from_code_dict,
    normalize_code,
    to_code_dict,
    to_sina_symbol,
    to_wide,
)
from research.datafeed.proxy import direct_connection, ensure_direct_connection

__all__ = [
    "ensure_direct_connection",
    "direct_connection",
    "normalize_code",
    "to_sina_symbol",
    "to_code_dict",
    "from_code_dict",
    "to_wide",
]
