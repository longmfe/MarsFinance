# -*- coding: utf-8 -*-
"""绕开 Windows 系统代理，保证 akshare 的网络请求能直连。

背景（本机实测）：未处理时 ``ak.stock_zh_a_hist`` 抛 ``ProxyError``，
``ak.index_stock_cons_csindex`` 拿到的是代理返回的错误页；清空代理环境变量
并置 ``NO_PROXY='*'`` 之后，同样的主机直连返回 200。

akshare 内部是模块级的 ``requests.get(...)`` 调用，拿不到
``Session.trust_env`` 这个开关，因此**环境变量是唯一的杠杆**。
"""

import os
from contextlib import contextmanager

PROXY_ENV_VARS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)

_APPLIED = False


def ensure_direct_connection(force: bool = False) -> None:
    """清除代理环境变量并设置 ``NO_PROXY='*'``。

    **副作用：修改 os.environ。** 幂等，进程内只真正执行一次。

    Args:
        force: 为 True 时忽略"已执行"标志，重新应用一次
    """
    global _APPLIED
    if _APPLIED and not force:
        return

    for var in PROXY_ENV_VARS:
        os.environ.pop(var, None)

    os.environ["NO_PROXY"] = "*"
    os.environ["no_proxy"] = "*"
    _APPLIED = True


@contextmanager
def direct_connection():
    """``ensure_direct_connection`` 的上下文管理器版本，退出时恢复原值。

    仅在需要局部生效、不想污染整个进程时使用；取数路径用
    ``ensure_direct_connection`` 即可。
    """
    saved = {var: os.environ.get(var) for var in PROXY_ENV_VARS}
    saved["NO_PROXY"] = os.environ.get("NO_PROXY")
    saved["no_proxy"] = os.environ.get("no_proxy")

    try:
        ensure_direct_connection(force=True)
        yield
    finally:
        global _APPLIED
        for var, value in saved.items():
            if value is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = value
        _APPLIED = False
