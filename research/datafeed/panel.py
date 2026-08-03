# -*- coding: utf-8 -*-
"""面板 schema 与代码格式的统一，以及向仓库既有约定的桥接。

各数据源的股票代码格式互不相同（成分股接口给 6 位裸码、新浪要 ``sh`` 前缀、
仓库其余部分用 ``.SH`` 后缀），这里收敛到唯一规范形式并提供转换器。

面板约定
--------
- **行情面板**：MultiIndex ``(date: Timestamp, code: str)``，列为
  open/high/low/close/volume/amount/outstanding_share
- **因子面板**：MultiIndex ``(date, code)``，若干 float 列
- **报告面板**：MultiIndex ``(code, period: Timestamp)``，另有 ``ann_date`` 列

选 ``(date, code)`` 是因为截面操作最自然：``panel.loc[d]`` 取单个截面，
``unstack("code")`` 得到 T×N 矩阵供排序与协方差使用。
"""

from typing import Dict

import pandas as pd

# 交易所后缀推断：6 开头沪市（含 688 科创板），0/3 深市（含 300 创业板），
# 4/8/92 北交所
_SH_PREFIXES = ("6",)
_SZ_PREFIXES = ("0", "3")
_BJ_PREFIXES = ("4", "8", "92")


def normalize_code(raw: str) -> str:
    """把任意常见格式的 A 股代码归一为 ``"600519.SH"``。

    支持 ``600519`` / ``sh600519`` / ``SH600519`` / ``600519.SH`` /
    ``600519.sh``。

    Args:
        raw: 原始代码

    Returns:
        str: ``"<6位数字>.<SH|SZ|BJ>"``

    Raises:
        ValueError: 无法解析出 6 位数字，或前缀不属于已知交易所
    """
    if raw is None:
        raise ValueError("股票代码不能为 None")

    text = str(raw).strip().upper().replace("_", ".")

    # 去掉已有的交易所标记，只留数字
    for token in (".SH", ".SZ", ".BJ", ".XSHG", ".XSHE"):
        if text.endswith(token):
            text = text[: -len(token)]
            break
    for token in ("SH", "SZ", "BJ"):
        if text.startswith(token) and len(text) > len(token):
            text = text[len(token) :]
            break

    # 先确认剩下的确实是数字再补零：否则空串会被 zfill 成 "000000" 而静默通过
    if not text or not text.isdigit():
        raise ValueError(f"无法解析股票代码: {raw!r}")

    digits = text.zfill(6)
    if len(digits) != 6:
        raise ValueError(f"无法解析股票代码: {raw!r}")

    if digits.startswith(_SH_PREFIXES):
        market = "SH"
    elif digits.startswith(_BJ_PREFIXES):
        market = "BJ"
    elif digits.startswith(_SZ_PREFIXES):
        market = "SZ"
    else:
        raise ValueError(f"未知交易所前缀: {raw!r}")

    return f"{digits}.{market}"


def to_sina_symbol(code: str) -> str:
    """``"600519.SH"`` → ``"sh600519"``（新浪行情接口的入参格式）。

    Raises:
        ValueError: 北交所标的，新浪该接口不支持
    """
    normalized = normalize_code(code)
    digits, market = normalized.split(".")
    if market == "BJ":
        raise ValueError(f"新浪日线接口不支持北交所标的: {code!r}")
    return f"{market.lower()}{digits}"


def to_em_symbol(code: str) -> str:
    """``"600519.SH"`` → ``"SH600519"``（东方财富逐股接口的入参格式）。"""
    digits, market = normalize_code(code).split(".")
    return f"{market}{digits}"


def to_code_dict(panel: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """MultiIndex 面板 → 仓库既有的 ``{代码: DataFrame}`` 约定。

    每个 DataFrame 的 index 为 ``YYYYMMDD`` 字符串，与
    ``data_loader/data_loader.py`` 的产物一致，可直接喂给 ``backtest/``。

    Args:
        panel: MultiIndex (date, code) 面板

    Returns:
        dict: {规范代码: 按日期升序的 DataFrame}
    """
    result = {}
    for code, sub in panel.groupby(level="code", sort=True):
        frame = sub.reset_index(level="code", drop=True).sort_index()
        frame.index = frame.index.strftime("%Y%m%d")
        result[str(code)] = frame
    return result


def from_code_dict(
    data: Dict[str, pd.DataFrame], date_col: str = "date"
) -> pd.DataFrame:
    """``{代码: DataFrame}`` → MultiIndex (date, code) 面板。

    日期既可以在索引上，也可以是名为 ``date_col`` 的列 —— 仓库两种写法都有。

    Args:
        data: {代码: 行情 DataFrame}
        date_col: 日期列名（若日期在列上）

    Returns:
        pd.DataFrame: MultiIndex (date, code) 面板，按日期、代码排序
    """
    frames = []
    for code, frame in data.items():
        frame = frame.copy()

        if date_col in frame.columns:
            index = pd.to_datetime(frame[date_col])
            frame = frame.drop(columns=[date_col])
        else:
            index = pd.to_datetime(frame.index)

        frame.index = pd.MultiIndex.from_arrays(
            [index, [normalize_code(code)] * len(frame)], names=["date", "code"]
        )
        frames.append(frame)

    if not frames:
        return pd.DataFrame(
            index=pd.MultiIndex.from_arrays([[], []], names=["date", "code"])
        )

    return pd.concat(frames).sort_index()


def to_wide(panel: pd.DataFrame, field: str) -> pd.DataFrame:
    """从面板中取一个字段，展开成 ``(date × code)`` 宽表。

    Args:
        panel: MultiIndex (date, code) 面板
        field: 列名

    Returns:
        pd.DataFrame: index 为日期、columns 为代码的宽表
    """
    if field not in panel.columns:
        raise KeyError(f"面板中没有列 {field!r}，现有列: {list(panel.columns)}")
    return panel[field].unstack("code").sort_index()
