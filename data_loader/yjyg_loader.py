# -*- coding: utf-8 -*-
"""业绩预告事件加载器（数据源：akshare / 东方财富）。

事件驱动策略的关键在于**公告日期**：只有公告落地之后，信息才是公开可交易的。
``ak.stock_yjyg_em`` 返回的 ``公告日期`` 与报告期相差中位数约 20 个自然日，
且有约 6% 的公告发生在报告期结束**之前**（最早提前 165 天、最晚滞后 121 天），
因此任何按报告期对齐的做法都会引入未来函数。本模块只输出公告日期。

akshare 采用延迟导入（与 ``data_loader._xtdata`` 同一模式），
未安装 akshare 时不影响包的其余部分使用。
"""

import os
from typing import List, Optional

import pandas as pd

# 默认口径：归母净利润。akshare 返回的粒度是 (股票代码, 预测指标)，
# 一只股票一个报告期最多 6 行（归母净利润/扣非净利润/每股收益/营业收入/
# 非经常性损益/净利润），**且各行的预告类型与变动幅度可以互相矛盾**
# （实测有同一天归母净利润「略增」而扣非「略减」的案例）。
# 不做该过滤会重复计数事件并产生自相矛盾的信号。
DEFAULT_INDICATOR = "归属于上市公司股东的净利润"

# 上交所 / 深交所 / 北交所代码前缀 -> QMT 后缀。
# 200(深B) 与 900(沪B) 为 B 股，不纳入。
_SH_PREFIXES = ("600", "601", "603", "605", "688", "689")
_SZ_PREFIXES = ("000", "001", "002", "003", "300", "301")
_BJ_PREFIXES = ("43", "83", "87", "88", "920")


def _akshare():
    """延迟导入 akshare，未安装时给出可操作的报错信息。"""
    try:
        import akshare
    except ImportError as exc:
        raise ImportError(
            "缺少 akshare：业绩预告数据依赖它，请执行 `pip install akshare>=1.15`。"
        ) from exc
    return akshare


def quarter_ends(start_date: str, end_date: str) -> List[str]:
    """生成区间内的季末报告期列表（YYYYMMDD）。

    ``ak.stock_yjyg_em`` 的 ``date`` 参数必须是季末日期，
    传入非季末（如 20240401）会抛出 TypeError。

    Args:
        start_date: 开始日期 (YYYYMMDD)
        end_date: 结束日期 (YYYYMMDD)

    Returns:
        list: 升序的季末日期字符串列表
    """
    periods = []
    start_year = int(start_date[:4])
    end_year = int(end_date[:4])
    for year in range(start_year, end_year + 1):
        for month_day in ("0331", "0630", "0930", "1231"):
            period = f"{year}{month_day}"
            if start_date <= period <= end_date:
                periods.append(period)
    return periods


def to_qmt_code(code6: str) -> Optional[str]:
    """六位代码转 QMT 格式；B 股与无法识别的代码返回 None。

    Args:
        code6: 六位数字代码，如 '600519'

    Returns:
        str | None: 如 '600519.SH'；B股/未知返回 None
    """
    code6 = str(code6).strip()
    if len(code6) != 6 or not code6.isdigit():
        return None
    if code6.startswith(_SH_PREFIXES):
        return f"{code6}.SH"
    if code6.startswith(_SZ_PREFIXES):
        return f"{code6}.SZ"
    if code6.startswith(_BJ_PREFIXES):
        return f"{code6}.BJ"
    # 200 深B / 900 沪B 及其余未知前缀
    return None


def fetch_yjyg_period(period: str) -> pd.DataFrame:
    """抓取单个报告期的业绩预告原始表。

    空报告期或无效报告期返回空 DataFrame，而不是让 akshare 的
    ``TypeError: 'NoneType' object is not subscriptable`` 冒出来。

    Args:
        period: 季末报告期 (YYYYMMDD)

    Returns:
        pd.DataFrame: akshare 原始返回；无数据时为空表
    """
    ak = _akshare()
    try:
        return ak.stock_yjyg_em(date=period)
    except TypeError:
        # 该报告期无数据：接口返回 result: null，akshare 未做保护
        return pd.DataFrame()
    except ValueError as exc:
        # akshare 用位置赋值 big_df.columns = [...]，上游增删字段会导致
        # 所有报告期同时失效——这也是本模块坚持落盘缓存的原因。
        raise ValueError(
            f"akshare 解析业绩预告失败（报告期 {period}）：{exc}。"
            "很可能是东方财富接口字段变动，需要升级 akshare 或修正列映射。"
        ) from exc


def load_yjyg_events(
    start_date: str,
    end_date: str,
    indicator: str = DEFAULT_INDICATOR,
    cache_dir: str = "data/yjyg",
    refresh: bool = False,
) -> pd.DataFrame:
    """加载并规范化业绩预告事件表。

    Args:
        start_date: 开始日期 (YYYYMMDD)，按报告期筛选
        end_date: 结束日期 (YYYYMMDD)
        indicator: 预测指标口径，默认归母净利润
        cache_dir: 每个报告期的 parquet 缓存目录
        refresh: True 时忽略缓存重新抓取

    Returns:
        pd.DataFrame: 列为 code / notice_date / period / type / amp / name
            - code:        QMT 格式，如 '600519.SH'
            - notice_date: 公告日期，'YYYYMMDD' 字符串
            - period:      报告期，'YYYYMMDD' 字符串
            - type:        预告类型（预增/略增/扭亏/...，共 11 类）
            - amp:         业绩变动幅度（%）
    """
    periods = quarter_ends(start_date, end_date)
    if not periods:
        return _empty_events()

    frames = []
    for period in periods:
        raw = _load_period_cached(period, cache_dir, refresh)
        if raw.empty:
            continue
        frames.append(_normalize_period(raw, period, indicator))

    frames = [f for f in frames if not f.empty]
    if not frames:
        return _empty_events()

    events = pd.concat(frames, ignore_index=True)

    # 同一 (code, notice_date) 可能对应多个报告期（如三季报与年度预告同日发布），
    # 保留较晚的报告期——信息更前瞻。排序后 keep='last' 结果是确定的。
    events = events.sort_values(["code", "notice_date", "period"], kind="mergesort")
    events = events.drop_duplicates(subset=["code", "notice_date"], keep="last")

    return events.reset_index(drop=True)


def _empty_events() -> pd.DataFrame:
    """空事件表（保持列与 dtype 稳定，便于下游无分支处理）。"""
    return pd.DataFrame(
        {
            "code": pd.Series(dtype="object"),
            "notice_date": pd.Series(dtype="object"),
            "period": pd.Series(dtype="object"),
            "type": pd.Series(dtype="object"),
            "amp": pd.Series(dtype="float64"),
            "name": pd.Series(dtype="object"),
        }
    )


def _load_period_cached(period: str, cache_dir: str, refresh: bool) -> pd.DataFrame:
    """带 parquet 落盘缓存的单期抓取。"""
    if not cache_dir:
        return fetch_yjyg_period(period)

    cache_path = os.path.join(cache_dir, f"yjyg_{period}.parquet")
    if not refresh and os.path.exists(cache_path):
        return pd.read_parquet(cache_path)

    raw = fetch_yjyg_period(period)
    os.makedirs(cache_dir, exist_ok=True)
    # 空表也落盘：避免对无数据的报告期反复发起网络请求
    raw.to_parquet(cache_path, index=False)
    return raw


def _normalize_period(raw: pd.DataFrame, period: str, indicator: str) -> pd.DataFrame:
    """把 akshare 单期原始表规范化为标准事件表。"""
    if raw.empty or "预测指标" not in raw.columns:
        return _empty_events()

    df = raw[raw["预测指标"] == indicator].copy()
    if df.empty:
        return _empty_events()

    out = pd.DataFrame()
    out["code"] = df["股票代码"].map(to_qmt_code)
    out["notice_date"] = df["公告日期"].map(_to_date_string)
    # akshare 内部构造了「报告日期」列却在最终选列时丢弃了它，此处补回
    out["period"] = period
    out["type"] = df["预告类型"]
    out["amp"] = pd.to_numeric(df["业绩变动幅度"], errors="coerce")
    out["name"] = df["股票简称"]

    # code 为 None 的是 B 股或未知前缀；缺公告日期或预告类型的事件无法定位/判别。
    # amp 允许为 NaN（实测约 0.2%），由策略侧的幅度判据兜底。
    out = out.dropna(subset=["code", "notice_date", "type"])
    return out.reset_index(drop=True)


def _to_date_string(value) -> Optional[str]:
    """公告日期归一为 'YYYYMMDD' 字符串。

    akshare 返回的是 ``datetime.date`` 对象。行情索引是 'YYYYMMDD' 字符串，
    统一成同一类型后，对齐环节可以直接用字符串比较（字典序即时间序），
    不必让 datetime 进入对齐逻辑。
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, str):
        digits = value.replace("-", "").replace("/", "").strip()
        return digits[:8] if len(digits) >= 8 else None
    try:
        return pd.Timestamp(value).strftime("%Y%m%d")
    except (ValueError, TypeError):
        return None
