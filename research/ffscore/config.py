# -*- coding: utf-8 -*-
"""FFScore 的全部可调参数 —— 尤其是研报未披露、由本实现自行决定的部分。

我们没有华泰研报原文。与其对每个细节做一次静默的猜测，不如把每个选择都变成
一个具名参数、写清默认值与理由，再由 ``research.cscv`` 的参数网格统一扫描。
这样"不知道研报怎么做的"就从一个隐患变成了一份敏感性分析。

标注 ``研报未披露`` 的条目就是那些需要读者自行判断的地方。
"""

from dataclasses import asdict, dataclass
from typing import Dict


@dataclass
class FFScoreConfig:
    """FFScore 复现的参数集合。

    仓库其余部分不用 dataclass，这里用是因为参数要被 CSCV 网格成百上千次地
    构造与比较，需要 ``asdict`` 与相等性 —— 手搓 dict 反而更容易出错。
    """

    # --- 信号口径 ---

    #: ROA / 周转率的资产分母。Piotroski 原文用**期初**总资产；均值总资产是
    #: 常见变体。``"beginning"`` | ``"average"``
    ta_basis: str = "beginning"

    #: F5（ΔLEVER）的杠杆口径。**研报未披露**，且批量财报接口没有长期负债拆分。
    #: ``"debt_ratio"``（默认，Δ资产负债率 < 0）为代理口径；
    #: ``"noncurrent"`` 用 Δ(非流动负债/总资产)，需逐股明细数据。
    lever_definition: str = "debt_ratio"

    #: F7（EQ_OFFER）允许的股本增幅。**无法区分增发与送转股**，容忍度用来
    #: 消化小幅股本变动。0.0 表示任何增发都判 0。
    eq_offer_tolerance: float = 0.0

    #: 至少要有多少个信号可算，否则该股当期不参与。9 表示要求信号完整。
    min_signals: int = 8

    # --- 股票池 ---

    #: 账面市值比最高的分位（Piotroski 取最高五分位 = 0.2）。**研报未披露**
    #: 其具体分位，1.0 表示不做价值筛选。
    bm_quantile: float = 0.2

    exclude_st: bool = True

    #: 金融股的流动/非流动分类在银行报表中不适用（实测银行的流动资产为空），
    #: 因此排除金融股既是方法论要求也是数据必需。
    exclude_financials: bool = True

    min_listing_days: int = 365
    require_positive_equity: bool = True
    exclude_suspended: bool = True

    # --- 组合构建 ---

    #: 调仓节奏。**研报未披露**。``"annual_may"``（默认）在每年 5 月首个交易日
    #: 调仓，对应年报 4/30 的法定披露截止；``"cn_report"`` 为 5/9/11 月三次；
    #: ``"M"`` / ``"Q"`` 为月度 / 季度。
    rebalance: str = "annual_may"

    #: 分组方式。F-Score 是 0-9 的整数、并列极多，``rank(pct=True)`` 分十档会
    #: 把档位边界切进并列块内部，归属变成任意的。``"score_value"``（默认）按
    #: 分值分桶，与 Piotroski 一致；``"quantile"`` 留给连续型变体。
    group_by: str = "score_value"

    n_groups: int = 10

    #: 多头切点。**研报未披露**是 F>=8 还是取最高一档。
    long_threshold: int = 8
    short_threshold: int = 1

    weighting: str = "equal"

    # --- 交易摩擦（沿用仓库 backtest/stock_backtest.py 的量级） ---

    commission: float = 0.001
    slippage: float = 0.001

    #: 印花税（卖出单边）。仓库既有引擎漏了这一项，新引擎显式计入。
    stamp_tax: float = 0.001

    #: 信号日与成交日的间隔（交易日）。1 表示次日成交，与仓库既有引擎
    #: "看 T-1 数据、按 T 价格成交"的结构性前视隔离一致。
    execution_lag: int = 1

    # --- 频率 ---

    #: ``"annual"``（默认，Piotroski 原始定义，且年报 Q4 累计即全年，
    #: 无需累计差分）或 ``"ttm"``。
    frequency: str = "annual"

    benchmark: str = "sh000300"

    def to_dict(self) -> Dict:
        return asdict(self)

    def screen_rules(self) -> Dict:
        """抽出 ``datafeed.fundamentals.screen_universe`` 需要的规则子集。"""
        return {
            "exclude_st": self.exclude_st,
            "exclude_financials": self.exclude_financials,
            "min_listing_days": self.min_listing_days,
            "require_positive_equity": self.require_positive_equity,
            "exclude_suspended": self.exclude_suspended,
        }

    def validate(self) -> None:
        """校验参数取值，尽早给出可读的报错。"""
        if self.ta_basis not in ("beginning", "average"):
            raise ValueError(
                f"ta_basis 只能是 beginning / average，得到 {self.ta_basis!r}"
            )
        if self.lever_definition not in ("debt_ratio", "noncurrent"):
            raise ValueError(
                f"lever_definition 只能是 debt_ratio / noncurrent，"
                f"得到 {self.lever_definition!r}"
            )
        if self.group_by not in ("score_value", "quantile"):
            raise ValueError(
                f"group_by 只能是 score_value / quantile，得到 {self.group_by!r}"
            )
        if self.frequency not in ("annual", "ttm"):
            raise ValueError(f"frequency 只能是 annual / ttm，得到 {self.frequency!r}")
        if not 0 < self.bm_quantile <= 1.0:
            raise ValueError(f"bm_quantile 必须落在 (0, 1]，得到 {self.bm_quantile}")
        if not 0 <= self.min_signals <= 9:
            raise ValueError(f"min_signals 必须落在 [0, 9]，得到 {self.min_signals}")


#: 默认参数字典，由 dataclass 派生 —— 避免两处各写一遍导致漂移。
FFSCORE_DEFAULTS: Dict = asdict(FFScoreConfig())


def default_config(**overrides) -> FFScoreConfig:
    """构造配置，允许逐项覆盖默认值。"""
    config = FFScoreConfig(**{**FFSCORE_DEFAULTS, **overrides})
    config.validate()
    return config
