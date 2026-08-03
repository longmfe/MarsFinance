# -*- coding: utf-8 -*-
"""ffscore —— 华泰《价值选股之 FFScore 模型》/ Piotroski F-Score A 股复现。

参考：华泰证券金融工程，2017-02-09，《华泰价值选股之 FFScore 模型 ——
比乔斯基选股模型 A 股实证研究》；原始文献 Piotroski, J. D. (2000),
*Value Investing: The Use of Historical Financial Statement Information to
Separate Winners from Losers*, Journal of Accounting Research.

我们没有研报原文，因此每一个"研报未披露"的方法论选择都在
``research.ffscore.config`` 里以具名参数出现，并由 CSCV 网格统一扫描 ——
把信息缺口转化为可发布的敏感性分析，而不是一次静默的猜测。
"""

from research.ffscore.config import FFSCORE_DEFAULTS, FFScoreConfig
from research.ffscore.score import compute_ffscore, compute_signals
from research.ffscore.signals import SIGNAL_FUNCTIONS, SIGNAL_NAMES

__all__ = [
    "FFSCORE_DEFAULTS",
    "FFScoreConfig",
    "compute_signals",
    "compute_ffscore",
    "SIGNAL_FUNCTIONS",
    "SIGNAL_NAMES",
]
