# -*- coding: utf-8 -*-
"""research —— 券商金工研报与学术论文的可复现实现。

本包与仓库其余部分（已归档的 QMT 量价研究）相互独立，包含三项复现：

- ``research.ffscore``    华泰《价值选股之 FFScore 模型》—— Piotroski F-Score A 股实证
- ``research.cscv``       Bailey/Borwein/López de Prado (2016) 回测过拟合概率 PBO
- ``research.allocation`` 风险预算 + 机器学习资产配置

共享地基位于 ``research.datafeed``（akshare 数据源、磁盘缓存、时点对齐）
与 ``research.metrics``（绩效指标）。

本模块刻意不做子包的重导入：akshare / lightgbm / cvxpy 均为可选重依赖，
应在真正用到时才被导入。
"""

__version__ = "0.1.0"
