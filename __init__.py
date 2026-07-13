# -*- coding: utf-8 -*-
"""MarsFinance - 量化交易研究框架"""

__version__ = "0.1.0"
__author__ = "黄隆"
__email__ = "longmfe@163.com"

from .app.core.signal_generator import VolumePriceSignalGenerator
from .app.utils.data_processor import StrategyDataProcessor
from .backtest import PortfolioBacktest, StockBacktest
from .data_loader import DataLoader
from .strategies import (
    enhanced_volume_price_strategy,
    optimized_volume_price_strategy,
)

__all__ = [
    "DataLoader",
    "PortfolioBacktest",
    "StockBacktest",
    "enhanced_volume_price_strategy",
    "optimized_volume_price_strategy",
    "VolumePriceSignalGenerator",
    "StrategyDataProcessor",
]
