"""
MarsFinance - 量化交易研究平台
"""

__version__ = "0.1.0"
__author__ = "黄隆"
__email__ = "longmfe@163.com"

from .data_loader import DataLoader
from .backtest import PortfolioBacktest, StockBacktest
from .strategies import EnhancedVolumePriceStrategy
from .app.core.signal_generator import VolumePriceSignalGenerator
from .app.pipelines.strategy_pipeline import StrategyPipeline
from .app.utils.data_processor import StrategyDataProcessor

__all__ = [
    'DataLoader',
    'PortfolioBacktest', 
    'StockBacktest',
    'EnhancedVolumePriceStrategy',
    'VolumePriceSignalGenerator',
    'StrategyPipeline',
    'StrategyDataProcessor'
]

