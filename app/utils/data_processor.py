"""
策略开发数据预处理工具
"""

import pandas as pd
import numpy as np

class StrategyDataProcessor:
    """策略数据处理器"""
    
    @staticmethod
    def calculate_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        df = data.copy()
        
        # 价格指标
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(10).std()
        df['momentum'] = df['close'] / df['close'].shift(10) - 1
        
        # 成交量指标
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        df['volume_zscore'] = (df['volume'] - df['volume_ma']) / df['volume'].rolling(20).std()
        
        return df
    
    @staticmethod
    def detect_market_regime(data: pd.DataFrame) -> str:
        """检测市场状态"""
        volatility = data['returns'].std() * 100
        
        if volatility > 2.0:
            return "high_volatility"
        elif volatility < 0.5:
            return "low_volatility"
        else:
            return "normal"
