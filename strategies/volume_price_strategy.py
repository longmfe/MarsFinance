import pandas as pd
from typing import Union

def enhanced_volume_price_strategy(data: pd.DataFrame, volume_period: int = 20) -> int:
    """
    增强版量价策略
    
    Args:
        data: 包含价格和成交量数据的DataFrame
        volume_period: 成交量移动平均周期
        
    Returns:
        交易信号: 1(买入), -1(卖出), 0(持有)
    """
    if len(data) < volume_period:
        return 0
    
    df = data.copy()
    
    # 计算价格变化和成交量比率
    df['price_change'] = df['close'].pct_change()
    df['volume_ma'] = df['volume'].rolling(window=volume_period).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma']
    
    current_price_change = df['price_change'].iloc[-1]
    current_volume_ratio = df['volume_ratio'].iloc[-1]
    
    # 买入信号
    if (current_price_change > 0 and current_volume_ratio > 1.2) or \
       (current_price_change < 0 and current_volume_ratio < 0.8):
        return 1
    
    # 卖出信号  
    elif (current_price_change < 0 and current_volume_ratio > 1.2) or \
         (current_price_change > 0 and current_volume_ratio < 0.8):
        return -1
    
    else:
        return 0
