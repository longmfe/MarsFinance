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
    current_volume_ratio = df['volume_ratio'].iloc[-2]
    
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

def dynamic_threshold_adjustment(data, base_buy_threshold=1.2, base_sell_threshold=0.8):
    """基于波动率的动态阈值调整 - 立即实现"""
    if len(data) < 20:
        return base_buy_threshold, base_sell_threshold

    # 计算市场波动率
    volatility = data['close'].pct_change().std() * 100

    # 动态调整阈值
    if volatility > 2.0:  # 高波动
        return base_buy_threshold * 1.1, base_sell_threshold * 0.9
    elif volatility < 0.5:  # 低波动
        return base_buy_threshold * 0.9, base_sell_threshold * 1.1
    else:  # 正常波动
        return base_buy_threshold, base_sell_threshold

def volume_quality_filter(data, volume_period=20):
    """成交量质量过滤 - 立即实现"""
    df = data.copy()
    df['volume_ma'] = df['volume'].rolling(window=volume_period).mean()
    df['volume_std'] = df['volume'].rolling(window=volume_period).std()
    df['volume_zscore'] = (df['volume'] - df['volume_ma']) / df['volume_std']

    current_volume_z = df['volume_zscore'].iloc[-1]

    # 过滤异常成交量（避免因单日巨量产生错误信号）
    return abs(current_volume_z) <= 3  # 3倍标准差内的正常成交量

def price_momentum_confirmation(data, short_period=3, long_period=10):
    """价格动量确认 - 立即实现"""
    df = data.copy()
    df['momentum_short'] = df['close'] / df['close'].shift(short_period) - 1
    df['momentum_long'] = df['close'] / df['close'].shift(long_period) - 1

    current_momentum_short = df['momentum_short'].iloc[-1]
    current_momentum_long = df['momentum_long'].iloc[-1]

    # 动量确认规则
    if current_momentum_short > 0.01 and current_momentum_long > 0:
        return 1  # 加强买入信号
    elif current_momentum_short < -0.01 and current_momentum_long < 0:
        return -1  # 加强卖出信号
    else:
        return 0  # 中性

def optimized_volume_price_strategy(data, volume_period=20):
    """集成所有优化的最终版本 - 立即实现"""
    # 1. 成交量质量检查
    if not volume_quality_filter(data, volume_period):
        return 0

    # 2. 动态阈值调整
    buy_threshold, sell_threshold = dynamic_threshold_adjustment(data)

    # 3. 基础量价信号
    df = data.copy()
    df['price_change'] = df['close'].pct_change()
    df['volume_ma'] = df['volume'].rolling(window=volume_period).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma']

    current_price_change = df['price_change'].iloc[-1]
    current_volume_ratio = df['volume_ratio'].iloc[-1]

    # 4. 基础信号生成
    base_signal = 0
    if (current_price_change > 0 and current_volume_ratio > buy_threshold) or \
       (current_price_change < 0 and current_volume_ratio < sell_threshold):
        base_signal = 1
    elif (current_price_change < 0 and current_volume_ratio > buy_threshold) or \
         (current_price_change > 0 and current_volume_ratio < sell_threshold):
        base_signal = -1

    # 5. 动量确认
    momentum_confirmation = price_momentum_confirmation(data)

    # 6. 综合判断
    if base_signal != 0 and momentum_confirmation == base_signal:
        return base_signal  # 量价信号和动量确认一致，加强信号
    elif base_signal != 0 and momentum_confirmation == 0:
        return base_signal  # 量价信号有效，动量中性，使用原信号
    else:
        return 0  # 信号冲突或无效，不交易
