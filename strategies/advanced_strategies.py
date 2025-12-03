import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

from xtquant import xtdata
import datetime
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class AdvancedStrategies:
    """高级策略库"""
    
    @staticmethod
    def dual_moving_average_cross(data, short_window=5, long_window=20):
        """双均线交叉策略"""
        if len(data) < long_window:
            return 0
        
        df = data.copy()
        df['short_ma'] = df['close'].rolling(window=short_window).mean()
        df['long_ma'] = df['close'].rolling(window=long_window).mean()
        
        current_short = df['short_ma'].iloc[-1]
        current_long = df['long_ma'].iloc[-1]
        
        if len(df) > 1:
            prev_short = df['short_ma'].iloc[-2]
            prev_long = df['long_ma'].iloc[-2]
        else:
            return 0
        
        # 金叉买入，死叉卖出
        if prev_short <= prev_long and current_short > current_long:
            return 1
        elif prev_short >= prev_long and current_short < current_long:
            return -1
        else:
            return 0
    
    @staticmethod
    def rsi_mean_reversion(data, period=14, oversold=30, overbought=70):
        """RSI均值回归策略"""
        if len(data) < period + 1:
            return 0
        
        df = data.copy()
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        current_rsi = rsi.iloc[-1]
        
        if current_rsi < oversold:
            return 1  # 超卖，买入
        elif current_rsi > overbought:
            return -1  # 超买，卖出
        else:
            return 0
    
    @staticmethod
    def bollinger_breakout(data, period=20, num_std=2):
        """布林带突破策略"""
        if len(data) < period:
            return 0
        
        df = data.copy()
        df['middle_band'] = df['close'].rolling(window=period).mean()
        df['std'] = df['close'].rolling(window=period).std()
        df['upper_band'] = df['middle_band'] + (df['std'] * num_std)
        df['lower_band'] = df['middle_band'] - (df['std'] * num_std)
        
        current_close = df['close'].iloc[-1]
        current_upper = df['upper_band'].iloc[-1]
        current_lower = df['lower_band'].iloc[-1]
        prev_close = df['close'].iloc[-2] if len(df) > 1 else current_close
        
        # 上突破买入，下突破卖出
        if prev_close <= current_upper and current_close > current_upper:
            return 1
        elif prev_close >= current_lower and current_close < current_lower:
            return -1
        else:
            return 0
    
    @staticmethod
    def macd_crossover(data, fast_period=12, slow_period=26, signal_period=9):
        """MACD交叉策略"""
        if len(data) < slow_period + signal_period:
            return 0
        
        df = data.copy()
        exp1 = df['close'].ewm(span=fast_period, adjust=False).mean()
        exp2 = df['close'].ewm(span=slow_period, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=signal_period, adjust=False).mean()
        histogram = macd - signal
        
        current_macd = macd.iloc[-1]
        current_signal = signal.iloc[-1]
        prev_macd = macd.iloc[-2] if len(macd) > 1 else current_macd
        prev_signal = signal.iloc[-2] if len(signal) > 1 else current_signal
        
        # MACD上穿信号线买入，下穿信号线卖出
        if prev_macd <= prev_signal and current_macd > current_signal:
            return 1
        elif prev_macd >= prev_signal and current_macd < current_signal:
            return -1
        else:
            return 0
    
    @staticmethod
    def volume_price_confirmation(data, volume_period=20):
        """量价确认策略"""
        if len(data) < volume_period:
            return 0
        
        df = data.copy()
        df['price_change'] = df['close'].pct_change()
        df['volume_ma'] = df['volume'].rolling(window=volume_period).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        current_price_change = df['price_change'].iloc[-1]
        current_volume_ratio = df['volume_ratio'].iloc[-2]
        prev_price_change = df['price_change'].iloc[-2] if len(df) > 1 else 0
        
        # 买入信号
        # 价涨量增买入，价跌量跌买入
        if (current_price_change > 0 and current_volume_ratio > 3) or \
           (current_price_change < 0 and current_volume_ratio < 0.5):
            return 1
        
        # 卖出信号  
        # 价涨量跌卖出，价跌量增卖出
        elif (current_price_change < 0 and current_volume_ratio > 3) or \
             (current_price_change > 0 and current_volume_ratio < 0.5):
            return -1
        
        else:
            return 0

    @staticmethod
    def mean_reversion(data, lookback=20, z_threshold=2):
        """均值回归策略（Z-score）"""
        if len(data) < lookback:
            return 0
        
        df = data.copy()
        returns = df['close'].pct_change().dropna()
        
        if len(returns) < lookback:
            return 0
        
        current_return = returns.iloc[-1]
        mean_return = returns.tail(lookback).mean()
        std_return = returns.tail(lookback).std()
        
        if std_return == 0:
            return 0
        
        z_score = (current_return - mean_return) / std_return
        
        # Z-score极端值回归
        if z_score < -z_threshold:
            return 1  # 超卖回归
        elif z_score > z_threshold:
            return -1  # 超买回归
        else:
            return 0
    
    @staticmethod
    def momentum_strategy(data, momentum_period=10, ma_period=20):
        """动量策略"""
        if len(data) < max(momentum_period, ma_period):
            return 0
        
        df = data.copy()
        df['momentum'] = df['close'] / df['close'].shift(momentum_period) - 1
        df['ma'] = df['close'].rolling(window=ma_period).mean()
        
        current_momentum = df['momentum'].iloc[-1]
        current_close = df['close'].iloc[-1]
        current_ma = df['ma'].iloc[-1]
        
        # 动量强劲且价格在均线上方买入
        if current_momentum > 0.02 and current_close > current_ma:
            return 1
        # 负动量且价格在均线下方卖出
        elif current_momentum < -0.02 and current_close < current_ma:
            return -1
        else:
            return 0

