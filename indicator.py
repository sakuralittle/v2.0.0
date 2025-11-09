"""
整合指標模組 - 用於 backtrader 策略
提供各種技術指標供其他程式呼叫
"""

import backtrader as bt
import numpy as np

# 導入所有指標
from indicators.hullma import WMA, HMA, EHMA, THMA, HullMAIndicator
from indicators.rsi import RSI, RSIIndicator
from indicators.cci import CCI, CCIIndicator

__all__ = [
    'WMA',
    'HMA', 
    'EHMA',
    'THMA',
    'HullMAIndicator',
    'RSI',
    'RSIIndicator',
    'CCI',
    'CCIIndicator',
]


# 便利函數：直接在策略中使用
class IndicatorFactory:
    """指標工廠 - 提供便利的指標建立方法"""
    
    @staticmethod
    def  wma(data, period=10):
        """建立加權移動平均"""
        return WMA(data, period=period)
    
    @staticmethod
    def  hma(data, period=55):
        """建立 Hull Moving Average"""
        return HMA(data, period=period)
    
    @staticmethod
    def  ehma(data, period=55):
        """建立 Exponential Hull Moving Average"""
        return EHMA(data, period=period)
    
    @staticmethod
    def  thma(data, period=55):
        """建立 Triple Hull Moving Average"""
        return THMA(data, period=period)
    
    @staticmethod
    def  hullma_indicator(data, period=55, mode='hma', length_mult=1.0):
        """建立 Hull MA 自訂指標"""
        return HullMAIndicator(data, period=period, mode=mode, length_mult=length_mult)
    
    @staticmethod
    def  rsi(data, period=14):
        """建立相對強度指數"""
        return RSI(data, period=period)
    
    @staticmethod
    def  rsi_indicator(data, period=14, overbought_level=70, oversold_level=30):
        """建立 RSI 自訂指標"""
        return RSIIndicator(data, period=period, overbought_level=overbought_level, oversold_level=oversold_level)
    
    @staticmethod
    def  cci(data, period=20):
        """建立商品通道指數"""
        return CCI(data, period=period)
    
    @staticmethod
    def  cci_indicator(data, period=20, overbought_level=100, oversold_level=-100):
        """建立 CCI 自訂指標"""
        return CCIIndicator(data, period=period, overbought_level=overbought_level, oversold_level=oversold_level)
