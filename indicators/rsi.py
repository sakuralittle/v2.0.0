import backtrader as bt
import numpy as np


class RSI(bt.Indicator):
    """相對強度指數 (Relative Strength Index)"""
    lines = ('rsi',)
    params = (('period', 14),)
    
    def __init__(self):
        # 使用 backtrader 內建的 RSI
        self.lines.rsi = bt.indicators.RSI(self.data.close, period=self.p.period)


class RSIIndicator(bt.Indicator):
    """RSI 自訂指標 - 用於 backtrader"""
    lines = ('rsi', 'signal')
    params = (
        ('period', 14),
        ('overbought_level', 70),
        ('oversold_level', 30),
    )
    
    def __init__(self):
        # 計算 RSI
        self.lines.rsi = bt.indicators.RSI(self.data.close, period=self.p.period)
        
        # 信號: RSI > overbought 為 1 (超買), RSI < oversold 為 -1 (超賣), 否則為 0 (中性)
        self.lines.signal = bt.If(
            self.lines.rsi > self.p.overbought_level,
            1,
            bt.If(self.lines.rsi < self.p.oversold_level, -1, 0)
        )
        
        
