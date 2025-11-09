import backtrader as bt
import numpy as np


class CCI(bt.Indicator):
    """商品通道指數 (Commodity Channel Index)"""
    lines = ('cci',)
    params = (('period', 20),)
    
    def __init__(self):
        # 使用 backtrader 內建的 CCI
        self.lines.cci = bt.indicators.CCI(self.data, period=self.p.period)


class CCIIndicator(bt.Indicator):
    """CCI 自訂指標 - 用於 backtrader"""
    lines = ('cci', 'signal')
    params = (
        ('period', 20),
        ('overbought_level', 100),
        ('oversold_level', -100),
    )
    
    def __init__(self):
        # 計算 CCI
        self.lines.cci = bt.indicators.CCI(self.data, period=self.p.period)
        
        # 信號: CCI > overbought 為 1 (超買), CCI < oversold 為 -1 (超賣), 否則為 0 (中性)
        self.lines.signal = bt.If(
            self.lines.cci > self.p.overbought_level,
            1,
            bt.If(self.lines.cci < self.p.oversold_level, -1, 0)
        )
        
        
