import backtrader as bt
import numpy as np


class WMA(bt.Indicator):
    """加權移動平均 (Weighted Moving Average)"""
    lines = ('wma',)
    params = (('period', 10),)
    
    def __init__(self):
        # 使用 backtrader 內建的 WMA
        self.lines.wma = bt.indicators.WMA(self.data.close, period=self.p.period)


class HMA(bt.Indicator):
    """Hull Moving Average"""
    lines = ('hma',)
    params = (('period', 55),)
    
    def __init__(self):
        half_period = int(self.p.period / 2)
        sqrt_period = int(np.sqrt(self.p.period))
        
        # 使用 self.data 而不是 self.data.close
        wma1 = bt.indicators.WMA(self.data, period=half_period)
        wma2 = bt.indicators.WMA(self.data, period=self.p.period)
        wma_diff = 2 * wma1 - wma2
        
        self.lines.hma = bt.indicators.WMA(wma_diff, period=sqrt_period)


class EHMA(bt.Indicator):
    """Exponential Hull Moving Average"""
    lines = ('ehma',)
    params = (('period', 55),)
    
    def __init__(self):
        half_period = int(self.p.period / 2)
        sqrt_period = int(np.sqrt(self.p.period))
        
        ema1 = bt.indicators.EMA(self.data, period=half_period)
        ema2 = bt.indicators.EMA(self.data, period=self.p.period)
        ema_diff = 2 * ema1 - ema2
        
        self.lines.ehma = bt.indicators.WMA(ema_diff, period=sqrt_period)


class THMA(bt.Indicator):
    """Triple Hull Moving Average"""
    lines = ('thma',)
    params = (('period', 55),)
    
    def __init__(self):
        third_period = int(self.p.period / 3)
        half_period = int(self.p.period / 2)
        
        wma1 = bt.indicators.WMA(self.data, period=third_period)
        wma2 = bt.indicators.WMA(self.data, period=half_period)
        wma3 = bt.indicators.WMA(self.data, period=self.p.period)
        wma_diff = wma1 * 3 - wma2 - wma3
        
        self.lines.thma = bt.indicators.WMA(wma_diff, period=self.p.period)


class HullMAIndicator(bt.Indicator):
    """Hull MA 自訂指標 - 用於 backtrader"""
    lines = ('mhull', 'shull', 'signal')
    params = (
        ('period', 55),
        ('mode', 'hma'),  # 'hma', 'ehma', 'thma'
        ('length_mult', 1.0),
    )
    
    def __init__(self):
        adjusted_period = int(self.p.period * self.p.length_mult)
        
        # 選擇模式
        if self.p.mode == 'hma':
            hull = HMA(self.data, period=adjusted_period)
            hull_line = hull.hma
        elif self.p.mode == 'ehma':
            hull = EHMA(self.data, period=adjusted_period)
            hull_line = hull.ehma
        elif self.p.mode == 'thma':
            hull = THMA(self.data, period=int(adjusted_period / 2))
            hull_line = hull.thma
        else:
            raise ValueError(f"未知的模式: {self.p.mode}")
        
        # MHULL 為當前 Hull MA
        self.lines.mhull = hull_line
        
        # SHULL 為往後移動 2 個週期的 Hull MA
        self.lines.shull = hull_line(-2)
        
        # 信號: MHULL > SHULL 為 1 (上升), 否則為 -1 (下降)
        self.lines.signal = bt.If(self.lines.mhull > self.lines.shull, 1, -1)