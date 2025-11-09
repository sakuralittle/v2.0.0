import backtrader as bt
import pandas as pd
import numpy as np
import os
from indicator import IndicatorFactory
from agent import create_sac_agent
from config import SAC_CONFIG, DATA_CONFIG
from collections import deque

class strategy1(bt.Strategy):
    def __init__(self):
        # 初始化 SAC 代理指標
        self.agent_ind = agent_indicator(self.data)
    
    def next(self):
        # 從 agent_indicator 獲取動作信號
        action = self.agent_ind.lines.agent_signal[0]
        
        # 根據代理的動作進行交易決策
        if not self.position:
            # 沒有持倉時
            if action > 0.5:  # 買入信號
                self.buy()
        else:
            # 已有持倉時
            if action < -0.5:  # 賣出信號
                self.close()

class agent_indicator(bt.Indicator):
    """
    SAC 代理指標 - 僅推理
    
    功能：
    1. 加載預訓練的 SAC 模型
    2. 構建狀態向量（HullMA, RSI, CCI, 價格變化等）
    3. 使用代理選擇動作
    4. 輸出交易信號
    """
    lines = ('agent_signal',)
    params = (
        ('model_config', SAC_CONFIG),
    )
    def __init__(self):
        # 初始化技術指標
        self.hullma = IndicatorFactory.hullma_indicator(self.data)
        self.rsi = IndicatorFactory.rsi(
            self.data, 
            period=self.params.model_config['rsi_period']
        )
        self.cci = IndicatorFactory.cci(
            self.data, 
            period=self.params.model_config['cci_period']
        )
        
        self.state_dim = self.params.model_config['state_dim']
        
        # 創建 SAC 代理
        self.agent = create_sac_agent(
            state_dim=self.state_dim,
            action_dim=1,
            learning_rate=self.params.model_config['learning_rate'],
            gamma=self.params.model_config['gamma'],
            tau=self.params.model_config['tau'],
            alpha=self.params.model_config['alpha']
        )
        
        # ========== Z-score 正規化用的滾動窗口 ==========
        self.normalization_window = self.params.model_config.get('normalization_window', 100)
        self.state_history = deque(maxlen=self.normalization_window)
        
        # ========== 加載預訓練模型 ==========
        model_path = self.params.model_config['model_path']
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}，請先訓練模型")
        
        print(f"加載 SAC 模型: {model_path}")
        self.agent.load(model_path)
        self.model_path = model_path
        print("模型加載完成！")
        

    def _build_state(self):
        """構建狀態向量（未正規化）"""
        current_price = self.data.close[0]
        
        # HMA 的三個輸出
        mhull = self.hullma.mhull[0]
        shull = self.hullma.shull[0]
        signal = self.hullma.signal[0]
        
        # RSI 和 CCI
        rsi = self.rsi.rsi[0]
        cci = self.cci.cci[0]
        
        # 前 n 根 K 線的各自變化率
        period = self.p.model_config['price_change_period']
        price_changes = []
        
        for i in range(1, period + 1):
            if len(self) > i:
                price_change = (self.data.close[-i+1] - self.data.close[-i]) / self.data.close[-i]
            else:
                price_change = 0
            price_changes.append(price_change)
        
        # 成交量
        volume = self.data.volume[0]
        
        # 組合狀態向量：mhull + shull + signal + volume + current_price + price_changes + rsi + cci
        state_list = [
            mhull,
            shull,
            signal,
            volume,
            current_price
        ]
        state_list.extend(price_changes)
        state_list.extend([rsi, cci])
        
        return np.array(state_list, dtype=np.float32)
    
    def _normalize_state(self, state):
        """使用 Z-score 正規化狀態向量"""
        # 添加當前狀態到歷史記錄
        self.state_history.append(state.copy())
        
        # 如果歷史數據不足，返回原始狀態（或全零）
        if len(self.state_history) < 2:
            return np.zeros_like(state)
        
        # 計算歷史數據的 mean 和 std
        history_array = np.array(self.state_history)
        mean = np.mean(history_array, axis=0)
        std = np.std(history_array, axis=0)
        
        # 避免除以零，對於 std 為 0 的特徵，保持原值
        std = np.where(std < 1e-8, 1.0, std)
        
        # Z-score 正規化
        normalized_state = (state - mean) / std
        
        return normalized_state

    def next(self):
        """每根 K 線執行一次"""
        # 構建原始狀態
        raw_state = self._build_state()
        
        # Z-score 正規化
        normalized_state = self._normalize_state(raw_state)
        
        # 使用正規化後的狀態選擇動作
        action = self.agent.select_action(normalized_state)
        
        # 輸出動作信號
        self.lines.agent_signal[0] = action[0]
        



if __name__ == '__main__':
    cerebro = bt.Cerebro()
    cerebro.addstrategy(strategy1)

    df = pd.read_csv('h:/program_dev/trading_system/v1.0.0/data/data_15m.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

    data = bt.feeds.PandasData(
        dataname=df,
        datetime=None,
        open='open',
        high='high',
        low='low',
        close='close',
        volume='volume',
        openinterest=-1
    )
    cerebro.adddata(data)

    cerebro.broker.setcash(100000.0)
    cerebro.broker.setcommission(commission=0.001)

    print(f'初始資金: {cerebro.broker.getvalue():.2f}')
    cerebro.run()
    print(f'最終資金: {cerebro.broker.getvalue():.2f}')
    
    cerebro.plot()