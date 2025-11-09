import backtrader as bt
import pandas as pd
import numpy as np
import os
from indicator import IndicatorFactory
from agent import create_ddpg_agent
from config import DDPG_CONFIG, DATA_CONFIG

class strategy1(bt.Strategy):
    def __init__(self):
        # 初始化 DDPG 代理指標
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
    DDPG 代理指標 - 完整實現
    
    功能：
    1. 初始化 DDPG 代理
    2. 構建狀態向量
    3. 使用代理選擇動作
    4. 計算獎勵並訓練
    5. 輸出交易信號
    """
    lines = ('agent_signal',)
    params = (
        ('model_config', DDPG_CONFIG),
    )
    def __init__(self):
        # 初始化技術指標
        self.hullma = IndicatorFactory.hullma_indicator(self.data)
        
        self.state_dim = self.params.model_config['state_dim']
        self.agent = create_ddpg_agent(
            state_dim=self.state_dim,
            action_dim=1,
            learning_rate=self.params.model_config['learning_rate'],
            gamma=self.params.model_config['gamma'],
            tau=self.params.model_config['tau'],
            use_lstm=self.params.model_config['use_lstm'],
            lstm_hidden_dim=self.params.model_config['lstm_hidden_dim']
        )
        
        # ========== 模型加載邏輯 ==========
        model_path = self.params.model_config['model_path']
        if os.path.exists(model_path):
            # 模型存在，加載模型
            print(f"加載已存在的模型: {model_path}")
            self.agent.load(model_path)
            self.model_loaded = True
        else:
            # 模型不存在，新建模型
            print(f"模型不存在，建立新模型: {model_path}")
            self.model_loaded = False
        
        self.model_path = model_path
        
        # 用於存儲上一個狀態、動作、價格（計算獎勵時需要）
        self.prev_state = None
        self.prev_action = None
        self.prev_price = None
        

    def _build_state(self):
        """構建狀態向量"""
        current_price = self.data.close[0]
        
        # HMA 的三個輸出
        mhull = self.hullma.mhull[0]
        shull = self.hullma.shull[0]
        signal = self.hullma.signal[0]
        
        # 前 n 根 K 線的各自變化率（追加 n 個特徵）
        period = self.p.model_config['price_change_period']
        price_changes = []
        
        for i in range(1, period + 1):
            if len(self) > i:
                # 計算第 i 根 K 線相對於第 i+1 根的變化率
                price_change = (self.data.close[-i+1] - self.data.close[-i]) / self.data.close[-i]
            else:
                price_change = 0
            price_changes.append(price_change)
        
        # 成交量正規化
        volume = self.data.volume[0] / 1000000
        
        # 組合狀態向量
        state_list = [
            mhull,
            shull,
            signal,
            volume,
            current_price
        ]
        
        # 追加 n 個價格變化率特徵
        state_list.extend(price_changes)
        
        return np.array(state_list, dtype=np.float32), current_price

    
    def next(self):
        """每根 K 線執行一次"""
        current_state, current_price = self._build_state()
        action = self.agent.select_action(current_state, training=self.p.model_config['enable_training'])
        
        self._train_step(current_price, current_state)
        self._update_state(current_state, action, current_price)
        
        # 提取 action 的標量值（action 是 numpy array）
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

    print(f'Initial Portfolio Value: {cerebro.broker.getvalue():.2f}')
    cerebro.run()
    print(f'Final Portfolio Value: {cerebro.broker.getvalue():.2f}')
    
    # ========== 保存模型 ==========
    # 從策略中取得 agent_indicator 實例
    strat = cerebro.runstrats[0][0]
    agent_ind = strat.agent_ind
    model_path = agent_ind.model_path
    
    print(f"\n保存模型到: {model_path}")
    agent_ind.agent.save(model_path)
    print("模型保存完成!")

    cerebro.plot()