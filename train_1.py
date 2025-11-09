import backtrader as bt
import pandas as pd
import numpy as np
import os
import time
from datetime import datetime
from indicator import IndicatorFactory
from agent import create_sac_agent, ReplayBuffer
from config import SAC_CONFIG, DATA_CONFIG, BROKER_CONFIG, TRAIN_CONFIG
from collections import deque

class TrainingStrategy(bt.Strategy):
    """訓練用策略 - 與 train_agent_indicator 互動"""
    params = (
        ('trainer', None),  # train_agent_indicator 實例
        ('agent', None),     # SAC agent 實例
        ('train_mode', True),
    )
    
    def __init__(self):
        # 初始化技術指標（與 agent_indicator 相同）
        self.hullma = IndicatorFactory.hullma_indicator(self.data)
        self.rsi = IndicatorFactory.rsi(self.data, period=SAC_CONFIG['rsi_period'])
        self.cci = IndicatorFactory.cci(self.data, period=SAC_CONFIG['cci_period'])
        
        # 狀態正規化
        self.normalization_window = SAC_CONFIG.get('normalization_window', 100)
        self.state_history = deque(maxlen=self.normalization_window)
        
        # 經驗收集
        self.current_state = None
        self.current_action = None
        self.step_count = 0
        self.episode_reward = 0
    
    def _build_state(self):
        """構建狀態向量（與 agent_indicator 相同）"""
        current_price = self.data.close[0]
        mhull = self.hullma.mhull[0]
        shull = self.hullma.shull[0]
        signal = self.hullma.signal[0]
        rsi = self.rsi.rsi[0]
        cci = self.cci.cci[0]
        
        period = SAC_CONFIG['price_change_period']
        price_changes = []
        for i in range(1, period + 1):
            if len(self) > i:
                price_change = (self.data.close[-i+1] - self.data.close[-i]) / self.data.close[-i]
            else:
                price_change = 0
            price_changes.append(price_change)
        
        volume = self.data.volume[0]
        state_list = [mhull, shull, signal, volume, current_price]
        state_list.extend(price_changes)
        state_list.extend([rsi, cci])
        
        return np.array(state_list, dtype=np.float32)
    
    def _normalize_state(self, state):
        """Z-score 正規化"""
        self.state_history.append(state.copy())
        if len(self.state_history) < 2:
            return np.zeros_like(state)
        
        history_array = np.array(self.state_history)
        mean = np.mean(history_array, axis=0)
        std = np.std(history_array, axis=0)
        std = np.where(std < 1e-8, 1.0, std)
        normalized_state = (state - mean) / std
        return normalized_state
    
    def next(self):
        """每根 K 線執行一次"""
        # 構建並正規化狀態
        raw_state = self._build_state()
        state = self._normalize_state(raw_state)
        
        # 如果有前一個狀態，計算 reward 並存儲經驗
        if self.current_state is not None and self.p.trainer is not None:
            reward = self.p.trainer.calculate_reward()
            self.episode_reward += reward
            
            # 存儲經驗 (s, a, r, s', done=False)
            self.p.trainer.store_experience(
                self.current_state, 
                self.current_action, 
                reward, 
                state, 
                False
            )
            
            # 訓練 agent
            if self.step_count >= TRAIN_CONFIG['start_steps']:
                if self.step_count % TRAIN_CONFIG['update_frequency'] == 0:
                    self.p.trainer.train_agent(self.p.agent)
        
        # 使用 agent 選擇動作
        action = self.p.agent.select_action(state, evaluate=False)
        
        # 執行交易決策
        if not self.position:
            if action[0] > 0.5:  # 買入信號
                self.buy()
        else:
            if action[0] < -0.5:  # 賣出信號
                self.close()
        
        # 保存當前狀態和動作
        self.current_state = state
        self.current_action = action
        self.step_count += 1
    
    def stop(self):
        """Episode 結束時調用"""
        if self.p.trainer is not None:
            # 記錄最終 reward
            final_reward = self.p.trainer.calculate_reward()
            self.episode_reward += final_reward
            
            # 存儲最後的經驗 (done=True)
            if self.current_state is not None:
                raw_state = self._build_state()
                next_state = self._normalize_state(raw_state)
                self.p.trainer.store_experience(
                    self.current_state,
                    self.current_action,
                    final_reward,
                    next_state,
                    True
                )
            
            # 記錄 episode 統計
            self.p.trainer.episode_rewards.append(self.episode_reward)
            portfolio_return = self.p.trainer.get_portfolio_return()
            self.p.trainer.episode_returns.append(portfolio_return)


class strategy1(bt.Strategy):
    """推理用策略 - 使用預訓練模型"""
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

class train_agent_indicator():
    """
    SAC 代理指標訓練類別
    
    功能：
    1. 訓練 SAC 模型
    2. 計算 reward function
    3. 管理訓練過程
    """
    
    def __init__(self, cerebro, initial_cash=None):
        """
        初始化訓練環境
        
        Args:
            cerebro: backtrader 的 Cerebro 實例
            initial_cash: 初始資金（若為 None，則從 BROKER_CONFIG 讀取）
        """
        self.cerebro = cerebro
        self.initial_cash = initial_cash if initial_cash is not None else BROKER_CONFIG['initial_cash']
        self.previous_value = self.initial_cash
        
        # 訓練相關
        self.replay_buffer = ReplayBuffer(TRAIN_CONFIG['replay_buffer_size'])
        self.episode_rewards = []
        self.episode_returns = []
        self.training_losses = []
        
    def get_current_portfolio_value(self):
        """
        獲取當前投資組合總價值
        注意：此值包含現金 + 持倉市值（含未實現損益）
        
        Returns:
            float: 當前總資金（包含未實現損益）
        """
        return self.cerebro.broker.getvalue()
    
    def get_portfolio_return(self):
        """
        獲取當前投資組合收益率
        
        Returns:
            float: 收益率（百分比，例如 0.05 代表 5%）
        """
        current_value = self.get_current_portfolio_value()
        portfolio_return = (current_value - self.initial_cash) / self.initial_cash
        return portfolio_return
    
    def get_step_return(self):
        """
        獲取單步收益率（相對於上一步）
        
        Returns:
            float: 單步收益率
        """
        current_value = self.get_current_portfolio_value()
        step_return = (current_value - self.previous_value) / self.previous_value
        self.previous_value = current_value
        return step_return
    
    def calculate_reward(self, step_return=None):
        """
        計算獎勵函數
        
        Args:
            step_return: 單步收益率，如果為 None 則自動計算
            
        Returns:
            float: 獎勵值
        """
        if step_return is None:
            step_return = self.get_step_return()
        
        # 簡單的獎勵函數：直接使用收益率
        # 可以根據需求調整獎勵函數的設計
        reward = step_return * 100  # 放大收益率以便於訓練
        
        return reward
    
    def store_experience(self, state, action, reward, next_state, done):
        """
        存儲經驗到 replay buffer
        
        Args:
            state: 當前狀態
            action: 動作
            reward: 獎勵
            next_state: 下一狀態
            done: 是否結束
        """
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def train_agent(self, agent, batch_size=None):
        """
        訓練 agent
        
        Args:
            agent: SAC agent 實例
            batch_size: 批次大小（若為 None，則從 TRAIN_CONFIG 讀取）
            
        Returns:
            dict: 訓練損失
        """
        if batch_size is None:
            batch_size = TRAIN_CONFIG['batch_size']
        
        if len(self.replay_buffer) < batch_size:
            return None
        
        losses = agent.update(self.replay_buffer, batch_size)
        self.training_losses.append(losses)
        return losses
    
    def reset_episode_tracking(self):
        """
        重置 episode 追蹤變數
        """
        self.previous_value = self.initial_cash
    

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
        



def load_and_split_data(data_path, train_ratio=0.6, val_ratio=0.2):
    """
    載入並切分資料
    
    Args:
        data_path: 資料路徑
        train_ratio: 訓練集比例
        val_ratio: 驗證集比例
        
    Returns:
        tuple: (train_df, val_df, test_df)
    """
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    
    print(f"資料切分:")
    print(f"  訓練集: {len(train_df)} 筆 ({train_df.index[0]} ~ {train_df.index[-1]})")
    print(f"  驗證集: {len(val_df)} 筆 ({val_df.index[0]} ~ {val_df.index[-1]})")
    print(f"  測試集: {len(test_df)} 筆 ({test_df.index[0]} ~ {test_df.index[-1]})")
    
    return train_df, val_df, test_df


def run_training_episode(df, agent, trainer):
    """
    執行一個訓練 episode
    
    Args:
        df: 訓練資料
        agent: SAC agent
        trainer: train_agent_indicator 實例
        
    Returns:
        dict: episode 統計資訊
    """
    cerebro = bt.Cerebro()
    
    # 添加訓練策略
    cerebro.addstrategy(TrainingStrategy, trainer=trainer, agent=agent)
    
    # 添加資料
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
    
    # 設定 broker
    cerebro.broker.setcash(BROKER_CONFIG['initial_cash'])
    cerebro.broker.setcommission(commission=BROKER_CONFIG['commission'])
    
    # 更新 trainer 的 cerebro 引用
    trainer.cerebro = cerebro
    trainer.reset_episode_tracking()
    
    # 執行回測
    cerebro.run()
    
    # 返回統計資訊
    final_value = cerebro.broker.getvalue()
    portfolio_return = (final_value - BROKER_CONFIG['initial_cash']) / BROKER_CONFIG['initial_cash']
    
    return {
        'final_value': final_value,
        'return': portfolio_return
    }


def validate(df, agent):
    """
    驗證模型
    
    Args:
        df: 驗證資料
        agent: SAC agent
        
    Returns:
        dict: 驗證統計資訊
    """
    cerebro = bt.Cerebro()
    
    # 創建臨時 trainer（不進行訓練）
    temp_trainer = train_agent_indicator(cerebro)
    
    # 添加訓練策略（但不訓練）
    cerebro.addstrategy(TrainingStrategy, trainer=None, agent=agent)
    
    # 添加資料
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
    
    # 設定 broker
    cerebro.broker.setcash(BROKER_CONFIG['initial_cash'])
    cerebro.broker.setcommission(commission=BROKER_CONFIG['commission'])
    
    # 執行回測
    cerebro.run()
    
    # 返回統計資訊
    final_value = cerebro.broker.getvalue()
    portfolio_return = (final_value - BROKER_CONFIG['initial_cash']) / BROKER_CONFIG['initial_cash']
    
    return {
        'final_value': final_value,
        'return': portfolio_return
    }


def test_model(df, model_path):
    """
    使用測試集評估模型
    
    Args:
        df: 測試資料
        model_path: 模型路徑
        
    Returns:
        dict: 測試統計資訊
    """
    print("\n" + "="*70)
    print("開始測試模型")
    print("="*70)
    
    # 創建 agent（推理模式）
    import torch
    device_config = TRAIN_CONFIG.get('device', 'auto')
    force_cuda = TRAIN_CONFIG.get('force_cuda', False)
    
    agent = create_sac_agent(
        state_dim=SAC_CONFIG['state_dim'],
        action_dim=SAC_CONFIG['action_dim'],
        learning_rate=SAC_CONFIG['learning_rate'],
        gamma=SAC_CONFIG['gamma'],
        tau=SAC_CONFIG['tau'],
        alpha=SAC_CONFIG['alpha'],
        train_mode=False,  # 推理模式
        device=device_config,
        force_cuda=force_cuda
    )
    print(f"使用設備: {agent.device}")
    
    # 載入最佳模型
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return None
    
    print(f"載入模型: {model_path}")
    agent.load(model_path)
    
    # 執行測試
    cerebro = bt.Cerebro()
    cerebro.addstrategy(TrainingStrategy, trainer=None, agent=agent)
    
    # 添加資料
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
    
    # 設定 broker
    initial_cash = BROKER_CONFIG['initial_cash']
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=BROKER_CONFIG['commission'])
    
    print(f"\n測試資料: {len(df)} 筆 ({df.index[0]} ~ {df.index[-1]})")
    print(f"初始資金: ${initial_cash:,.2f}\n")
    
    # 執行回測
    test_start_time = time.time()
    cerebro.run()
    test_time = time.time() - test_start_time
    
    # 計算統計
    final_value = cerebro.broker.getvalue()
    portfolio_return = (final_value - initial_cash) / initial_cash
    profit = final_value - initial_cash
    
    # 顯示測試結果
    print("="*70)
    print("測試結果")
    print("="*70)
    print(f"初始資金: ${initial_cash:,.2f}")
    print(f"最終資金: ${final_value:,.2f}")
    print(f"淨利潤:   ${profit:,.2f}")
    print(f"收益率:   {portfolio_return*100:.2f}%")
    print(f"測試時間: {test_time:.2f}s")
    print("="*70)
    
    # 顯示回測圖表
    print("\n正在生成回測圖表...")
    try:
        cerebro.plot(style='candlestick')
    except Exception as e:
        print(f"圖表顯示失敗: {e}")
        print("提示：如果在無GUI環境中運行，可能無法顯示圖表")
    
    return {
        'initial_value': initial_cash,
        'final_value': final_value,
        'profit': profit,
        'return': portfolio_return,
        'test_time': test_time
    }


def train_sac_model(train_df, val_df, num_episodes=None, enable_validation=True):
    """
    訓練 SAC 模型
    
    Args:
        train_df: 訓練資料
        val_df: 驗證資料
        num_episodes: 訓練回合數（若為 None，則從 TRAIN_CONFIG 讀取）
        enable_validation: 是否啟用驗證
    """
    if num_episodes is None:
        num_episodes = TRAIN_CONFIG['episodes']
    
    print("\n" + "="*70)
    print("開始訓練 SAC 模型")
    print("="*70)
    
    # 檢查並顯示設備信息
    import torch
    device_config = TRAIN_CONFIG.get('device', 'auto')
    force_cuda = TRAIN_CONFIG.get('force_cuda', False)
    
    print("\n🖥️  設備配置:")
    print(f"  設備選項: {device_config}")
    print(f"  強制使用 CUDA: {force_cuda}")
    print(f"  CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU 型號: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA 版本: {torch.version.cuda}")
    
    # 創建 SAC agent（訓練模式）
    try:
        agent = create_sac_agent(
            state_dim=SAC_CONFIG['state_dim'],
            action_dim=SAC_CONFIG['action_dim'],
            learning_rate=SAC_CONFIG['learning_rate'],
            gamma=SAC_CONFIG['gamma'],
            tau=SAC_CONFIG['tau'],
            alpha=SAC_CONFIG['alpha'],
            train_mode=True,
            device=device_config,
            force_cuda=force_cuda
        )
        print(f"  ✅ 使用設備: {agent.device}")
    except RuntimeError as e:
        print(f"\n❌ 設備配置錯誤: {e}")
        print("\n提示: 如果要使用 CPU 訓練，請在 config.py 中設置:")
        print("  TRAIN_CONFIG['force_cuda'] = False")
        print("  或")
        print("  TRAIN_CONFIG['device'] = 'cpu'")
        raise
    
    # 創建 trainer
    cerebro_placeholder = bt.Cerebro()  # 佔位用
    trainer = train_agent_indicator(cerebro_placeholder)
    
    # 訓練循環
    best_val_return = -float('inf')
    no_improvement_count = 0
    start_time = time.time()
    
    for episode in range(1, num_episodes + 1):
        episode_start_time = time.time()
        
        # 執行訓練 episode
        stats = run_training_episode(train_df, agent, trainer)
        
        episode_time = time.time() - episode_start_time
        elapsed_time = time.time() - start_time
        
        # 打印訓練進度
        if episode % TRAIN_CONFIG['print_frequency'] == 0:
            avg_loss = np.mean([l['critic_loss'] for l in trainer.training_losses[-100:]]) if trainer.training_losses else 0
            print(f"\n[Episode {episode}/{num_episodes}]")
            print(f"  時間: {episode_time:.2f}s (總計: {elapsed_time/60:.2f}min)")
            print(f"  收益率: {stats['return']*100:.2f}%")
            print(f"  最終資金: ${stats['final_value']:.2f}")
            print(f"  Replay Buffer: {len(trainer.replay_buffer)}")
            print(f"  平均 Critic Loss: {avg_loss:.4f}")
        
        # 驗證
        if enable_validation and episode % TRAIN_CONFIG['print_frequency'] == 0:
            val_stats = validate(val_df, agent)
            val_return = val_stats['return']
            print(f"  驗證收益率: {val_return*100:.2f}%")
            
            # 檢查是否為最佳模型
            if val_return > best_val_return:
                best_val_return = val_return
                no_improvement_count = 0
                # 保存最佳模型
                os.makedirs(os.path.dirname(SAC_CONFIG['model_path']), exist_ok=True)
                agent.save(SAC_CONFIG['model_path'])
                print(f"  ✓ 保存最佳模型 (驗證收益率: {best_val_return*100:.2f}%)")
            else:
                no_improvement_count += 1
        
        # 定期保存模型
        if episode % TRAIN_CONFIG['save_frequency'] == 0:
            checkpoint_path = SAC_CONFIG['model_path'].replace('.pth', f'_ep{episode}.pth')
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            agent.save(checkpoint_path)
            print(f"  ✓ 保存檢查點: {checkpoint_path}")
        
        # 早停檢查
        if enable_validation and no_improvement_count >= TRAIN_CONFIG['early_stopping_patience']:
            print(f"\n早停: 驗證收益率已 {no_improvement_count} 個 episode 沒有改善")
            break
        
        # 檢查是否達到目標收益率
        if enable_validation and best_val_return >= TRAIN_CONFIG['target_return']:
            print(f"\n達到目標收益率 {TRAIN_CONFIG['target_return']*100:.2f}%，停止訓練")
            break
    
    total_time = time.time() - start_time
    print("\n" + "="*70)
    print(f"訓練完成！總時間: {total_time/60:.2f} 分鐘")
    print(f"最佳驗證收益率: {best_val_return*100:.2f}%")
    print(f"模型已保存至: {SAC_CONFIG['model_path']}")
    print("="*70)


if __name__ == '__main__':
    # 載入並切分資料
    data_path = DATA_CONFIG['data_path']
    train_df, val_df, test_df = load_and_split_data(
        data_path,
        train_ratio=DATA_CONFIG['train_ratio'],
        val_ratio=DATA_CONFIG['val_ratio']
    )
    
    # 訓練模型
    train_sac_model(
        train_df, 
        val_df,
        num_episodes=TRAIN_CONFIG['episodes'],
        enable_validation=DATA_CONFIG['enable_validation']
    )
    
    # 使用測試集評估最佳模型
    test_stats = test_model(test_df, SAC_CONFIG['model_path'])
    
    # 最終總結
    print("\n" + "="*70)
    print("訓練與測試流程完成！")
    print("="*70)
    if test_stats:
        print(f"✅ 測試集收益率: {test_stats['return']*100:.2f}%")
        print(f"✅ 模型已保存至: {SAC_CONFIG['model_path']}")
    print("="*70)