"""
SAC 代理訓練腳本 - 使用 Walk Forward 方法

功能：
1. 使用 Walk Forward 方法進行時間序列交叉驗證
2. 訓練 SAC 代理進行交易決策
3. 評估每個窗口的訓練效果
4. 保存訓練好的模型
"""

import pandas as pd
import numpy as np
import os
from collections import deque
from datetime import datetime

from agent import create_sac_agent, ReplayBuffer
from config import SAC_CONFIG, DATA_CONFIG, WALK_FORWARD_CONFIG, TRAIN_CONFIG
from reward_function import calculate_reward
from indicator import IndicatorFactory


class TradingEnvironment:
    """交易環境"""
    def __init__(self, data, initial_cash=100000, commission=0.001):
        """
        參數：
            data: pandas DataFrame，包含 OHLCV 數據
            initial_cash: 初始資金
            commission: 手續費率
        """
        self.data = data.reset_index(drop=True)
        self.initial_cash = initial_cash
        self.commission = commission
        
        # 先計算技術指標
        self._calculate_indicators()
        
        # 再重置環境（因為 reset 會調用 _get_state，需要技術指標）
        self.reset()
    
    def _calculate_indicators(self):
        """計算技術指標"""
        # 這裡使用簡單的方法計算指標，實際應該使用 IndicatorFactory
        # HullMA (簡化版，實際應該用正確的 HullMA)
        self.data['mhull'] = self.data['close'].rolling(window=20).mean()
        self.data['shull'] = self.data['close'].rolling(window=10).mean()
        self.data['signal'] = self.data['mhull'] - self.data['shull']
        
        # RSI
        delta = self.data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.data['rsi'] = 100 - (100 / (1 + rs))
        
        # CCI
        tp = (self.data['high'] + self.data['low'] + self.data['close']) / 3
        sma = tp.rolling(window=20).mean()
        mad = tp.rolling(window=20).apply(lambda x: np.abs(x - x.mean()).mean())
        self.data['cci'] = (tp - sma) / (0.015 * mad)
        
        # 填充 NaN
        self.data.fillna(0, inplace=True)
    
    def reset(self):
        """重置環境"""
        self.current_step = 0
        self.cash = self.initial_cash
        self.position = 0  # 0: 無持倉, 1: 持倉
        self.entry_price = 0
        self.total_profit = 0
        self.state_history = deque(maxlen=SAC_CONFIG['normalization_window'])
        return self._get_state()
    
    def _get_state(self):
        """構建狀態向量"""
        if self.current_step >= len(self.data):
            return None
        
        row = self.data.iloc[self.current_step]
        
        # 基礎特徵
        current_price = row['close']
        mhull = row['mhull']
        shull = row['shull']
        signal = row['signal']
        rsi = row['rsi']
        cci = row['cci']
        volume = row['volume']
        
        # 價格變化率
        price_changes = []
        for i in range(1, SAC_CONFIG['price_change_period'] + 1):
            if self.current_step >= i:
                prev_price = self.data.iloc[self.current_step - i]['close']
                price_change = (current_price - prev_price) / prev_price
            else:
                price_change = 0
            price_changes.append(price_change)
        
        # 組合狀態
        state = [mhull, shull, signal, volume, current_price] + price_changes + [rsi, cci]
        state = np.array(state, dtype=np.float32)
        
        # Z-score 正規化
        return self._normalize_state(state)
    
    def _normalize_state(self, state):
        """Z-score 正規化"""
        self.state_history.append(state.copy())
        
        if len(self.state_history) < 2:
            return np.zeros_like(state)
        
        history_array = np.array(self.state_history)
        mean = np.mean(history_array, axis=0)
        std = np.std(history_array, axis=0)
        std = np.where(std < 1e-8, 1.0, std)
        
        return (state - mean) / std
    
    def step(self, action):
        """執行動作
        
        參數：
            action: 動作值，範圍 [-1, 1]
                    > 0.3: 買入
                    < -0.3: 賣出
                    其他: 持有
        
        返回：
            next_state, reward, done, info
        """
        if self.current_step >= len(self.data) - 1:
            return None, 0, True, {}
        
        current_price = self.data.iloc[self.current_step]['close']
        next_price = self.data.iloc[self.current_step + 1]['close']
        price_change = (next_price - current_price) / current_price
        
        # 獲取技術指標（用於獎勵計算）
        rsi = self.data.iloc[self.current_step]['rsi']
        cci = self.data.iloc[self.current_step]['cci']
        
        # 執行交易
        action_value = action[0]
        
        if self.position == 0 and action_value > 0.3:
            # 買入
            self.position = 1
            self.entry_price = next_price
            transaction_cost = self.commission
        elif self.position == 1 and action_value < -0.3:
            # 賣出
            profit = (next_price - self.entry_price) / self.entry_price
            self.total_profit += profit
            self.position = 0
            transaction_cost = self.commission
        else:
            # 持有
            transaction_cost = 0
        
        # 計算獎勵
        reward = calculate_reward(
            position=self.position,
            action=action_value,
            price_change=price_change,
            rsi=rsi,
            cci=cci,
            commission=self.commission
        )
        
        # 下一步
        self.current_step += 1
        next_state = self._get_state()
        done = self.current_step >= len(self.data) - 1
        
        info = {
            'position': self.position,
            'total_profit': self.total_profit,
            'price': next_price
        }
        
        return next_state, reward, done, info


def train_episode(env, agent, replay_buffer, config):
    """訓練一個 episode
    
    返回：
        total_reward: 總獎勵
        total_profit: 總收益
        steps: 步數
    """
    state = env.reset()
    total_reward = 0
    steps = 0
    
    while True:
        # 選擇動作
        if len(replay_buffer) < config['start_steps']:
            # 隨機探索
            action = np.random.uniform(-1, 1, size=(1,))
        else:
            # 使用策略
            action = agent.select_action(state, evaluate=False)
        
        # 執行動作
        next_state, reward, done, info = env.step(action)
        
        if next_state is None:
            break
        
        # 存儲經驗
        replay_buffer.push(state, action, reward, next_state, done)
        
        # 更新網絡
        if len(replay_buffer) >= config['batch_size'] and steps % config['update_frequency'] == 0:
            agent.update(replay_buffer, config['batch_size'])
        
        state = next_state
        total_reward += reward
        steps += 1
        
        if done:
            break
    
    return total_reward, info['total_profit'], steps


def evaluate(env, agent):
    """評估代理"""
    state = env.reset()
    total_reward = 0
    
    while True:
        action = agent.select_action(state, evaluate=True)
        next_state, reward, done, info = env.step(action)
        
        if next_state is None:
            break
        
        state = next_state
        total_reward += reward
        
        if done:
            break
    
    return total_reward, info['total_profit']


def walk_forward_train(data, sac_config, walk_config, train_config):
    """Walk Forward 訓練
    
    參數：
        data: 完整的數據集
        sac_config: SAC 配置
        walk_config: Walk Forward 配置
        train_config: 訓練配置
    """
    print("=" * 80)
    print("開始 Walk Forward 訓練")
    print("=" * 80)
    
    train_window = walk_config['train_window']
    test_window = walk_config['test_window']
    step_size = walk_config['step_size']
    
    total_data_len = len(data)
    results = []
    
    # 創建模型保存目錄
    os.makedirs('models', exist_ok=True)
    
    # Walk Forward 循環
    window_idx = 0
    start_idx = 0
    
    while start_idx + train_window + test_window <= total_data_len:
        window_idx += 1
        
        # 分割訓練和測試數據
        train_end = start_idx + train_window
        test_end = train_end + test_window
        
        train_data = data.iloc[start_idx:train_end].copy()
        test_data = data.iloc[train_end:test_end].copy()
        
        print(f"\n{'=' * 80}")
        print(f"Window {window_idx}")
        print(f"訓練期間: {start_idx} ~ {train_end} ({len(train_data)} 筆)")
        print(f"測試期間: {train_end} ~ {test_end} ({len(test_data)} 筆)")
        print(f"{'=' * 80}")
        
        # 創建訓練環境和代理
        train_env = TradingEnvironment(train_data)
        agent = create_sac_agent(
            state_dim=sac_config['state_dim'],
            action_dim=sac_config['action_dim'],
            learning_rate=sac_config['learning_rate'],
            gamma=sac_config['gamma'],
            tau=sac_config['tau'],
            alpha=sac_config['alpha'],
            train_mode=True
        )
        replay_buffer = ReplayBuffer(train_config['replay_buffer_size'])
        
        # 訓練
        print(f"\n開始訓練...")
        best_reward = -np.inf
        patience_counter = 0
        
        for episode in range(train_config['episodes']):
            total_reward, total_profit, steps = train_episode(train_env, agent, replay_buffer, train_config)
            
            if (episode + 1) % train_config['print_frequency'] == 0:
                print(f"Episode {episode + 1}/{train_config['episodes']}: "
                      f"Reward={total_reward:.2f}, Profit={total_profit:.4f}, Steps={steps}")
            
            # 保存最佳模型
            if total_reward > best_reward:
                best_reward = total_reward
                patience_counter = 0
                model_path = f"models/sac_window_{window_idx}_best.pth"
                agent.save(model_path)
            else:
                patience_counter += 1
            
            # 早停
            if patience_counter >= train_config['early_stopping_patience']:
                print(f"早停於 Episode {episode + 1}")
                break
        
        # 載入最佳模型
        agent.load(f"models/sac_window_{window_idx}_best.pth")
        
        # 在測試集上評估
        print(f"\n開始測試...")
        test_env = TradingEnvironment(test_data)
        test_reward, test_profit = evaluate(test_env, agent)
        
        print(f"測試結果: Reward={test_reward:.2f}, Profit={test_profit:.4f}")
        
        results.append({
            'window': window_idx,
            'train_start': start_idx,
            'train_end': train_end,
            'test_end': test_end,
            'train_reward': best_reward,
            'test_reward': test_reward,
            'test_profit': test_profit
        })
        
        # 滑動窗口
        start_idx += step_size
    
    # 保存最終模型
    final_model_path = sac_config['model_path']
    agent.save(final_model_path)
    print(f"\n最終模型已保存至: {final_model_path}")
    
    # 打印總結
    print(f"\n{'=' * 80}")
    print("Walk Forward 訓練完成")
    print(f"{'=' * 80}")
    
    results_df = pd.DataFrame(results)
    print("\n訓練結果總結:")
    print(results_df)
    
    avg_test_profit = results_df['test_profit'].mean()
    print(f"\n平均測試收益: {avg_test_profit:.4f}")
    
    # 保存結果
    results_df.to_csv('models/walk_forward_results.csv', index=False)
    print(f"結果已保存至: models/walk_forward_results.csv")
    
    return results_df


if __name__ == '__main__':
    # 載入數據
    print("載入數據...")
    df = pd.read_csv(DATA_CONFIG['data_path'])
    
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
    
    print(f"數據載入完成，共 {len(df)} 筆資料")
    print(f"日期範圍: {df.index[0]} ~ {df.index[-1]}")
    
    # 開始訓練
    results = walk_forward_train(
        data=df,
        sac_config=SAC_CONFIG,
        walk_config=WALK_FORWARD_CONFIG,
        train_config=TRAIN_CONFIG
    )
    
    print("\n訓練完成！")