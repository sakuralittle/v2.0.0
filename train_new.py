"""
SAC 代理訓練腳本 - 使用 Walk Forward 方法（重構版）

功能：
1. 使用 Walk Forward 方法進行時間序列交叉驗證
2. 訓練 SAC 代理進行交易決策
3. 使用 Backtrader 的 IndicatorFactory 計算技術指標
4. 評估每個窗口的訓練效果
5. 保存訓練好的模型
"""

import pandas as pd
import numpy as np
import os
import time
import backtrader as bt
from collections import deque
from datetime import datetime, timedelta

from agent import create_sac_agent, ReplayBuffer
from config import SAC_CONFIG, DATA_CONFIG, WALK_FORWARD_CONFIG, TRAIN_CONFIG
from reward_function import calculate_reward
from indicator import IndicatorFactory


class IndicatorCalculator(bt.Strategy):
    """用於預先計算所有技術指標的策略"""
    def __init__(self):
        # 使用 IndicatorFactory 計算技術指標
        self.hullma = IndicatorFactory.hullma_indicator(self.data)
        self.rsi = IndicatorFactory.rsi(self.data, period=SAC_CONFIG['rsi_period'])
        self.cci = IndicatorFactory.cci(self.data, period=SAC_CONFIG['cci_period'])
        
        # 儲存指標值
        self.indicator_values = {
            'mhull': [],
            'shull': [],
            'signal': [],
            'rsi': [],
            'cci': []
        }
    
    def next(self):
        """每根 K 線記錄指標值"""
        self.indicator_values['mhull'].append(self.hullma.mhull[0])
        self.indicator_values['shull'].append(self.hullma.shull[0])
        self.indicator_values['signal'].append(self.hullma.signal[0])
        self.indicator_values['rsi'].append(self.rsi.rsi[0])
        self.indicator_values['cci'].append(self.cci.cci[0])


class TradingEnvironment:
    """交易環境 - 使用 Backtrader 計算指標，自行管理交易邏輯"""
    def __init__(self, data, initial_cash=100000, commission=0.001):
        """
        參數：
            data: pandas DataFrame，包含 OHLCV 數據
            initial_cash: 初始資金
            commission: 手續費率
        """
        self.raw_data = data.copy().reset_index(drop=True)
        self.initial_cash = initial_cash
        self.commission = commission
        
        # 使用 Backtrader 預先計算所有指標
        print("計算技術指標...")
        self._precalculate_indicators()
        print("指標計算完成")
        
        self.reset()
    
    def _precalculate_indicators(self):
        """使用 Backtrader 預先計算所有技術指標"""
        data = self.raw_data.copy()
        
        # 確保有時間索引
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.date_range(start='2020-01-01', periods=len(data), freq='15min')
        
        # 創建 Backtrader 數據源
        bt_data = bt.feeds.PandasData(
            dataname=data,
            datetime=None,
            open='open',
            high='high',
            low='low',
            close='close',
            volume='volume',
            openinterest=-1
        )
        
        # 運行 Backtrader 計算指標
        cerebro = bt.Cerebro()
        cerebro.adddata(bt_data)
        cerebro.addstrategy(IndicatorCalculator)
        
        strategies = cerebro.run()
        calculator = strategies[0]
        
        # 將指標值添加到數據中
        # 注意：Backtrader 需要一定數量的數據來初始化指標，所以前面幾根 K 線沒有值
        # 我們需要對齊長度
        data_len = len(self.raw_data)
        indicator_len = len(calculator.indicator_values['mhull'])
        
        # 前面補 0
        padding = data_len - indicator_len
        
        for key in calculator.indicator_values:
            padded_values = [0] * padding + calculator.indicator_values[key]
            self.raw_data[key] = padded_values[:data_len]  # 確保長度一致
    
    def reset(self):
        """重置環境"""
        self.current_step = 0
        self.cash = self.initial_cash
        self.position = 0  # 0: 無持倉, 1: 持倉
        self.entry_price = 0
        self.total_profit = 0
        self.num_trades = 0
        self.state_history = deque(maxlen=SAC_CONFIG['normalization_window'])
        return self._get_state()
    
    def _get_state(self):
        """構建狀態向量"""
        if self.current_step >= len(self.raw_data):
            return None
        
        row = self.raw_data.iloc[self.current_step]
        
        # 從預計算的指標獲取值
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
                prev_price = self.raw_data.iloc[self.current_step - i]['close']
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
        """執行一步
        
        參數：
            action: 動作值，範圍 [-1, 1]
        
        返回：
            next_state, reward, done, info
        """
        if self.current_step >= len(self.raw_data) - 1:
            return None, 0, True, {}
        
        current_price = self.raw_data.iloc[self.current_step]['close']
        next_price = self.raw_data.iloc[self.current_step + 1]['close']
        price_change = (next_price - current_price) / current_price
        
        # 獲取技術指標（用於獎勵計算）
        rsi = self.raw_data.iloc[self.current_step]['rsi']
        cci = self.raw_data.iloc[self.current_step]['cci']
        
        # 執行交易（使用 Backtrader 風格的資金管理）
        action_value = action[0]
        
        if self.position == 0 and action_value > 0.3:
            # 買入：滿倉
            self.position = 1
            self.entry_price = next_price * (1 + self.commission)  # 加上手續費
        elif self.position == 1 and action_value < -0.3:
            # 賣出
            profit = (next_price * (1 - self.commission) - self.entry_price) / self.entry_price
            self.total_profit += profit
            self.cash = self.cash * (1 + profit)  # 更新資金
            self.position = 0
            self.num_trades += 1
        
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
        done = self.current_step >= len(self.raw_data) - 1
        
        # 計算總資產值
        current_value = self.cash
        if self.position == 1:
            # 如果有持倉，計算當前市值
            current_value = self.cash * (1 + (next_price - self.entry_price) / self.entry_price)
        
        info = {
            'position': self.position,
            'total_profit': self.total_profit,
            'num_trades': self.num_trades,
            'portfolio_value': current_value,
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
    
    # 計算總窗口數
    total_windows = 0
    temp_idx = 0
    while temp_idx + train_window + test_window <= total_data_len:
        total_windows += 1
        temp_idx += step_size
    
    print(f"總共將訓練 {total_windows} 個窗口")
    print("=" * 80)
    
    # Walk Forward 循環
    window_idx = 0
    start_idx = 0
    window_times = []
    overall_start_time = time.time()
    
    while start_idx + train_window + test_window <= total_data_len:
        window_idx += 1
        window_start_time = time.time()
        
        # 分割訓練和測試數據
        train_end = start_idx + train_window
        test_end = train_end + test_window
        
        train_data = data.iloc[start_idx:train_end].copy()
        test_data = data.iloc[train_end:test_end].copy()
        
        print(f"\n{'=' * 80}")
        print(f"Window {window_idx}/{total_windows}")
        print(f"訓練期間: {start_idx} ~ {train_end} ({len(train_data)} 筆)")
        print(f"測試期間: {train_end} ~ {test_end} ({len(test_data)} 筆)")
        
        # 顯示整體進度和剩餘時間
        if window_times:
            avg_window_time = np.mean(window_times)
            remaining_windows = total_windows - window_idx + 1
            estimated_total_remaining = avg_window_time * remaining_windows
            
            if estimated_total_remaining < 3600:
                time_str = f"{estimated_total_remaining/60:.1f}分鐘"
            else:
                time_str = f"{estimated_total_remaining/3600:.1f}小時"
            
            print(f"進度: {window_idx}/{total_windows} ({100*window_idx/total_windows:.1f}%), 預估剩餘: {time_str}")
        
        print(f"{'=' * 80}")
        
        # 創建訓練環境和代理
        train_env = TradingEnvironment(train_data, initial_cash=100000, commission=0.001)
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
        
        # 時間追蹤
        episode_start_time = time.time()
        episode_times = []
        
        for episode in range(train_config['episodes']):
            ep_start = time.time()
            total_reward, total_profit, steps = train_episode(train_env, agent, replay_buffer, train_config)
            ep_time = time.time() - ep_start
            episode_times.append(ep_time)
            
            if (episode + 1) % train_config['print_frequency'] == 0:
                # 計算平均時間和剩餘時間
                avg_time = np.mean(episode_times[-10:])  # 使用最近10個episode的平均時間
                remaining_episodes = train_config['episodes'] - (episode + 1)
                estimated_remaining = avg_time * remaining_episodes
                
                # 格式化剩餘時間
                if estimated_remaining < 60:
                    time_str = f"{estimated_remaining:.0f}秒"
                elif estimated_remaining < 3600:
                    time_str = f"{estimated_remaining/60:.1f}分鐘"
                else:
                    time_str = f"{estimated_remaining/3600:.1f}小時"
                
                print(f"Episode {episode + 1}/{train_config['episodes']}: "
                      f"Reward={total_reward:.2f}, Profit={total_profit:.4f}, Steps={steps} | "
                      f"耗時={ep_time:.1f}秒, 預估剩餘={time_str}")
            
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
        
        # 訓練總時間
        total_train_time = time.time() - episode_start_time
        print(f"\n訓練完成，總耗時: {total_train_time/60:.1f}分鐘")
        
        # 載入最佳模型
        agent.load(f"models/sac_window_{window_idx}_best.pth")
        
        # 在測試集上評估
        print(f"\n開始測試...")
        test_env = TradingEnvironment(test_data, initial_cash=100000, commission=0.001)
        test_reward, test_profit = evaluate(test_env, agent)
        
        print(f"測試結果: Reward={test_reward:.2f}, Profit={test_profit:.4f}")
        
        # 記錄窗口耗時
        window_time = time.time() - window_start_time
        window_times.append(window_time)
        print(f"當前窗口耗時: {window_time/60:.1f}分鐘")
        
        results.append({
            'window': window_idx,
            'train_start': start_idx,
            'train_end': train_end,
            'test_end': test_end,
            'train_reward': best_reward,
            'test_reward': test_reward,
            'test_profit': test_profit,
            'window_time_minutes': window_time / 60
        })
        
        # 滑動窗口
        start_idx += step_size
    
    # 總耗時
    total_elapsed = time.time() - overall_start_time
    
    # 保存最終模型
    final_model_path = sac_config['model_path']
    agent.save(final_model_path)
    print(f"\n最終模型已保存至: {final_model_path}")
    
    # 打印總結
    print(f"\n{'=' * 80}")
    print("Walk Forward 訓練完成")
    print(f"總耗時: {total_elapsed/3600:.2f}小時 ({total_elapsed/60:.1f}分鐘)")
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
