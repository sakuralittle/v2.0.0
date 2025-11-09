SAC_CONFIG = {
    # 狀態維度計算：mhull(1) + shull(1) + signal(1) + volume(1) + current_price(1) + price_changes(5) + rsi(1) + cci(1) = 12
    # 所有特徵使用 Z-score 正規化（基於滾動窗口 100 筆數據）
    'state_dim': 12,
    'action_dim': 1,
    'learning_rate': 3e-4,
    'gamma': 0.99,
    'tau': 0.005,
    'alpha': 0.2,  # 熵溫度參數
    'price_change_period': 5,  # 前 n 根 K 線的變化率（會追加 n 個特徵）
    'rsi_period': 14,  # RSI 週期
    'cci_period': 20,  # CCI 週期
    'normalization_window': 100,  # Z-score 正規化的滾動窗口大小
    'model_path': 'models/sac_model.pth',  # 模型保存路徑
}

# ========== 資料集配置 ==========
DATA_CONFIG = {
    'data_path': 'h:/program_dev/trading_system/v1.0.0/data/data_15m.csv',  # 資料路徑
    'train_ratio': 0.6,         # 訓練集比例
    'val_ratio': 0.2,           # 驗證集比例
    'test_ratio': 0.2,          # 測試集比例
    'enable_validation': True,  # 是否啟用驗證
}

# ========== Walk Forward 配置 ==========
WALK_FORWARD_CONFIG = {
    'train_window': 5000,       # 每次訓練窗口大小（K線數量）
    'test_window': 1000,        # 每次測試窗口大小
    'step_size': 500,           # 滑動步長
    'min_train_size': 2000,     # 最小訓練數據量
}

# ========== 訓練配置 ==========
TRAIN_CONFIG = {
    'episodes': 100,            # 每個訓練窗口的訓練回合數
    'batch_size': 256,          # 批次大小
    'replay_buffer_size': 100000,  # Replay Buffer 大小
    'update_frequency': 1,      # 每 N 步更新一次網絡
    'start_steps': 1000,        # 開始訓練前的隨機探索步數
    'save_frequency': 10,       # 每 N 個 episode 保存一次模型
    'print_frequency': 5,       # 每 N 個 episode 打印一次訓練信息
    'early_stopping_patience': 20,  # 早停耐心值
    'target_return': 0.2,       # 目標收益率（用於早停）
}

# ========== Broker 配置 ==========
BROKER_CONFIG = {
    'initial_cash': 100000.0,   # 初始資金
    'commission': 0.001,        # 手續費率（0.1%）
}