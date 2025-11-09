DDPG_CONFIG = {
    # 狀態維度計算：mhull(1) + shull(1) + signal(1) + volume(1) + current_price(1) + price_changes(5) = 10
    'state_dim': 10,
    'action_dim': 1,
    'learning_rate': 1e-3,
    'gamma': 0.99,
    'tau': 0.001,
    'enable_training': True,
    'use_lstm': False,
    'lstm_hidden_dim': 128,
    'batch_size': 128,
    'price_change_period': 5,  # 前 n 根 K 線的變化率（會追加 n 個特徵）
    'model_path': 'models/model.pth',  # 模型保存路徑
}

# ========== 資料集配置 ==========
DATA_CONFIG = {
    'train_ratio': 0.6,         # 訓練集比例
    'val_ratio': 0.2,           # 驗證集比例
    'test_ratio': 0.2,          # 測試集比例
    'enable_validation': True,  # 是否啟用驗證
}