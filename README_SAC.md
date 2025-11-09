# SAC 模型使用說明

## 簡介

本系統現已支持兩種強化學習算法：
- **DDPG** (Deep Deterministic Policy Gradient)
- **SAC** (Soft Actor-Critic)

SAC 相比 DDPG 的優勢：
1. **更穩定的訓練**：使用雙 Q 網絡減少過估計
2. **自動探索**：通過最大熵目標自動平衡探索與利用
3. **更高的樣本效率**：通常需要更少的訓練數據

## 快速開始

### 1. 使用 SAC 模型（推薦）

```python
from main import agent_indicator
from config import SAC_CONFIG

# 在策略中初始化
self.agent_ind = agent_indicator(
    self.data,
    model_config=SAC_CONFIG,
    model_type='sac'
)
```

### 2. 使用 DDPG 模型

```python
from main import agent_indicator
from config import DDPG_CONFIG

# 在策略中初始化
self.agent_ind = agent_indicator(
    self.data,
    model_config=DDPG_CONFIG,
    model_type='ddpg'
)
```

## 配置說明

### SAC 配置 (`config.py`)

```python
SAC_CONFIG = {
    'state_dim': 10,           # 狀態維度
    'action_dim': 1,           # 動作維度
    'learning_rate': 3e-4,     # 學習率
    'gamma': 0.99,             # 折扣因子
    'tau': 0.005,              # 目標網絡軟更新係數
    'alpha': 0.2,              # 熵溫度參數（SAC 特有）
    'batch_size': 256,         # 批次大小
    'model_path': 'models/sac_model.pth',
}
```

### 關鍵參數說明

- **alpha**: 控制探索程度，值越大探索越多
  - 範圍：0.1 ~ 0.5
  - 推薦：0.2（默認）
  
- **batch_size**: SAC 通常需要更大的批次
  - DDPG: 128
  - SAC: 256（推薦）

## 完整示例

運行 `example_sac.py` 查看完整的 SAC 使用示例：

```bash
python example_sac.py
```

## 模型切換

只需修改兩個參數即可切換模型：

```python
# 方法 1：使用 SAC
agent_indicator(data, model_config=SAC_CONFIG, model_type='sac')

# 方法 2：使用 DDPG
agent_indicator(data, model_config=DDPG_CONFIG, model_type='ddpg')
```

## 代碼結構

```
agent.py           # SAC 和 DDPG 實現
├── SACAgent       # SAC 代理類
│   ├── Actor      # 高斯策略網絡
│   └── Critic     # 雙 Q 網絡
├── DDPGAgent      # DDPG 代理類
│   ├── DDPGActor  # 確定性策略網絡
│   └── DDPGCritic # Q 網絡
└── ReplayBuffer   # 經驗回放緩衝區（共用）

main.py            # 主程序和指標類
config.py          # 配置文件（DDPG_CONFIG, SAC_CONFIG）
example_sac.py     # SAC 使用示例
```

## 性能比較建議

1. 先使用 SAC 訓練，通常能獲得更好的結果
2. 如果 SAC 訓練不穩定，嘗試調整 `alpha` 參數
3. DDPG 在某些簡單環境下可能訓練更快

## 注意事項

1. 確保安裝 PyTorch：`pip install torch`
2. 模型文件會自動保存到 `models/` 目錄
3. SAC 和 DDPG 使用不同的模型文件，不會互相覆蓋
