# SAC 交易系統 - 推理模式

## 簡介

本系統使用 SAC (Soft Actor-Critic) 強化學習模型進行交易決策，僅支持推理模式（不含訓練功能）。

## 系統架構

```
agent.py           # SAC 模型實現（僅推理）
├── Actor          # 策略網絡（高斯策略）
├── Critic         # 價值網絡（雙 Q 網絡）
└── SACAgent       # SAC 代理類

main.py            # 主程序和回測策略
├── strategy1      # 交易策略
└── agent_indicator # SAC 指標（整合模型與技術指標）

config.py          # 配置文件
indicator.py       # 技術指標工廠
```

## 狀態向量組成

系統使用以下特徵構建狀態向量（共 12 維），**所有特徵均使用 Z-score 正規化**：

1. **HullMA 指標** (3 維)
   - mhull: 中期 Hull MA
   - shull: 短期 Hull MA  
   - signal: 交叉信號

2. **基礎特徵** (2 維)
   - volume: 成交量
   - current_price: 當前價格

3. **價格變化率** (5 維)
   - 前 5 根 K 線的各自變化率

4. **技術指標** (2 維)
   - RSI: 相對強度指數 (週期 14)
   - CCI: 商品通道指數 (週期 20)

### Z-score 正規化

所有特徵使用滾動窗口 Z-score 正規化：
- **公式**: `(x - mean) / std`
- **窗口大小**: 100 筆歷史數據（可配置）
- **優點**: 
  - 消除不同特徵的量綱差異
  - 使模型訓練更穩定
  - 提高收斂速度

## 使用方法

### 1. 配置設定

在 `config.py` 中調整參數：

```python
SAC_CONFIG = {
    'state_dim': 12,           # 狀態維度（固定）
    'action_dim': 1,           # 動作維度（固定）
    'rsi_period': 14,          # RSI 週期
    'cci_period': 20,          # CCI 週期
    'normalization_window': 100,  # Z-score 正規化窗口大小
    'model_path': 'models/sac_model.pth',  # 模型路徑
}
```

### 2. 準備模型

確保已訓練的 SAC 模型存在於指定路徑：
```
models/sac_model.pth
```

### 3. 運行回測

```bash
python main.py
```

## 文件說明

- **agent.py**: SAC 模型實現，包含 Actor、Critic 網絡和推理邏輯
- **main.py**: 主程序，包含策略和指標實現
- **config.py**: 配置參數
- **indicator.py**: 技術指標工廠類
- **indicators/**: 技術指標實現（HullMA, RSI, CCI）

## 注意事項

1. **必須先訓練模型**：本系統僅支持推理，需要預先訓練好的模型
2. **狀態維度固定**：如果修改技術指標配置，需確保狀態維度保持 12 維
3. **模型兼容性**：加載的模型必須與當前狀態維度匹配

## 依賴項

```bash
pip install torch backtrader pandas numpy
```

## 交易邏輯

系統根據 SAC 模型輸出的動作值進行交易：
- **action > 0.5**: 買入信號
- **action < -0.5**: 賣出信號（平倉）
- **-0.5 ≤ action ≤ 0.5**: 持有

## 擴展指標

如需添加更多技術指標到狀態：

1. 在 `agent_indicator.__init__()` 中初始化指標
2. 在 `_build_state()` 中添加到 state_list
3. 更新 `config.py` 中的 `state_dim`
4. 重新訓練模型以匹配新的狀態維度
