"""
獎勵函數模組

提供多種獎勵函數供訓練使用，可以根據需求自定義修改
"""
import numpy as np


def calculate_reward_v1(position, action, price_change, commission=0.001):
    """
    基礎獎勵函數 v1
    
    參數:
        position: 當前持倉狀態 (0: 無持倉, 1: 持倉)
        action: 代理輸出的動作 (-1 到 1 之間)
        price_change: 價格變化率 (當前價格 - 前一價格) / 前一價格
        commission: 手續費率
    
    返回:
        reward: 獎勵值
    """
    reward = 0.0
    
    # 動作閾值
    buy_threshold = 0.3
    sell_threshold = -0.3
    
    # 持倉時的獎勵（根據價格變化）
    if position == 1:
        # 持倉期間，價格上漲給予正獎勵，下跌給予負獎勵
        reward = price_change * 100  # 放大獎勵信號
        
        # 如果在持倉時選擇賣出
        if action < sell_threshold:
            # 如果是獲利賣出，給予額外獎勵
            if price_change > 0:
                reward += 0.5
            # 如果是止損賣出，減少懲罰
            else:
                reward += 0.2
            
            # 扣除交易成本
            reward -= commission * 100
    
    # 無持倉時的獎勵
    else:
        # 無持倉期間，避免價格下跌的風險給予小獎勵
        if price_change < 0:
            reward = abs(price_change) * 50  # 躲過下跌
        
        # 如果選擇買入
        if action > buy_threshold:
            # 在價格上漲趨勢買入，給予獎勵
            if price_change > 0:
                reward += 0.3
            # 在價格下跌時買入，給予懲罰
            else:
                reward -= 0.3
            
            # 扣除交易成本
            reward -= commission * 100
    
    return reward


def calculate_reward_v2(position, action, price_change, rsi, cci, commission=0.001):
    """
    進階獎勵函數 v2 - 考慮技術指標
    
    參數:
        position: 當前持倉狀態 (0: 無持倉, 1: 持倉)
        action: 代理輸出的動作 (-1 到 1 之間)
        price_change: 價格變化率
        rsi: RSI 指標值 (0-100)
        cci: CCI 指標值
        commission: 手續費率
    
    返回:
        reward: 獎勵值
    """
    reward = 0.0
    
    buy_threshold = 0.3
    sell_threshold = -0.3
    
    # 持倉時的獎勵
    if position == 1:
        # 基礎價格變化獎勵
        reward = price_change * 100
        
        # RSI 超買區域（>70）應該賣出
        if rsi > 70 and action < sell_threshold:
            reward += 0.5  # 獎勵在超買時賣出
        elif rsi > 70 and action > buy_threshold:
            reward -= 0.5  # 懲罰在超買時繼續持有或買入
        
        # 賣出動作
        if action < sell_threshold:
            if price_change > 0:
                reward += 0.5
            reward -= commission * 100
    
    # 無持倉時的獎勵
    else:
        if price_change < 0:
            reward = abs(price_change) * 50
        
        # RSI 超賣區域（<30）應該買入
        if rsi < 30 and action > buy_threshold:
            reward += 0.5  # 獎勵在超賣時買入
        elif rsi < 30 and action < sell_threshold:
            reward -= 0.3  # 懲罰在超賣時不買入
        
        # CCI 極端值也考慮
        if cci < -100 and action > buy_threshold:
            reward += 0.3  # CCI 超賣時買入
        
        # 買入動作
        if action > buy_threshold:
            if price_change > 0:
                reward += 0.3
            else:
                reward -= 0.3
            reward -= commission * 100
    
    return reward


def calculate_reward_v3(position, action, price_change, total_return, commission=0.001):
    """
    簡化獎勵函數 v3 - 直接優化總收益
    
    參數:
        position: 當前持倉狀態 (0: 無持倉, 1: 持倉)
        action: 代理輸出的動作 (-1 到 1 之間)
        price_change: 價格變化率
        total_return: 當前總收益率
        commission: 手續費率
    
    返回:
        reward: 獎勵值
    """
    reward = 0.0
    
    buy_threshold = 0.3
    sell_threshold = -0.3
    
    # 直接使用收益變化作為獎勵
    if position == 1:
        reward = price_change * 100
    else:
        # 無持倉時保持小幅正獎勵（鼓勵資金安全）
        reward = 0.1
    
    # 交易成本
    if action > buy_threshold or action < sell_threshold:
        reward -= commission * 100
    
    # 懲罰過度交易（太頻繁的買賣）
    if abs(action) < 0.2:  # 中性動作
        reward += 0.05  # 小獎勵鼓勵謹慎
    
    return reward


# 默認使用的獎勵函數
def calculate_reward(position, action, price_change, rsi=None, cci=None, total_return=None, commission=0.001):
    """
    預設獎勵函數（使用 v1）
    
    可以根據需要切換不同版本的獎勵函數
    """
    # 預設使用 v1
    return calculate_reward_v1(position, action, price_change, commission)
    
    # 如果需要使用 v2（需要提供 rsi, cci）
    # if rsi is not None and cci is not None:
    #     return calculate_reward_v2(position, action, price_change, rsi, cci, commission)
    
    # 如果需要使用 v3（需要提供 total_return）
    # if total_return is not None:
    #     return calculate_reward_v3(position, action, price_change, total_return, commission)
