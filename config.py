#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
配置模块
========

包含所有实验配置参数和全局设置
"""

import torch
import numpy as np
import warnings
import matplotlib.pyplot as plt

# 忽略警告
warnings.filterwarnings('ignore')


# ================================================================================
# 设备配置
# ================================================================================

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ================================================================================
# 随机种子设置
# ================================================================================

def set_seed(seed: int = 42):
    """设置所有随机种子以确保实验可复现"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ================================================================================
# 实验配置
# ================================================================================

class Config:
    """实验配置类"""
    
    # 数据配置
    TICKER = 'AAPL'                    # 股票代码
    START_DATE = '2010-01-01'          # 数据开始日期
    END_DATE = None                     # 数据结束日期（None表示到最后）
    
    # 序列配置
    SEQUENCE_LENGTH = 60               # 输入序列长度
    PREDICTION_HORIZON = 1             # 预测未来多少天
    
    # 训练配置
    BATCH_SIZE = 32                    # 批次大小
    EPOCHS = 150                       # 最大训练轮数
    LEARNING_RATE = 0.001              # 学习率
    PATIENCE = 25                      # 早停耐心值
    
    # 模型配置
    HIDDEN_SIZE = 128                  # 隐藏层维度
    NUM_LAYERS = 2                     # 网络层数
    DROPOUT = 0.2                      # Dropout比率
    
    # 数据划分比例
    TRAIN_RATIO = 0.7                  # 训练集比例
    VAL_RATIO = 0.15                   # 验证集比例
    
    # 保存路径
    SAVE_DIR = './results'             # 结果保存目录
    
    @classmethod
    def to_dict(cls) -> dict:
        """将配置转换为字典"""
        return {
            'ticker': cls.TICKER,
            'sequence_length': cls.SEQUENCE_LENGTH,
            'batch_size': cls.BATCH_SIZE,
            'epochs': cls.EPOCHS,
            'learning_rate': cls.LEARNING_RATE,
            'hidden_size': cls.HIDDEN_SIZE,
            'num_layers': cls.NUM_LAYERS,
            'save_dir': cls.SAVE_DIR
        }
    
    @classmethod
    def print_config(cls):
        """打印配置信息"""
        print("\n实验配置:")
        print(f"  - 股票代码: {cls.TICKER}")
        print(f"  - 序列长度: {cls.SEQUENCE_LENGTH}")
        print(f"  - 批次大小: {cls.BATCH_SIZE}")
        print(f"  - 训练轮数: {cls.EPOCHS}")
        print(f"  - 学习率: {cls.LEARNING_RATE}")
        print(f"  - 隐藏层维度: {cls.HIDDEN_SIZE}")
        print(f"  - 网络层数: {cls.NUM_LAYERS}")


# ================================================================================
# 可视化设置
# ================================================================================

def setup_plotting():
    """设置绘图参数"""
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['figure.dpi'] = 100


# 初始化
set_seed(42)
setup_plotting()
print(f"使用设备: {DEVICE}")

