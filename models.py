#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
神经网络模型模块
================

包含LSTM、GRU、Attention-LSTM、Transformer-LSTM等模型定义
"""

import torch
import torch.nn as nn


# ================================================================================
# LSTM模型
# ================================================================================

class LSTMModel(nn.Module):
    """
    LSTM股票预测模型
    
    架构：
        - 多层LSTM + Dropout
        - 全连接层输出预测值
    """
    
    def __init__(self, input_size: int, hidden_size: int = 128, 
                 num_layers: int = 2, dropout: float = 0.2):
        """
        初始化LSTM模型
        
        Args:
            input_size: 输入特征维度
            hidden_size: LSTM隐藏层维度
            num_layers: LSTM层数
            dropout: Dropout比率
        """
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        prediction = self.fc(last_output)
        return prediction.squeeze(-1)


# ================================================================================
# GRU模型
# ================================================================================

class GRUModel(nn.Module):
    """
    GRU股票预测模型
    
    架构：
        - 多层GRU + Dropout
        - 全连接层输出预测值
    """
    
    def __init__(self, input_size: int, hidden_size: int = 128, 
                 num_layers: int = 2, dropout: float = 0.2):
        """
        初始化GRU模型
        
        Args:
            input_size: 输入特征维度
            hidden_size: GRU隐藏层维度
            num_layers: GRU层数
            dropout: Dropout比率
        """
        super(GRUModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        gru_out, h_n = self.gru(x)
        last_output = gru_out[:, -1, :]
        prediction = self.fc(last_output)
        return prediction.squeeze(-1)


# ================================================================================
# Transformer-LSTM混合模型
# ================================================================================

class TransformerLSTMModel(nn.Module):
    """
    Transformer-LSTM混合模型
    
    结合LSTM的时序建模能力与Transformer的自注意力机制，
    通过残差连接融合两种特征表示。
    
    架构：
        - LSTM层：捕捉时序依赖关系
        - 多头自注意力层：建模全局特征交互
        - 残差连接：稳定训练过程
        - 全连接层：输出预测值
    """
    
    def __init__(self, input_size: int, d_model: int = 64, 
                 nhead: int = 4, num_encoder_layers: int = 1,
                 lstm_hidden_size: int = 128, lstm_layers: int = 2,
                 dropout: float = 0.2):
        """
        初始化Transformer-LSTM混合模型
        
        Args:
            input_size: 输入特征维度
            d_model: 模型维度（保留接口兼容性）
            nhead: 注意力头数
            num_encoder_layers: 编码器层数（保留接口兼容性）
            lstm_hidden_size: LSTM隐藏层维度
            lstm_layers: LSTM层数
            dropout: Dropout比率
        """
        super(TransformerLSTMModel, self).__init__()
        
        # LSTM编码层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        
        # 多头自注意力层
        self.self_attn = nn.MultiheadAttention(
            embed_dim=lstm_hidden_size,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        self.attn_norm = nn.LayerNorm(lstm_hidden_size)
        
        # 全连接输出层
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # LSTM编码
        lstm_out, _ = self.lstm(x)  # (batch, seq, hidden)
        
        # 自注意力增强（残差连接）
        attn_out, _ = self.self_attn(lstm_out, lstm_out, lstm_out)
        enhanced = self.attn_norm(lstm_out + attn_out)  # 残差连接
        
        # 使用最后时间步
        last_output = enhanced[:, -1, :]
        
        prediction = self.fc(last_output)
        return prediction.squeeze(-1)


# ================================================================================
# 带注意力机制的LSTM模型
# ================================================================================

class AttentionLSTMModel(nn.Module):
    """
    带注意力机制的LSTM模型
    
    在标准LSTM基础上引入时间注意力机制，自适应地学习不同时间步的重要性权重，
    通过加权融合历史信息与最近信息来提升预测性能。
    
    架构：
        - LSTM层：提取时序特征
        - 时间注意力层：计算各时间步的重要性权重
        - 加权融合：结合注意力上下文与最后时间步输出
        - 全连接层：输出预测值
    """
    
    def __init__(self, input_size: int, hidden_size: int = 128, 
                 num_layers: int = 2, dropout: float = 0.2):
        """
        初始化Attention-LSTM模型
        
        Args:
            input_size: 输入特征维度
            hidden_size: LSTM隐藏层维度
            num_layers: LSTM层数
            dropout: Dropout比率
        """
        super(AttentionLSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM编码层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 时间注意力层：学习各时间步的重要性权重
        self.time_attention = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        
        # 全连接输出层
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 (batch_size, sequence_length, input_size)
            
        Returns:
            预测值 (batch_size,)
        """
        # LSTM编码
        lstm_out, _ = self.lstm(x)  # (batch, seq, hidden)
        
        # 计算时间注意力权重
        attn_scores = self.time_attention(lstm_out)  # (batch, seq, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)
        
        # 注意力加权上下文向量
        context = torch.sum(attn_weights * lstm_out, dim=1)  # (batch, hidden)
        last_out = lstm_out[:, -1, :]  # (batch, hidden)
        
        # 融合长期上下文与短期信息
        combined = 0.7 * last_out + 0.3 * context
        
        prediction = self.fc(combined)
        return prediction.squeeze(-1)


# ================================================================================
# 模型工厂函数
# ================================================================================

def create_model(model_name: str, input_size: int, **kwargs) -> nn.Module:
    """
    创建模型的工厂函数
    
    Args:
        model_name: 模型名称 ('LSTM', 'GRU', 'Attention-LSTM', 'Transformer-LSTM')
        input_size: 输入特征维度
        **kwargs: 其他模型参数
        
    Returns:
        创建的模型实例
    """
    models = {
        'LSTM': LSTMModel,
        'GRU': GRUModel,
        'Attention-LSTM': AttentionLSTMModel,
        'Transformer-LSTM': TransformerLSTMModel
    }
    
    if model_name not in models:
        raise ValueError(f"未知模型: {model_name}. 可用模型: {list(models.keys())}")
    
    # 根据模型类型设置默认参数
    if model_name == 'Transformer-LSTM':
        default_kwargs = {
            'd_model': kwargs.get('d_model', 64),
            'nhead': kwargs.get('nhead', 4),
            'num_encoder_layers': kwargs.get('num_encoder_layers', 1),
            'lstm_hidden_size': kwargs.get('lstm_hidden_size', 64),
            'lstm_layers': kwargs.get('lstm_layers', 1),
            'dropout': kwargs.get('dropout', 0.1)
        }
        return models[model_name](input_size, **default_kwargs)
    elif model_name == 'Attention-LSTM':
        default_kwargs = {
            'hidden_size': kwargs.get('hidden_size', 128),
            'num_layers': kwargs.get('num_layers', 2),
            'dropout': kwargs.get('dropout', 0.1)
        }
        return models[model_name](input_size, **default_kwargs)
    else:
        default_kwargs = {
            'hidden_size': kwargs.get('hidden_size', 128),
            'num_layers': kwargs.get('num_layers', 2),
            'dropout': kwargs.get('dropout', 0.2)
        }
        return models[model_name](input_size, **default_kwargs)


def get_model_info(model: nn.Module) -> dict:
    """
    获取模型信息
    
    Args:
        model: 神经网络模型
        
    Returns:
        包含模型参数信息的字典
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'model_name': model.__class__.__name__
    }

