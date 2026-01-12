#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练器模块
==========

包含模型训练、预测和评估功能
"""

import numpy as np
from typing import Tuple, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import DEVICE


# ================================================================================
# 训练器类
# ================================================================================

class StockPredictor:
    """
    股票预测模型训练器
    
    功能：
        - 模型训练
        - 验证与早停
        - 性能评估
    """
    
    def __init__(self, model: nn.Module, device: torch.device = DEVICE):
        """
        初始化训练器
        
        Args:
            model: 神经网络模型
            device: 计算设备
        """
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.MSELoss()
        self.train_losses = []
        self.val_losses = []
        self.best_model_state = None
        
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              epochs: int = 100, learning_rate: float = 0.001,
              patience: int = 15, min_delta: float = 1e-6,
              scheduler_type: str = 'plateau') -> Dict:
        """
        训练模型
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epochs: 训练轮数
            learning_rate: 学习率
            patience: 早停耐心值
            min_delta: 最小改进阈值
            scheduler_type: 学习率调度器类型 ('plateau' 或 'cosine')
            
        Returns:
            训练历史字典
        """
        optimizer = AdamW(self.model.parameters(), lr=learning_rate, 
                         weight_decay=1e-5)
        
        if scheduler_type == 'plateau':
            scheduler = ReduceLROnPlateau(optimizer, mode='min', 
                                         factor=0.5, patience=5)
        else:
            scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        print("\n" + "="*60)
        print("开始训练")
        print("="*60)
        
        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            train_batches = 0
            
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
                train_batches += 1
            
            avg_train_loss = train_loss / train_batches
            self.train_losses.append(avg_train_loss)
            
            # 验证阶段
            self.model.eval()
            val_loss = 0.0
            val_batches = 0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    outputs = self.model(batch_X)
                    loss = self.criterion(outputs, batch_y)
                    
                    val_loss += loss.item()
                    val_batches += 1
            
            avg_val_loss = val_loss / val_batches
            self.val_losses.append(avg_val_loss)
            
            if scheduler_type == 'plateau':
                scheduler.step(avg_val_loss)
            else:
                scheduler.step()
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch [{epoch+1:3d}/{epochs}] | "
                      f"Train Loss: {avg_train_loss:.6f} | "
                      f"Val Loss: {avg_val_loss:.6f} | "
                      f"LR: {current_lr:.6f}")
            
            if avg_val_loss < best_val_loss - min_delta:
                best_val_loss = avg_val_loss
                patience_counter = 0
                self.best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\n早停触发于第 {epoch+1} 轮")
                    break
        
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        print("\n训练完成!")
        print(f"最佳验证损失: {best_val_loss:.6f}")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': best_val_loss
        }
    
    def predict(self, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用模型进行预测
        
        Args:
            data_loader: 数据加载器
            
        Returns:
            predictions: 预测值数组
            actuals: 实际值数组
        """
        self.model.eval()
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for batch_X, batch_y in data_loader:
                batch_X = batch_X.to(self.device)
                outputs = self.model(batch_X)
                
                predictions.extend(outputs.cpu().numpy())
                actuals.extend(batch_y.numpy())
        
        return np.array(predictions), np.array(actuals)
    
    def evaluate(self, predictions: np.ndarray, actuals: np.ndarray,
                scaler: MinMaxScaler = None, feature_idx: int = -1) -> Dict:
        """
        评估模型性能
        
        Args:
            predictions: 预测值（归一化后）
            actuals: 实际值（归一化后）
            scaler: 归一化缩放器（用于反归一化）
            feature_idx: 目标特征的索引
            
        Returns:
            包含各项评估指标的字典
        """
        if scaler is not None:
            n_features = scaler.scale_.shape[0]
            pred_full = np.zeros((len(predictions), n_features))
            actual_full = np.zeros((len(actuals), n_features))
            
            pred_full[:, feature_idx] = predictions
            actual_full[:, feature_idx] = actuals
            
            pred_inverse = scaler.inverse_transform(pred_full)[:, feature_idx]
            actual_inverse = scaler.inverse_transform(actual_full)[:, feature_idx]
        else:
            pred_inverse = predictions
            actual_inverse = actuals
        
        mae = mean_absolute_error(actual_inverse, pred_inverse)
        mse = mean_squared_error(actual_inverse, pred_inverse)
        rmse = np.sqrt(mse)
        r2 = r2_score(actual_inverse, pred_inverse)
        
        mape = np.mean(np.abs((actual_inverse - pred_inverse) / 
                             (actual_inverse + 1e-10))) * 100
        
        if len(actual_inverse) > 1:
            actual_direction = np.sign(np.diff(actual_inverse))
            pred_direction = np.sign(np.diff(pred_inverse))
            direction_accuracy = np.mean(actual_direction == pred_direction) * 100
        else:
            direction_accuracy = 0.0
        
        metrics = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'MAPE': mape,
            'R2': r2,
            'Direction_Accuracy': direction_accuracy
        }
        
        print("\n" + "="*60)
        print("模型评估指标")
        print("="*60)
        for name, value in metrics.items():
            if name == 'R2':
                print(f"{name:20s}: {value:.6f}")
            elif name in ['MAPE', 'Direction_Accuracy']:
                print(f"{name:20s}: {value:.2f}%")
            else:
                print(f"{name:20s}: {value:.4f}")
        print("="*60)
        
        return metrics


# ================================================================================
# 超参数调优
# ================================================================================

def hyperparameter_tuning(X: np.ndarray, y: np.ndarray, 
                         input_size: int,
                         param_grid: Dict = None) -> Dict:
    """
    超参数调优
    
    Args:
        X: 输入特征
        y: 目标值
        input_size: 输入特征维度
        param_grid: 参数网格
        
    Returns:
        最佳参数和对应结果
    """
    from models import LSTMModel
    from data_loader import StockDataset
    
    if param_grid is None:
        param_grid = {
            'hidden_size': [64, 128, 256],
            'num_layers': [1, 2, 3],
            'learning_rate': [0.001, 0.0005, 0.0001]
        }
    
    print("\n" + "="*60)
    print("超参数调优")
    print("="*60)
    print(f"参数网格: {param_grid}")
    
    best_params = None
    best_val_loss = float('inf')
    results = []
    
    for hidden_size in param_grid['hidden_size']:
        for num_layers in param_grid['num_layers']:
            for lr in param_grid['learning_rate']:
                print(f"\n测试参数: hidden_size={hidden_size}, "
                      f"num_layers={num_layers}, lr={lr}")
                
                train_size = int(len(X) * 0.8)
                X_train, X_val = X[:train_size], X[train_size:]
                y_train, y_val = y[:train_size], y[train_size:]
                
                train_dataset = StockDataset(X_train, y_train)
                val_dataset = StockDataset(X_val, y_val)
                train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
                
                model = LSTMModel(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=0.2
                )
                
                predictor = StockPredictor(model)
                history = predictor.train(
                    train_loader=train_loader,
                    val_loader=val_loader,
                    epochs=30,
                    learning_rate=lr,
                    patience=10
                )
                
                val_loss = history['best_val_loss']
                results.append({
                    'hidden_size': hidden_size,
                    'num_layers': num_layers,
                    'learning_rate': lr,
                    'val_loss': val_loss
                })
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_params = {
                        'hidden_size': hidden_size,
                        'num_layers': num_layers,
                        'learning_rate': lr
                    }
    
    print(f"\n最佳参数: {best_params}")
    print(f"最佳验证损失: {best_val_loss:.6f}")
    
    return {'best_params': best_params, 'best_val_loss': best_val_loss, 
            'all_results': results}

