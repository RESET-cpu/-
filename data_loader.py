#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据加载模块
============

包含股票数据获取、预处理和特征工程功能
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Tuple, List
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader


# ================================================================================
# 股票数据加载器
# ================================================================================

class StockDataLoader:
    """
    股票数据加载器
    
    功能：
        - 从本地CSV文件加载股票数据
        - 数据清洗（缺失值处理、异常值检测）
        - 特征工程（技术指标计算）
    """
    
    def __init__(self, ticker: str = "AAPL", start_date: str = "2015-01-01", 
                 end_date: str = None):
        """
        初始化数据加载器
        
        Args:
            ticker: 股票代码（如 AAPL, GOOGL, MSFT）
            start_date: 开始日期
            end_date: 结束日期（默认为今天）
        """
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime("%Y-%m-%d")
        self.data = None
        self.scaler = MinMaxScaler()
        
    def fetch_data(self, csv_path: str = None) -> pd.DataFrame:
        """
        从本地CSV文件获取股票数据
        
        Args:
            csv_path: CSV文件路径（如果为None，则使用项目目录下的默认文件）
        
        Returns:
            包含股票数据的DataFrame
        """
        # 确定数据文件路径
        if csv_path and os.path.exists(csv_path):
            data_file = csv_path
        else:
            project_dir = os.path.dirname(os.path.abspath(__file__))
            data_file = os.path.join(project_dir, f"{self.ticker}_data.csv")
        
        # 检查文件是否存在
        if not os.path.exists(data_file):
            raise FileNotFoundError(
                f"数据文件不存在: {data_file}\n"
                f"请将股票数据CSV文件放置在项目目录下，命名为 {self.ticker}_data.csv"
            )
        
        return self._load_from_csv(data_file)
    
    def _load_from_csv(self, csv_path: str) -> pd.DataFrame:
        """
        从CSV文件加载股票数据
        
        Args:
            csv_path: CSV文件路径
            
        Returns:
            股票数据DataFrame
        """
        print(f"从本地文件读取数据: {csv_path}")
        
        # 读取CSV文件
        self.data = pd.read_csv(csv_path, index_col='Date', parse_dates=True)
        
        # 标准化列名（处理大小写不一致的情况）
        col_mapping = {
            'open': 'Open', 'high': 'High', 'low': 'Low', 
            'close': 'Close', 'volume': 'Volume',
            'openint': 'OpenInt', 'adj close': 'Adj_Close'
        }
        new_columns = []
        for col in self.data.columns:
            new_col = col_mapping.get(col.lower().strip(), col)
            new_columns.append(new_col)
        self.data.columns = new_columns
        
        # 删除不需要的列
        columns_to_drop = ['OpenInt', 'Adj_Close']
        for col in columns_to_drop:
            if col in self.data.columns:
                self.data = self.data.drop(col, axis=1)
        
        # 确保数据按日期排序
        self.data = self.data.sort_index()
        
        # 应用日期过滤
        if self.start_date:
            self.data = self.data[self.data.index >= self.start_date]
        if self.end_date:
            self.data = self.data[self.data.index <= self.end_date]
        
        print(f"成功读取 {len(self.data)} 条数据记录")
        print(f"数据时间范围: {self.data.index[0].date()} 至 {self.data.index[-1].date()}")
        print(f"数据列: {list(self.data.columns)}")
        
        return self.data
    
    def handle_missing_values(self) -> pd.DataFrame:
        """处理缺失值"""
        if self.data is None:
            raise ValueError("请先调用 fetch_data() 获取数据")
        
        missing_count = self.data.isnull().sum()
        if missing_count.sum() > 0:
            print(f"缺失值统计:\n{missing_count[missing_count > 0]}")
            self.data = self.data.ffill()
            self.data = self.data.bfill()
            self.data = self.data.dropna()
            print(f"缺失值处理完成，剩余 {len(self.data)} 条数据")
        else:
            print("数据无缺失值")
        
        return self.data
    
    def add_technical_indicators(self) -> pd.DataFrame:
        """
        添加技术指标作为特征
        
        包含的技术指标：
            1. 移动平均线 (MA5, MA10, MA20, MA50)
            2. 相对强弱指标 (RSI)
            3. MACD指标
            4. 布林带 (Bollinger Bands)
            5. 成交量变化率
            6. 价格变化率
            7. 波动率
        """
        if self.data is None:
            raise ValueError("请先调用 fetch_data() 获取数据")
        
        df = self.data.copy()
        
        # 1. 移动平均线
        for period in [5, 10, 20, 50]:
            df[f'MA{period}'] = df['Close'].rolling(window=period).mean()
        
        # 2. 相对强弱指标 (RSI)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # 3. MACD指标
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        # 4. 布林带
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + 2 * bb_std
        df['BB_Lower'] = df['BB_Middle'] - 2 * bb_std
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        
        # 5. 成交量变化率
        df['Volume_Change'] = df['Volume'].pct_change()
        df['Volume_MA5'] = df['Volume'].rolling(window=5).mean()
        
        # 6. 价格变化率
        df['Price_Change'] = df['Close'].pct_change()
        df['Price_Change_5d'] = df['Close'].pct_change(periods=5)
        
        # 7. 波动率 (20日滚动标准差)
        df['Volatility'] = df['Close'].rolling(window=20).std()
        df['Volatility_Ratio'] = df['Volatility'] / df['Close']
        
        # 8. 最高价/最低价比率
        df['HL_Ratio'] = df['High'] / (df['Low'] + 1e-10)
        
        # 9. 收盘价与开盘价差异
        df['CO_Diff'] = (df['Close'] - df['Open']) / (df['Open'] + 1e-10)
        
        # 删除NaN值
        df = df.dropna()
        
        # 替换无穷值为0
        df = df.replace([np.inf, -np.inf], 0)
        
        self.data = df
        print(f"添加技术指标后，数据维度: {df.shape}")
        print(f"特征列表: {list(df.columns)}")
        
        return self.data
    
    def prepare_sequences(self, target_col: str = 'Close', 
                         sequence_length: int = 60,
                         prediction_horizon: int = 1) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        准备时间序列数据用于模型训练
        
        Args:
            target_col: 预测目标列
            sequence_length: 输入序列长度
            prediction_horizon: 预测未来多少天
            
        Returns:
            X: 输入特征序列 (samples, sequence_length, features)
            y: 目标值 (samples,)
            feature_columns: 特征列名列表
        """
        if self.data is None:
            raise ValueError("请先调用 fetch_data() 获取数据")
        
        feature_columns = [col for col in self.data.columns if col != target_col]
        feature_columns.append(target_col)
        
        data_array = self.data[feature_columns].values
        
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = self.scaler.fit_transform(data_array)
        
        X, y = [], []
        for i in range(sequence_length, len(scaled_data) - prediction_horizon + 1):
            X.append(scaled_data[i - sequence_length:i])
            y.append(scaled_data[i + prediction_horizon - 1, -1])
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"序列数据准备完成:")
        print(f"  - 输入形状: {X.shape}")
        print(f"  - 输出形状: {y.shape}")
        print(f"  - 序列长度: {sequence_length}")
        print(f"  - 特征数量: {X.shape[2]}")
        
        return X, y, feature_columns


# ================================================================================
# 数据集类
# ================================================================================

class StockDataset(Dataset):
    """股票数据集类（PyTorch Dataset）"""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        初始化数据集
        
        Args:
            X: 输入特征 (samples, sequence_length, features)
            y: 目标值 (samples,)
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


# ================================================================================
# 辅助函数
# ================================================================================

def create_data_loaders(X: np.ndarray, y: np.ndarray, 
                       train_ratio: float = 0.7,
                       val_ratio: float = 0.15,
                       batch_size: int = 32) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建数据加载器
    
    Args:
        X: 输入特征
        y: 目标值
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        batch_size: 批次大小
        
    Returns:
        train_loader, val_loader, test_loader
    """
    train_size = int(len(X) * train_ratio)
    val_size = int(len(X) * val_ratio)
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]
    X_test = X[train_size + val_size:]
    y_test = y[train_size + val_size:]
    
    print(f"\n数据集划分:")
    print(f"  - 训练集: {len(X_train)} 样本")
    print(f"  - 验证集: {len(X_val)} 样本")
    print(f"  - 测试集: {len(X_test)} 样本")
    
    train_dataset = StockDataset(X_train, y_train)
    val_dataset = StockDataset(X_val, y_val)
    test_dataset = StockDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, y_test

