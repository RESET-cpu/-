#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
主程序入口
==========

基于神经网络的股票价格预测系统
支持多种运行模式：完整实验、单模型训练、生成对比图
"""

import os
import pickle
import numpy as np
import pandas as pd

# 导入自定义模块
from config import Config, DEVICE
from data_loader import StockDataLoader, create_data_loaders
from models import create_model, get_model_info
from trainer import StockPredictor
from utils import (
    plot_training_history, 
    plot_predictions, 
    plot_model_comparison,
    statistical_significance_test,
    print_experiment_summary
)


def prepare_data():
    """准备数据（公共函数）"""
    data_loader = StockDataLoader(
        ticker=Config.TICKER, 
        start_date=Config.START_DATE,
        end_date=Config.END_DATE
    )
    data_loader.fetch_data()
    data_loader.handle_missing_values()
    data_loader.add_technical_indicators()
    
    X, y, feature_columns = data_loader.prepare_sequences(
        target_col='Close',
        sequence_length=Config.SEQUENCE_LENGTH,
        prediction_horizon=Config.PREDICTION_HORIZON
    )
    
    train_loader, val_loader, test_loader, y_test = create_data_loaders(
        X, y,
        train_ratio=Config.TRAIN_RATIO,
        val_ratio=Config.VAL_RATIO,
        batch_size=Config.BATCH_SIZE
    )
    
    return data_loader, train_loader, val_loader, test_loader, y_test, X.shape[2]


def save_model_result(model_name: str, metrics: dict, predictions: np.ndarray, 
                      actuals: np.ndarray, save_dir: str):
    """
    保存单个模型的结果（用于后续生成对比图）
    """
    result_file = os.path.join(save_dir, f'{model_name}_result.pkl')
    result = {
        'model_name': model_name,
        'metrics': metrics,
        'predictions': predictions,
        'actuals': actuals
    }
    with open(result_file, 'wb') as f:
        pickle.dump(result, f)
    print(f"模型结果已保存至: {result_file}")


def load_model_results(save_dir: str) -> tuple:
    """
    加载所有已保存的模型结果
    """
    all_metrics = {}
    all_predictions = {}
    actuals = None
    
    model_names = ['LSTM', 'GRU', 'Attention-LSTM', 'Transformer-LSTM']
    
    for model_name in model_names:
        result_file = os.path.join(save_dir, f'{model_name}_result.pkl')
        if os.path.exists(result_file):
            with open(result_file, 'rb') as f:
                result = pickle.load(f)
            all_metrics[model_name] = result['metrics']
            all_predictions[model_name] = result['predictions']
            if actuals is None:
                actuals = result['actuals']
            print(f"已加载: {model_name}")
    
    return all_metrics, all_predictions, actuals


def run_experiment():
    """运行完整实验流程（训练所有模型）"""
    os.makedirs(Config.SAVE_DIR, exist_ok=True)
    
    print("="*80)
    print("基于神经网络的股票价格预测实验")
    print("="*80)
    Config.print_config()
    
    # 数据准备
    print("\n" + "="*60)
    print("第一阶段：数据准备")
    print("="*60)
    
    data_loader, train_loader, val_loader, test_loader, y_test, input_size = prepare_data()
    print(f"  - 输入特征维度: {input_size}")
    
    # 模型训练与评估
    print("\n" + "="*60)
    print("第二阶段：模型训练与评估")
    print("="*60)
    
    model_names = ['LSTM', 'GRU', 'Attention-LSTM', 'Transformer-LSTM']
    all_metrics = {}
    all_predictions = {}
    
    for model_name in model_names:
        print(f"\n{'='*40}")
        print(f"训练模型: {model_name}")
        print(f"{'='*40}")
        
        metrics, predictions, actuals = train_and_evaluate_model(
            model_name, input_size, data_loader,
            train_loader, val_loader, test_loader
        )
        
        all_metrics[model_name] = metrics
        all_predictions[model_name] = predictions
        
        # 保存结果
        save_model_result(model_name, metrics, predictions, actuals, Config.SAVE_DIR)
    
    # 模型对比分析
    generate_comparison(Config.SAVE_DIR, y_test)
    
    print("\n" + "="*80)
    print("实验完成！")
    print(f"结果文件保存在 {Config.SAVE_DIR} 目录下")
    print("="*80)
    
    return all_metrics, all_predictions


def train_and_evaluate_model(model_name: str, input_size: int, data_loader,
                             train_loader, val_loader, test_loader):
    """
    训练并评估单个模型
    """
    # 根据模型类型设置超参数
    if model_name in ['Attention-LSTM', 'Transformer-LSTM']:
        learning_rate = Config.LEARNING_RATE * 0.5
        patience = Config.PATIENCE + 10
    else:
        learning_rate = Config.LEARNING_RATE
        patience = Config.PATIENCE
    
    # 创建模型
    model = create_model(model_name, input_size)
    
    model_info = get_model_info(model)
    print(f"模型参数: 总计 {model_info['total_params']:,} | "
          f"可训练 {model_info['trainable_params']:,}")
    
    # 训练
    predictor = StockPredictor(model, device=DEVICE)
    history = predictor.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=Config.EPOCHS,
        learning_rate=learning_rate,
        patience=patience,
        scheduler_type='plateau'
    )
    
    # 评估
    predictions, actuals = predictor.predict(test_loader)
    metrics = predictor.evaluate(
        predictions=predictions,
        actuals=actuals,
        scaler=data_loader.scaler,
        feature_idx=-1
    )
    
    # 保存图表
    plot_training_history(
        history['train_losses'],
        history['val_losses'],
        save_path=os.path.join(Config.SAVE_DIR, f'{model_name}_training_history.png')
    )
    
    plot_predictions(
        actuals=actuals,
        predictions=predictions,
        title=f'{model_name} 模型预测结果',
        save_path=os.path.join(Config.SAVE_DIR, f'{model_name}_predictions.png')
    )
    
    return metrics, predictions, actuals


def run_single_model(model_name: str = 'LSTM', save_result: bool = True):
    """
    运行单个模型的实验
    
    Args:
        model_name: 模型名称
        save_result: 是否保存结果（用于后续生成对比图）
    """
    os.makedirs(Config.SAVE_DIR, exist_ok=True)
    
    print("="*80)
    print(f"运行单模型实验: {model_name}")
    print("="*80)
    
    # 数据准备
    data_loader, train_loader, val_loader, test_loader, y_test, input_size = prepare_data()
    
    # 训练和评估
    metrics, predictions, actuals = train_and_evaluate_model(
        model_name, input_size, data_loader,
        train_loader, val_loader, test_loader
    )
    
    # 保存结果
    if save_result:
        save_model_result(model_name, metrics, predictions, actuals, Config.SAVE_DIR)
        print(f"\n提示: 结果已保存。运行其他模型后，使用 --mode compare 生成对比图")
    
    return metrics


def generate_comparison(save_dir: str = None, y_test: np.ndarray = None):
    """
    从已保存的结果生成对比图
    """
    if save_dir is None:
        save_dir = Config.SAVE_DIR
    
    print("\n" + "="*60)
    print("生成模型对比分析")
    print("="*60)
    
    # 加载已保存的结果
    all_metrics, all_predictions, actuals = load_model_results(save_dir)
    
    if len(all_metrics) < 2:
        print(f"警告: 只找到 {len(all_metrics)} 个模型结果，至少需要2个才能对比")
        return
    
    print(f"\n已加载 {len(all_metrics)} 个模型的结果")
    
    # 生成对比图
    plot_model_comparison(
        all_metrics,
        save_path=os.path.join(save_dir, 'model_comparison.png')
    )
    
    # 统计显著性检验
    if y_test is None and actuals is not None:
        y_test = actuals
    
    if len(all_predictions) >= 2 and y_test is not None:
        significance_results = statistical_significance_test(all_predictions, y_test)
        significance_results.to_csv(
            os.path.join(save_dir, 'statistical_significance.csv'),
            index=False
        )
    
    # 保存汇总指标
    metrics_df = pd.DataFrame(all_metrics).T
    metrics_df.to_csv(os.path.join(save_dir, 'evaluation_metrics.csv'))
    
    # 打印摘要
    print_experiment_summary(all_metrics)
    
    print(f"\n对比图已保存至: {os.path.join(save_dir, 'model_comparison.png')}")


def list_results(save_dir: str = None):
    """列出已保存的模型结果"""
    if save_dir is None:
        save_dir = Config.SAVE_DIR
    
    print("\n已保存的模型结果:")
    print("-" * 40)
    
    model_names = ['LSTM', 'GRU', 'Attention-LSTM', 'Transformer-LSTM']
    found = 0
    
    for model_name in model_names:
        result_file = os.path.join(save_dir, f'{model_name}_result.pkl')
        if os.path.exists(result_file):
            with open(result_file, 'rb') as f:
                result = pickle.load(f)
            metrics = result['metrics']
            print(f"  ✓ {model_name:20s} | R²={metrics['R2']:.4f} | MAE={metrics['MAE']:.2f}")
            found += 1
        else:
            print(f"  ✗ {model_name:20s} | 未训练")
    
    print("-" * 40)
    print(f"共 {found}/4 个模型已完成训练")
    
    if found >= 2:
        print("\n提示: 可使用 --mode compare 生成对比图")


# ================================================================================
# 程序入口
# ================================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='股票价格预测神经网络实验',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python main.py                           # 运行完整实验（4个模型）
  python main.py --mode single --model LSTM    # 只训练LSTM
  python main.py --mode single --model GRU     # 只训练GRU
  python main.py --mode list                   # 查看已训练的模型
  python main.py --mode compare                # 生成对比图（需先训练模型）
        """
    )
    parser.add_argument('--mode', type=str, default='full', 
                       choices=['full', 'single', 'compare', 'list'],
                       help='运行模式: full(完整) / single(单模型) / compare(生成对比图) / list(查看结果)')
    parser.add_argument('--model', type=str, default='LSTM',
                       choices=['LSTM', 'GRU', 'Attention-LSTM', 'Transformer-LSTM'],
                       help='单模型模式下选择的模型')
    parser.add_argument('--ticker', type=str, default=None,
                       help='股票代码（如 AAPL, GOOGL）')
    parser.add_argument('--epochs', type=int, default=None,
                       help='训练轮数')
    parser.add_argument('--lr', type=float, default=None,
                       help='学习率')
    
    args = parser.parse_args()
    
    # 更新配置
    if args.ticker:
        Config.TICKER = args.ticker
    if args.epochs:
        Config.EPOCHS = args.epochs
    if args.lr:
        Config.LEARNING_RATE = args.lr
    
    # 运行实验
    if args.mode == 'full':
        run_experiment()
    elif args.mode == 'single':
        run_single_model(args.model)
    elif args.mode == 'compare':
        generate_comparison()
    elif args.mode == 'list':
        list_results()
