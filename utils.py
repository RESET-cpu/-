#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
工具函数模块
============

包含可视化和统计分析功能
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List
from scipy import stats


# ================================================================================
# 可视化函数
# ================================================================================

def plot_training_history(train_losses: List[float], val_losses: List[float],
                         save_path: str = None):
    """
    绘制训练历史曲线
    
    Args:
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        save_path: 保存路径
    """
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='训练损失', color='#2E86AB', linewidth=2)
    plt.plot(val_losses, label='验证损失', color='#E94F37', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('训练过程损失曲线', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='训练损失', color='#2E86AB', linewidth=2)
    plt.plot(val_losses, label='验证损失', color='#E94F37', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (log scale)', fontsize=12)
    plt.title('训练过程损失曲线（对数坐标）', fontsize=14)
    plt.yscale('log')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"训练历史图已保存至: {save_path}")
    
    plt.show()


def plot_predictions(actuals: np.ndarray, predictions: np.ndarray,
                    title: str = "股票价格预测结果",
                    save_path: str = None):
    """
    绘制预测结果对比图
    
    Args:
        actuals: 实际值数组
        predictions: 预测值数组
        title: 图表标题
        save_path: 保存路径
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. 时序对比图
    ax1 = axes[0, 0]
    ax1.plot(actuals, label='实际值', color='#2E86AB', linewidth=1.5, alpha=0.8)
    ax1.plot(predictions, label='预测值', color='#E94F37', linewidth=1.5, alpha=0.8)
    ax1.set_xlabel('样本索引', fontsize=11)
    ax1.set_ylabel('价格（归一化）', fontsize=11)
    ax1.set_title('预测值与实际值对比', fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. 散点图
    ax2 = axes[0, 1]
    ax2.scatter(actuals, predictions, alpha=0.5, color='#2E86AB', s=20)
    min_val = min(actuals.min(), predictions.min())
    max_val = max(actuals.max(), predictions.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='理想预测线')
    ax2.set_xlabel('实际值', fontsize=11)
    ax2.set_ylabel('预测值', fontsize=11)
    ax2.set_title('预测值 vs 实际值散点图', fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 3. 预测误差分布
    ax3 = axes[1, 0]
    errors = predictions - actuals
    ax3.hist(errors, bins=50, color='#2E86AB', alpha=0.7, edgecolor='white')
    ax3.axvline(x=0, color='#E94F37', linestyle='--', linewidth=2, label='零误差线')
    ax3.set_xlabel('预测误差', fontsize=11)
    ax3.set_ylabel('频次', fontsize=11)
    ax3.set_title(f'预测误差分布 (均值={errors.mean():.4f}, 标准差={errors.std():.4f})', fontsize=13)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # 4. 最近100个样本的详细对比
    ax4 = axes[1, 1]
    n_samples = min(100, len(actuals))
    ax4.plot(range(n_samples), actuals[-n_samples:], label='实际值', 
             color='#2E86AB', linewidth=2, marker='o', markersize=3)
    ax4.plot(range(n_samples), predictions[-n_samples:], label='预测值', 
             color='#E94F37', linewidth=2, marker='s', markersize=3)
    ax4.fill_between(range(n_samples), actuals[-n_samples:], predictions[-n_samples:],
                     alpha=0.3, color='gray')
    ax4.set_xlabel('样本索引', fontsize=11)
    ax4.set_ylabel('价格（归一化）', fontsize=11)
    ax4.set_title(f'最近{n_samples}个样本的预测对比', fontsize=13)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"预测结果图已保存至: {save_path}")
    
    plt.show()


def plot_model_comparison(metrics_dict: Dict[str, Dict], save_path: str = None):
    """
    绘制多模型性能对比图
    
    Args:
        metrics_dict: {模型名称: {指标名: 值}}
        save_path: 保存路径
    """
    models = list(metrics_dict.keys())
    metrics_names = ['MAE', 'RMSE', 'MAPE', 'R2', 'Direction_Accuracy']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    colors = ['#2E86AB', '#E94F37', '#F6BD60', '#84A98C', '#5C6B73']
    
    for idx, metric in enumerate(metrics_names):
        ax = axes[idx // 3, idx % 3]
        values = [metrics_dict[model].get(metric, 0) for model in models]
        
        bars = ax.bar(models, values, color=colors[:len(models)], alpha=0.8, 
                     edgecolor='white', linewidth=2)
        
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{val:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=10)
        
        ax.set_ylabel(metric, fontsize=11)
        ax.set_title(f'{metric} 对比', fontsize=13)
        ax.tick_params(axis='x', rotation=15)
        ax.grid(True, alpha=0.3, axis='y')
    
    # 隐藏最后一个空白子图并添加表格
    axes[1, 2].axis('off')
    ax_table = axes[1, 2]
    
    table_data = []
    for model in models:
        row = [model]
        for metric in ['MAE', 'RMSE', 'R2']:
            row.append(f"{metrics_dict[model].get(metric, 0):.4f}")
        table_data.append(row)
    
    table = ax_table.table(cellText=table_data,
                          colLabels=['模型', 'MAE', 'RMSE', 'R²'],
                          loc='center',
                          cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.5)
    
    plt.suptitle('模型性能对比分析', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"模型对比图已保存至: {save_path}")
    
    plt.show()


# ================================================================================
# 统计显著性分析
# ================================================================================

def statistical_significance_test(predictions_dict: Dict[str, np.ndarray],
                                  actuals: np.ndarray) -> pd.DataFrame:
    """
    进行统计显著性分析
    
    使用配对t检验和Wilcoxon符号秩检验比较不同模型的预测误差
    
    Args:
        predictions_dict: {模型名称: 预测值数组}
        actuals: 实际值数组
        
    Returns:
        显著性检验结果DataFrame
    """
    print("\n" + "="*60)
    print("统计显著性分析")
    print("="*60)
    
    models = list(predictions_dict.keys())
    results = []
    
    errors_dict = {model: np.abs(preds - actuals) 
                   for model, preds in predictions_dict.items()}
    
    for i, model1 in enumerate(models):
        for model2 in models[i+1:]:
            errors1 = errors_dict[model1]
            errors2 = errors_dict[model2]
            
            t_stat, t_pvalue = stats.ttest_rel(errors1, errors2)
            
            try:
                w_stat, w_pvalue = stats.wilcoxon(errors1, errors2)
            except ValueError:
                w_stat, w_pvalue = np.nan, np.nan
            
            diff = errors1 - errors2
            cohens_d = np.mean(diff) / (np.std(diff) + 1e-10)
            
            results.append({
                '模型对比': f'{model1} vs {model2}',
                't统计量': t_stat,
                't检验p值': t_pvalue,
                'Wilcoxon统计量': w_stat,
                'Wilcoxon p值': w_pvalue,
                "Cohen's d": cohens_d,
                '显著性(p<0.05)': '是' if t_pvalue < 0.05 else '否'
            })
    
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    return results_df


def print_experiment_summary(metrics_dict: Dict[str, Dict]):
    """
    打印实验结果摘要
    
    Args:
        metrics_dict: {模型名称: {指标名: 值}}
    """
    metrics_df = pd.DataFrame(metrics_dict).T
    
    print("\n" + "="*60)
    print("实验结果摘要")
    print("="*60)
    print(metrics_df.round(4).to_string())
    
    best_model_mae = metrics_df['MAE'].idxmin()
    best_model_rmse = metrics_df['RMSE'].idxmin()
    best_model_r2 = metrics_df['R2'].idxmax()
    
    print(f"\n最佳模型:")
    print(f"  - MAE最低: {best_model_mae} ({metrics_df.loc[best_model_mae, 'MAE']:.4f})")
    print(f"  - RMSE最低: {best_model_rmse} ({metrics_df.loc[best_model_rmse, 'RMSE']:.4f})")
    print(f"  - R²最高: {best_model_r2} ({metrics_df.loc[best_model_r2, 'R2']:.4f})")
    
    return metrics_df

