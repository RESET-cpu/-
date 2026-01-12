## 股票收盘价预测（LSTM / GRU / Attention-LSTM / Transformer-LSTM）

本项目提供一个可复现的深度学习时间序列预测基线，用于**基于历史 OHLCV 与技术指标序列预测下一交易日收盘价**，并支持多模型对比、可视化与统计显著性检验。

---

## 功能特性

- **端到端流程**：读取本地 CSV → 缺失值处理 → 技术指标特征工程 → 归一化 → 序列样本构造 → 训练/验证/测试划分 → 训练评估 → 结果落盘
- **多模型对比**：`LSTM`、`GRU`、`Attention-LSTM`、`Transformer-LSTM`
- **训练增强**：早停、学习率调度（ReduceLROnPlateau）、梯度裁剪、AdamW
- **可视化输出**：训练曲线、预测对比、误差分布、模型对比图
- **统计检验**：配对 t 检验、Wilcoxon 符号秩检验、Cohen’s d（效果量）

---

## 快速开始

### 环境要求

- Python ≥ 3.8

### 安装依赖

```bash
pip install -r requirements.txt
```

### 准备数据

- 默认读取项目根目录下的 `AAPL_data.csv`
- 若使用其它标的，请放置 `${TICKER}_data.csv` 到项目根目录，并在运行时传入 `--ticker ${TICKER}`

数据来源示例：[Kaggle “Price Volume Data for All US Stocks & ETFs”](https://www.kaggle.com/datasets/borismarjanovic/price-volume-data-for-all-us-stocks-etfs)。

### 一键运行完整对比实验

```bash
python main.py
```

运行完成后，所有结果会输出到 `results/`。

---

## 使用方式（命令行）

### 完整实验（训练 4 个模型 + 生成对比与统计检验）

```bash
python main.py --mode full
```

### 单模型训练

```bash
python main.py --mode single --model GRU
```

### 仅生成对比图与显著性检验（需要 `results/*_result.pkl` 已存在）

```bash
python main.py --mode compare
```

### 查看已保存的结果

```bash
python main.py --mode list
```

### 常用参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--mode` | `full` | `full` / `single` / `compare` / `list` |
| `--model` | `LSTM` | `LSTM` / `GRU` / `Attention-LSTM` / `Transformer-LSTM` |
| `--ticker` | `AAPL` | 股票代码，对应读取 `${TICKER}_data.csv` |
| `--epochs` | `150` | 最大训练轮数 |
| `--lr` | `0.001` | 学习率（部分模型会在程序中自动减半） |

---

## 方法概览

### 任务定义

使用长度为 \(L=60\) 的时间窗口作为输入序列，预测未来 \(H=1\) 个交易日后的收盘价（回归）。

### 特征工程

在 OHLCV 基础上构建技术指标特征，最终特征维度为 25（含 `Close`），包括 MA、RSI、MACD、布林带、波动率、成交量变化、收益率等（详见 `data_loader.py`）。

### 训练策略（与实现一致）

- 损失函数：MSE
- 优化器：AdamW（`weight_decay=1e-5`）
- 梯度裁剪：`max_norm=1.0`
- 学习率调度：ReduceLROnPlateau（factor=0.5，patience=5）
- 早停：默认 `Config.PATIENCE=25`；复杂模型自动增加耐心值
- 可复现性：默认随机种子 42（见 `config.py`）

---

## 结果（示例：AAPL）

以下表格来自 `results/evaluation_metrics.csv`（示例环境的运行结果，供快速参考；不同设备/随机性/数据区间会导致数值变化）。

| 模型 | MAE ↓ | RMSE ↓ | MAPE ↓ | \(R^2\) ↑ | 方向准确率 ↑ |
|------|------:|-------:|-------:|----------:|-------------:|
| LSTM | 7.8936 | 9.7872 | 5.2558% | 0.7257 | 49.4662% |
| GRU | **4.5961** | **5.4647** | **3.1451%** | **0.9145** | 50.1779% |
| Attention-LSTM | 6.2759 | 7.5354 | 4.2603% | 0.8374 | **54.8043%** |
| Transformer-LSTM | 9.6897 | 12.1394 | 6.4290% | 0.5780 | 50.5338% |

显著性检验结果（配对 t 检验 + Wilcoxon + Cohen’s d）可见 `results/statistical_significance.csv`。

> 说明：当前实现中，显著性检验基于**归一化空间**的绝对误差 \(|\hat{y}-y|\)，而 MAE/RMSE/MAPE/\(R^2\) 在评估时会先反归一化到价格空间（见 `trainer.py`）。

---

## 输出文件说明（`results/`）

- **`evaluation_metrics.csv`**：MAE/MSE/RMSE/MAPE/\(R^2\)/Direction Accuracy
- **`statistical_significance.csv`**：配对 t 检验、Wilcoxon 检验、Cohen’s d
- **`*_training_history.png`**：训练/验证损失曲线（含对数坐标）
- **`*_predictions.png`**：预测 vs 实际、误差分布、局部对比
- **`model_comparison.png`**：多模型指标对比图
- **`*_result.pkl`**：缓存预测与指标，便于后续只生成对比分析

---

## 项目结构

```
.
├── main.py              # 程序入口：训练/评估/对比
├── config.py            # 配置、随机种子、设备选择
├── data_loader.py       # 数据加载、特征工程、序列构造、划分
├── models.py            # 模型定义（LSTM/GRU/Attention/Transformer-LSTM）
├── trainer.py           # 训练器：训练/预测/指标评估
├── utils.py             # 可视化与统计显著性分析
├── requirements.txt
├── README.md
├── AAPL_data.csv
└── results/
```

---

## 参考

- LSTM: Hochreiter S, Schmidhuber J. Long short-term memory. Neural Computation, 1997.
- GRU: Cho K, et al. Learning phrase representations using RNN encoder-decoder. arXiv:1406.1078.
- Attention/Transformer: Vaswani A, et al. Attention is all you need. NeurIPS, 2017.
