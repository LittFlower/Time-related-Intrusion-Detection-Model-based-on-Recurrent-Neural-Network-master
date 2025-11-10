# Time-related Intrusion Detection Model based on RNN

> 使用 **Stacked Sparse Autoencoder (SSAE)** 进行特征压缩，再配合多种 RNN 结构（LSTM / GRU / 双向变体）完成对 UNSW-NB15 数据集的时间序列入侵检测。

![Framework](figure/framework.png)

## 目录

- [简介](#简介)
- [核心特性](#核心特性)
- [项目结构](#项目结构)
- [环境准备](#环境准备)
- [数据准备](#数据准备)
- [训练流程](#训练流程)
- [评估与可视化](#评估与可视化)
- [推理与服务](#推理与服务)
- [常见问题](#常见问题)
- [引用](#引用)

## 简介

整个流程可拆分为两个阶段：

1. **预训练阶段**  
   通过 SSAE 将 196 维的流量特征降维至 32 维，并导出可复用的编码器权重。  
   ![SSAE](figure/Sparse%20AE.png)

2. **序列建模阶段**  
   利用 `TimeseriesGenerator` 将编码后的数据组织成时间窗口，再使用单/双向 LSTM、GRU 等结构进行二分类。  
   ![LSTM](figure/LSTM.png) ![GRU](figure/GRU.png)

最终能得到随时间变化的告警曲线，用于对比真实标签与模型预测：  
![Prediction Wave](figure/wave_1.png)

## 核心特性

- **SSAE + RNN**：先用稀疏自编码器完成无监督特征学习，再用序列模型捕捉时间依赖。
- **可复用的预处理资产**：`feature_columns.json`、`scaler.pkl`、`saved_ae_*` 等文件让部署与二次训练更方便。
- **多脚本拆分**：`data_generator.py`、`classifier.py`、`evaluate_classifier.py`、`plot_wave_testing.py` 分别负责预训练、fine-tune、评估和可视化。
- **内置推理服务**：`app.py` 提供 Flask REST API 和简单 Web UI，可直接加载训练好的模型上线。

## 项目结构

```
.
├── app.py                     # Flask 推理服务
├── build_model.py             # SSAE 结构定义
├── classifier.py              # RNN 分类器训练脚本
├── data/                      # 数据集与中间产物 (需自行放置 UNSW-NB15 CSV)
├── data_generator.py          # SSAE 预训练 + 特征导出
├── data_processing.py         # 数据读取与归一化
├── evaluate_classifier.py     # 载入 best_model 进行评估
├── models/                    # 训练得到的 best_model.hdf5
├── plot_wave_testing.py       # 预测曲线可视化
├── preprocess_artifacts.py    # 生成 scaler 与特征列
├── requirements.txt           # 依赖清单
└── smoke_test_app.py          # 对 Flask 服务做快速回归测试
```

## 环境准备

建议使用 Pyenv + venv 管理 Python 版本。本仓库已经在 `.python-version` 中锁定 **Python 3.10.14**。

```bash
# 1. 安装并选择项目 Python 版本
pyenv install 3.10.14           # 若本地尚未安装
pyenv local 3.10.14

# 2. 创建并启用虚拟环境
python -m venv .venv
source .venv/bin/activate

# 3. 安装依赖
pip install -U pip
pip install -r requirements.txt

# 4. 验证安装
pip check
python -c "import tensorflow as tf; print(tf.__version__)"
```

> 如需 GPU 训练，请确保正确安装 CUDA/cuDNN，并让 TensorFlow 能检测到 GPU。`app.py` 会自动按需开启 GPU 显存按需增长。

## 数据准备

1. 从 [UNSW-NB15 官方页面](https://research.unsw.edu.au/projects/unsw-nb15-dataset) 下载 `UNSW_NB15_training-set.csv` 与 `UNSW_NB15_testing-set.csv`，放到仓库的 `data/` 目录。
2. 执行一次特征对齐与缩放器生成：

   ```bash
   source .venv/bin/activate
   python preprocess_artifacts.py
   ```

   会得到：
   - `data/feature_columns.json`
   - `data/scaler.pkl`

这些文件会被训练脚本、评估脚本以及 Flask 推理服务共享。

## 训练流程

### 1. 预训练 SSAE 并导出特征

```bash
python data_generator.py
```

该脚本会：

- 按照 `build_SAE` 定义为 3 层 SSAE 逐层预训练，并把最优权重保存在 `saved_ae_1/2/3`.
- 将原始特征编码到 32 维潜空间，分别保存 `data/encoded_train.npy`、`data/encoded_test.npy`。
- 暂存一个以 SSAE + LSTM 组成的基线分类器到 `saved_models_temp/best_model.hdf5`，供后续 fine-tune 使用。

### 2. 训练最终的 RNN 分类器

```bash
python classifier.py
```

脚本会读取 `encoded_*.npy`，使用 `TimeseriesGenerator` 生成滑动窗口，然后训练一个双向 LSTM 头部，最优模型保存到 `models/best_model.hdf5` 并输出混淆矩阵。训练过程会把 TensorBoard 日志写入 `logs/`。

如需尝试不同的时间步长、cell 类型或层数，可以直接修改脚本中对应的参数。

### 3.（可选）继续 fine-tune / grid search

你可以复制 `classifier.py`，将 LSTM 换为单向 LSTM、GRU、双向 GRU，或调整 `time_steps` 等超参，再结合 `evaluate_classifier.py` 获取指标，制作论文中的对比表格/折线图。

> 如需可视化训练过程，可在训练期间打开另一个终端运行：  
> `tensorboard --logdir logs --port 6006`

## 实验目标与复现指南

### 1. 不同 DAE 结构对分类效果的影响

1. 在 `build_model.py` 中修改 SSAE 的层数、隐层神经元数（如 256-64-32）以及稀疏系数 `rho`。
2. 运行 `python data_generator.py`，观察 `saved_ae_*/best_ae_*.hdf5` 的验证损失，并记录最终在 `saved_models_temp/best_model.hdf5` 上的验证准确率。
3. 将不同组合的 (rho, layer_size) 与对应指标整理成表格，即 README 原始描述中所需的 Table。

建议指标：`val_loss`（压缩重建质量）与 `val_accuracy`（基线分类器性能）。

### 2. 不同 RNN 结构与时间步长对性能影响

1. 在 `classifier.py` 内切换为以下四种结构之一：
   - 单向 LSTM
   - 单向 GRU
   - 双向 LSTM（默认实现）
   - 双向 GRU
2. 修改 `time_steps`（例如 1/3/5/7）并重新训练。
3. 执行 `python evaluate_classifier.py --time_steps <k>`（或在脚本内调用 `main(k)`）获取 `accuracy`、`precision`、`false alarm rate (FAR)` 等指标。
4. 以时间步长为横轴，绘制 4 条曲线对比准确率与 FAR，得到 README 描述中的折线图。

> 提示：当 `time_steps > 1` 时，`evaluate_classifier.py` 会自动偏移标签以对齐序列窗口，确保曲线数据有效。

### 3. 最优结构的测试曲线

1. 选定最优的 SSAE+RNN 参数组合后，确保 `models/best_model.hdf5`、`data/plot_prediction.npy`、`data/plot_original.npy` 均由该组合生成。
2. 运行 `python plot_wave_testing.py` 绘制最终对比图和累计错误图，可直接保存图像用于论文或汇报。

## 数据与模型产物速查

| 文件/目录 | 作用 |
|-----------|------|
| `data/UNSW_NB15_*.csv` | 原始训练/测试数据（需手动下载） |
| `data/feature_columns.json` | 记录 One-hot 后的特征列顺序 |
| `data/scaler.pkl` | `MinMaxScaler` 对象，保证推理阶段与训练一致 |
| `saved_ae_{1,2,3}/best_ae_*.hdf5` | SSAE 各层的最佳权重 |
| `saved_models_temp/best_model.hdf5` | 仅供预训练阶段验证使用的临时分类器 |
| `models/best_model.hdf5` | 最终部署使用的 RNN 分类器 |
| `data/encoded_{train,test}.npy` | 32 维潜空间特征，供下游训练/评估 |
| `data/plot_{prediction,original}.npy` | 评估后保存的预测/真实标签，用于绘图 |
| `logs/` | TensorBoard 日志，包含 loss/accuracy/Summary |

## 评估与可视化

1. **定量评估**

   ```bash
    python evaluate_classifier.py          # 默认为 time_steps=1, batch_size=1024
    # 或者
    python -c "import evaluate_classifier as ec; ec.main(time_steps=5, batch_size=512)"
   ```

   - 输出训练/测试集的混淆矩阵与 `classification_report`
   - 将 `plot_prediction.npy` 与 `plot_original.npy` 写到 `data/`
   - 保存文本报告到 `data/eval_report.txt`

2. **可视化预测波形**

   ```bash
   python plot_wave_testing.py
   ```

   第一个图展示真实标签、正确/错误预测点；第二个图为累计错误数，帮助判断模型在时间维度上的表现。

## 推理与服务

### 本地运行 Flask

```bash
source .venv/bin/activate
python app.py        # 默认监听 0.0.0.0:5000
```

服务启动时会自动加载：

- `models/best_model.hdf5`
- `saved_ae_*/best_ae_*.hdf5`
- `data/feature_columns.json` 与 `data/scaler.pkl`

若缺少这些文件，请先完成训练流程。

### REST API

- `GET /health`：返回模型加载状态。
- `POST /predict_encoded`：输入编码后的样本（32 维）或序列，返回概率与预测结果。

示例：

```bash
curl -X POST http://localhost:5000/predict_encoded \
  -H "Content-Type: application/json" \
  -d '{"vector": [0.12, 0.03, ..., 0.7], "threshold": 0.5}'
```

当你有未编码的原始 CSV，可在浏览器访问 `http://localhost:5000/`，上传 CSV 或使用页面提供的表单完成推理。`smoke_test_app.py` 可在无浏览器环境下快速验证接口：

```bash
python smoke_test_app.py
```

## 常见问题

- **FileNotFoundError: feature_columns.json / scaler.pkl**  
  运行 `python preprocess_artifacts.py` 并确认路径位于 `data/`。

- **Missing saved_ae or best_model.hdf5**  
  按顺序执行 `data_generator.py` 与 `classifier.py`。

- **TensorFlow 报错找不到 GPU**  
  本项目默认使用 CPU，如需 GPU 请正确安装 CUDA/cuDNN，并确保 `nvidia-smi` 与 `tf.config.list_physical_devices('GPU')` 可以找到设备。

- **内存不足**  
  可将 `batch_size` 调小（例如 `512` 或 `256`），或在 `data_generator.py` 中减少训练轮数。

## 引用

如果该方法对你的论文/项目有所帮助，请引用：

```
Lin Y, Wang J, Tu Y, et al. Time-Related Network Intrusion Detection Model:
A Deep Learning Method[C]//2019 IEEE Global Communications Conference (GLOBECOM). IEEE, 2019: 1-6.
```

如需进一步交流，请联系 heuwangjie@hrbeu.edu.cn。谢谢！
