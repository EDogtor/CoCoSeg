# CoCoSeg V2

基于CoCoNet架构的医疗图像分割模型 - 完全重构的双独立编码器架构

## 📋 项目简介

CoCoSeg是一个专门用于CT-PET双模态医疗图像分割的深度学习模型。本项目将CoCoNet从**图像融合**模型改造为**医疗图像分割**模型，专门用于CT-PET双模态512×512像素PNG图像的分割任务。最新版本（V2）完全重构了架构，实现了双独立编码器和四路特征融合。

## 🎯 项目特点

### V2架构亮点

- ✅ **双独立UNet编码器**: CT和PET各一个独立的编码器，保持模态特异性
- ✅ **双VGG19辅助编码器**: 提取多尺度特征用于融合
- ✅ **四路中期融合**: CT_UNet + CT_VGG + PET_UNet + PET_VGG
- ✅ **MAM注意力机制**: CAM模块增强关键特征
- ✅ **InstanceNorm**: 解决小batch训练不稳定问题（V1的关键修复）
- ✅ **组合损失函数**: Dice + CE保证稳定收敛
- ✅ **完整测试工具**: 详细的模型评估和可视化

### 与V1的主要区别

| 特性 | V1 | V2 |
|------|-----|-----|
| 主编码器 | 1个共享 | **2个独立** ✓ |
| 融合路径 | 3路 | **4路** ✓ |
| 归一化层 | ❌ BatchNorm | ✅ **InstanceNorm** ✓ |
| 损失函数 | ❌ 单一Focal | ✅ **Combined** ✓ |
| 参数量 | ~11.4M | **~13.4M** |
| 模态特异性 | ❌ 较差 | ✅ **优秀** |
| 训练稳定性 | ❌ 差 | ✅ **改进** |

## 🏗️ 模型架构

### 数据流

```
输入:
├── CT [B,1,512,512] ──┬─> CT_UNet_Encoder ──> CT_UNet特征 (32,64,128,256ch)
│                      └─> CT_VGG_Encoder  ──> CT_VGG特征 (64,128,256ch)
│
└── PET [B,1,512,512] ─┬─> PET_UNet_Encoder ──> PET_UNet特征 (32,64,128,256ch)
                       └─> PET_VGG_Encoder  ──> PET_VGG特征 (64,128,256ch)

中期融合（256ch层级）:
CT_UNet(256) + CT_VGG(256) + PET_UNet(256) + PET_VGG(256)
    ↓ CAM注意力 ↓
        1024 → 256ch

中期融合（128ch层级）:
CT_UNet(128) + CT_VGG(128) + PET_UNet(128) + PET_VGG(128)
    ↓ CAM注意力 ↓
        512 → 128ch

中期融合（64ch层级）:
CT_UNet(64) + CT_VGG(64) + PET_UNet(64) + PET_VGG(64)
    ↓ CAM注意力 ↓
        256 → 64ch

解码器 + 跳跃连接 → 输出 [B,1,512,512]
```

### 为什么选择双独立编码器？

1. **模态特异性**: CT和PET的信息完全不同，应该分开学习
2. **避免特征混淆**: 早期融合可能导致特征混淆
3. **更丰富表征**: 4路特征比3路提供更多互补信息
4. **医疗影像最佳实践**: 多模态分割的推荐架构

### 模型参数量

- **总参数**: ~13.4M
- **可训练参数**: ~8.6M
- **VGG19编码器**: ~4.8M (冻结，预训练权重)

## 📂 目录结构

```
CoCoSeg/
├── main.py                    # 主训练/测试脚本
├── test_model.py              # 模型测试和评估脚本
├── requirements.txt           # 依赖列表
├── README.md                  # 本文档
├── models/
│   ├── model.py              # V2双独立编码器架构
│   ├── P_loss.py             # 损失函数
│   ├── segmentation_loss.py  # 分割损失
│   ├── train_tasks.py        # 训练任务
│   └── measure_model.py      # 模型参数量计算
├── data/
│   └── dataset.py            # CT-PET-Mask数据集加载器
├── utils/
│   ├── attention.py          # CAM注意力模块
│   ├── utils.py              # 工具函数
│   ├── visualizer.py         # 可视化工具
│   ├── checkpoint.py         # 模型保存/加载
│   ├── ema.py                # 指数移动平均
│   └── save_image.py         # 图像保存工具
├── configs/                   # 配置文件
│   ├── default_config.json   # 默认配置
│   ├── focal_loss_config.json # Focal损失配置
│   └── large_batch_config.json # 大批次配置
├── pytorch_ssim/              # SSIM损失
│   └── __init__.py
├── logs/                      # 训练日志和模型
│   ├── best_model.pth        # 最佳模型
│   ├── latest.pth            # 最新模型
│   ├── checkpoint_epoch_*.pth # 定期checkpoint
│   ├── history.json          # 训练历史
│   └── config.json           # 训练配置
└── archive/                   # 归档文件（旧版本、测试脚本等）
```

## 🚀 快速开始

### 环境配置

#### 1. 创建虚拟环境

```bash
# 使用conda创建环境（推荐）
conda create -n cocoseg python=3.11
conda activate cocoseg

# 或使用venv
python -m venv cocoseg_env
source cocoseg_env/bin/activate  # Linux/Mac
# 或 cocoseg_env\Scripts\activate  # Windows
```

#### 2. 安装PyTorch

根据您的CUDA版本选择合适的PyTorch：

```bash
# CUDA 12.1/12.4（推荐）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CPU版本
pip install torch torchvision torchaudio
```

#### 3. 安装其他依赖

```bash
cd CoCoSeg
pip install -r requirements.txt
```

### 数据集准备

#### 方式1：预分割数据集（推荐）

如果数据已经分割好，目录结构：

```
dataset_split/
├── train/
│   ├── CT/
│   │   ├── patient_001_slice_0001.png
│   │   └── ...
│   ├── PET/
│   │   ├── patient_001_slice_0001.png
│   │   └── ...
│   └── masks/
│       ├── patient_001_slice_0001.png
│       └── ...
├── val/
│   ├── CT/
│   ├── PET/
│   └── masks/
└── test/
    ├── CT/
    ├── PET/
    └── masks/
```

#### 方式2：运行时分割数据集

如果数据未分割，目录结构：

```
dataset/
├── CT/              # CT图像目录
│   ├── patient_001_slice_0001.png
│   ├── patient_001_slice_0002.png
│   └── ...
├── PET/             # PET图像目录
│   ├── patient_001_slice_0001.png
│   ├── patient_001_slice_0002.png
│   └── ...
└── masks/           # Mask标注目录
    ├── patient_001_slice_0001.png
    ├── patient_001_slice_0002.png
    └── ...
```

**重要要求**：
- 三个文件夹内的文件名必须完全一致
- 文件命名格式：`patient_{ID}_slice_{num}.png`
- PNG格式灰度图像，推荐尺寸512×512
- 数据集加载器会自动检测并使用预分割模式（如果存在）

### 训练模型

#### 使用预分割数据集（推荐）

```bash
python main.py --train --use_gpu --dataset_root ./dataset_split \
    --epoch 30 --bs 8 --lr 1e-4 --loss_type combined
```

#### 使用运行时分割数据集

```bash
python main.py --train --use_gpu --dataset_root ./dataset \
    --epoch 30 --bs 8 --lr 1e-4 --loss_type combined
```

#### 使用配置文件训练

```bash
python main.py --train --config configs/default_config.json \
    --dataset_root ./dataset_split --use_gpu
```

#### 恢复训练

```bash
python main.py --train --use_gpu --dataset_root ./dataset_split \
    --resume --resume_ckpt ./logs/checkpoint_epoch_10.pth
```

### 测试模型

#### 使用预分割数据集

```bash
# 测试验证集
python test_model.py --ckpt ./logs/best_model.pth --use_gpu \
    --mode eval --dataset_root ./dataset_split --split val --save_predictions

# 测试测试集
python test_model.py --ckpt ./logs/best_model.pth --use_gpu \
    --mode eval --dataset_root ./dataset_split --split test
```

#### 使用运行时分割数据集

```bash
python test_model.py --ckpt ./logs/best_model.pth --use_gpu \
    --mode eval --dataset_root ./dataset --split val --save_predictions
```

## 📊 训练配置

### 命令行参数

#### 基础参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--epoch` | int | 500 | 训练轮数 |
| `--lr` | float | 1e-4 | 学习率 |
| `--bs` | int | 8 | 批次大小 |
| `--dataset_root` | str | ./dataset | 数据集根目录 |
| `--logdir` | str | ./logs/ | 模型保存目录 |
| `--use_gpu` | flag | False | 使用GPU训练 |
| `--val_ratio` | float | 0.2 | 验证集比例（运行时分割） |
| `--random_seed` | int | 42 | 随机种子 |

#### 损失函数参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--loss_type` | str | combined | 损失类型：dice/ce/combined/iou/tversky/focal |
| `--dice_weight` | float | 0.5 | Dice损失权重（combined模式） |
| `--ce_weight` | float | 0.5 | 交叉熵损失权重（combined模式） |

#### 训练控制参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--val_freq` | int | 1 | 验证频率（每N个epoch验证一次） |
| `--save_freq` | int | 5 | checkpoint保存频率 |
| `--num_workers` | int | 4 | 数据加载线程数 |
| `--resume` | flag | False | 恢复训练 |
| `--resume_ckpt` | str | ./logs/latest.pth | 恢复训练的checkpoint路径 |

### 损失函数类型

#### 1. Combined Loss (推荐) ✓

**Dice Loss + CrossEntropy Loss**

```bash
--loss_type combined --dice_weight 0.5 --ce_weight 0.5
```

**优点**：
- 兼顾Dice的类别不平衡处理能力和CE的稳定性
- 医疗图像分割的标准配置
- 训练稳定，收敛快

#### 2. Dice Loss

专注于IoU优化，对小目标友好：

```bash
--loss_type dice
```

#### 3. CrossEntropy Loss

经典的分类损失：

```bash
--loss_type ce
```

#### 4. IoU Loss

直接优化IoU指标：

```bash
--loss_type iou
```

#### 5. Tversky Loss

可调整FP/FN权重：

```bash
--loss_type tversky
```

#### 6. Focal Loss

处理难易样本不平衡：

```bash
--loss_type focal
```

### 损失函数选择建议

| 场景 | 推荐损失函数 | 参数 |
|------|------------|------|
| **通用分割** | Combined | dice_weight=0.5, ce_weight=0.5 |
| 小目标分割 | Dice | - |
| 类别不平衡严重 | Combined | dice_weight=0.6, ce_weight=0.4 |
| 难样本多 | Focal | alpha=1.0, gamma=2.0 |
| FP更严重 | Tversky | alpha=0.6, beta=0.4 |

### 配置文件示例

`configs/default_config.json`:

```json
{
    "epoch": 50,
    "lr": 0.0001,
    "bs": 8,
    "loss_type": "combined",
    "dice_weight": 0.5,
    "ce_weight": 0.5,
    "val_freq": 1,
    "save_freq": 5,
    "num_workers": 4,
    "use_gpu": true,
    "val_ratio": 0.2,
    "random_seed": 42,
    "logdir": "./logs/",
    "dataset_root": "./dataset_split"
}
```

## 📈 训练监控

### 实时指标

训练过程中会显示：
- **Loss**: 当前损失值
- **Dice**: Dice系数（越高越好，范围[0,1]）
- **IoU**: 交并比（越高越好，范围[0,1]）
- **Acc**: 准确率（越高越好，范围[0,1]）

### 保存的文件

训练过程中会自动保存：

```
logs/
├── config.json              # 训练配置
├── history.json             # 训练历史（loss, dice, iou等）
├── best_model.pth           # 最佳模型（按验证Dice）
├── latest.pth               # 最新模型
├── checkpoint_epoch_5.pth   # 定期checkpoint
├── checkpoint_epoch_10.pth
└── tensorboard/             # TensorBoard日志
```

### 可视化训练曲线

使用`history.json`绘制训练曲线：

```python
import json
import matplotlib.pyplot as plt

with open('logs/history.json', 'r') as f:
    history = json.load(f)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.legend()
plt.title('Loss Curve')

plt.subplot(1, 2, 2)
plt.plot(history['train_dice'], label='Train Dice')
plt.plot(history['val_dice'], label='Val Dice')
plt.legend()
plt.title('Dice Score')
plt.tight_layout()
plt.savefig('training_curves.png')
```

## 📊 评估指标

模型支持以下评估指标：

- **Dice系数**: 衡量重叠度，范围[0,1]，越大越好
- **IoU**: 交并比，范围[0,1]，越大越好
- **准确率**: 正确像素比例，范围[0,1]，越大越好

### V1性能参考

- Dice: 0.504 ± 0.341
- IoU: 0.376
- 准确率: 0.997

**注**: V2性能待训练后更新

## 🔧 超参数调优

### 学习率调整

使用学习率调度器自动调整（代码中已实现余弦退火）：

```python
scheduler = CosineAnnealingLR(optimizer, T_max=epoch, eta_min=1e-6)
```

### 批次大小选择

根据GPU显存选择：

- **显存4-8GB**: bs=2-4
- **显存8-16GB**: bs=8
- **显存16GB+**: bs=16

### 损失权重调整

如果分割效果不好，可以调整权重：

```bash
--dice_weight 0.7 --ce_weight 0.3  # 增大Dice权重
```

## 🐛 常见问题

### Q1: GPU显存不足

**解决方法**：
```bash
# 减小batch size
--bs 2

# 减少num_workers
--num_workers 2
```

### Q2: 训练损失不下降

**可能原因**：
1. 学习率过大
2. 数据预处理问题
3. 模型输出范围不对

**解决方法**：
- 降低学习率到1e-5: `--lr 1e-5`
- 检查数据加载是否正常
- 确认mask标注正确

### Q3: 验证Dice不提升

**可能原因**：
1. 过拟合
2. 数据分布不一致
3. 验证集过小

**解决方法**：
- 增加数据增强
- 检查训练/验证集分布
- 增加验证集样本数

### Q4: 训练不稳定

**已修复**: V2使用InstanceNorm代替BatchNorm

如果仍有问题：
- 降低学习率: `--lr 5e-5`
- 确保使用combined损失函数
- 检查数据质量

### Q5: CUDA版本不兼容

**RTX 5090需要PyTorch CUDA 12.4+**：

```bash
# 升级PyTorch到支持CUDA 12.4的版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## 🔄 版本历史

### V2 (最新)

- ✅ 双独立UNet编码器
- ✅ 四路特征融合
- ✅ InstanceNorm（解决V1训练不稳定）
- ✅ Combined损失函数
- ✅ 改进的MAM模块
- ✅ 完整测试工具

### V1

- ❌ 单共享UNet编码器
- ❌ 三路特征融合
- ❌ BatchNorm导致训练不稳定
- ❌ 单一Focal损失
- ❌ Dice: 0.504（表现差）

## 📦 依赖项

### 核心依赖

- **PyTorch**: >=2.1.0 (推荐CUDA 12.1+)
- **torchvision**: >=0.16.0
- **torchaudio**: >=2.1.0
- **numpy**: >=1.24.0,<2.0.0
- **opencv-python**: >=4.8.0
- **pillow**: >=10.0.0
- **imageio**: >=2.31.0

### 深度学习工具

- **scikit-image**: >=0.21.0
- **scipy**: >=1.11.0
- **h5py**: >=3.10.0
- **pandas**: >=2.0.0

### 可视化工具

- **visdom**: >=0.2.0
- **tensorboard**: >=2.14.0
- **matplotlib**: >=3.8.0

### 图像处理

- **pywavelets**: >=1.4.0
- **albumentations**: >=1.3.0 (数据增强)

### 工具库

- **tqdm**: >=4.66.0
- **pyyaml**: >=6.0.1
- **requests**: >=2.31.0

完整依赖列表请查看 `requirements.txt`。

## 📝 最佳实践

1. ✅ 使用Combined Loss开始训练
2. ✅ 从小batch size开始测试
3. ✅ 监控验证指标而非训练指标
4. ✅ 定期保存checkpoint
5. ✅ 使用固定随机种子确保可复现
6. ✅ 根据验证集调整超参数

## 🤝 贡献

本项目基于CoCoNet修改，欢迎提出改进建议。

## 📄 许可证

本项目基于原CoCoNet代码修改，遵循MIT许可证。

## 🙏 致谢

- CoCoNet原论文: Liu et al., "CoCoNet: Coupled Contrastive Learning Network with Multi-level Feature Ensemble for Multi-modality Image Fusion", IJCV, 2024
- 原项目: https://github.com/runjia0124/CoCoNet

## 📧 联系方式

如有问题或建议，请提交Issue。
