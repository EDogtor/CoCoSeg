# 配置文件说明

## 配置文件使用

本目录包含多个预定义的训练配置文件，您可以直接使用或根据自己的需求修改。

### 使用方法

#### 1. 使用默认配置

```bash
python main.py --train --config configs/default_config.json
```

#### 2. 覆盖部分参数

配置文件中的参数可以被命令行参数覆盖：

```bash
python main.py --train \
    --config configs/default_config.json \
    --dataset_root ./dataset_split \
    --bs 16  # 覆盖batch size
```

## 配置文件列表

### default_config.json
**标准配置**，适合大多数场景
- Epochs: 50
- Batch size: 8
- Loss: Dice+CE (0.5:0.5)
- 学习率: 1e-4

### large_batch_config.json
**大批次配置**，适合显存充足的情况
- Epochs: 50
- Batch size: 16
- Loss: Dice+CE (0.6:0.4) - 更侧重Dice
- 学习率: 2e-4

### focal_loss_config.json
**Focal Loss配置**，适合难易样本不平衡
- Epochs: 50
- Batch size: 8
- Loss: Focal Loss
- 学习率: 1e-4

## 自定义配置文件

您可以创建自己的配置文件：

```json
{
    "epoch": 100,
    "lr": 0.0001,
    "bs": 8,
    "loss_type": "combined",
    "dice_weight": 0.7,
    "ce_weight": 0.3,
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

## 参数说明

### 训练参数
- `epoch`: 训练轮数
- `lr`: 学习率
- `bs`: 批次大小

### 损失函数参数
- `loss_type`: 损失类型
  - `dice`: 仅使用Dice Loss
  - `ce`: 仅使用CrossEntropy Loss
  - `combined`: Dice + CE（推荐）
  - `iou`: IoU Loss
  - `tversky`: Tversky Loss
  - `focal`: Focal Loss
- `dice_weight`: Dice损失权重（当loss_type=combined时）
- `ce_weight`: CE损失权重（当loss_type=combined时）

### 其他参数
- `val_freq`: 每N个epoch验证一次
- `save_freq`: 每N个epoch保存一次checkpoint
- `num_workers`: 数据加载进程数
- `val_ratio`: 验证集比例（0.2 = 80%训练，20%验证）
- `random_seed`: 随机种子

## 推荐配置

### 新手入门
```bash
python main.py --train --config configs/default_config.json
```

### 快速实验
降低epoch和batch size：
```json
{
    "epoch": 10,
    "bs": 4
}
```

### 完整训练
使用大批次和更多epoch：
```json
{
    "epoch": 100,
    "bs": 16,
    "lr": 0.0002
}
```


