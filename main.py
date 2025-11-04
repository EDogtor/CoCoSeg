import os
import torch
import cv2
import argparse
import numpy as np
import imageio
import json
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from scipy.ndimage import distance_transform_edt

from models.model import Unet_resize_conv
from utils import fname_presuffix
from utils.ema import EMA
from data.dataset import LungSegmentationDataset, PCLT20KDataset
from models.segmentation_loss import CombinedSegLoss, DiceLoss, get_loss_function
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # change the CUDA index in your need

# 创建参数解析器
argparser = argparse.ArgumentParser(description='CoCoSeg: Medical Image Segmentation')
argparser.add_argument('--epoch', type=int, help='epoch number', default=500)
argparser.add_argument('--lr', type=float, help='learning rate', default=1e-4)
argparser.add_argument('--bs', type=int, help='batch size', default=8)
argparser.add_argument('--logdir', type=str, default='./logs/', help='checkpoint directory')
argparser.add_argument('--dataset_root', type=str, default='./dataset', help='dataset root directory')
argparser.add_argument('--train', action='store_true', help='train mode')
argparser.add_argument('--test', action='store_true', help='test mode')
argparser.add_argument('--test_ct', type=str, help='Directory of the test CT images')
argparser.add_argument('--test_pet', type=str, help='Directory of the test PET images')
argparser.add_argument('--resume', action='store_true', help='resume training')
argparser.add_argument('--resume_ckpt', type=str, default='./logs/latest.pth', help='checkpoint to resume from')
argparser.add_argument('--ckpt', type=str, default='./logs/latest.pth', help='checkpoint for testing')
argparser.add_argument('--use_gpu', action='store_true', help='use GPU')
argparser.add_argument('--save_dir', type=str, default='./results/', help='directory to save results')
argparser.add_argument('--val_ratio', type=float, default=0.2, help='validation ratio')
argparser.add_argument('--random_seed', type=int, default=42, help='random seed for data split')

# 新增：损失函数配置
argparser.add_argument('--loss_type', type=str, default='combined', 
                       choices=['dice', 'ce', 'combined', 'iou', 'tversky', 'focal'],
                       help='loss function type')
argparser.add_argument('--dice_weight', type=float, default=0.7, help='weight of dice loss')
argparser.add_argument('--ce_weight', type=float, default=0.3, help='weight of ce loss')
argparser.add_argument('--pos_weight', type=float, default=5.0, 
                       help='positive class weight for BCE loss (used for class imbalance)')

# 新增：训练配置
argparser.add_argument('--val_freq', type=int, default=1, help='validation frequency (epochs)')
argparser.add_argument('--save_freq', type=int, default=50, help='save frequency (epochs)')
argparser.add_argument('--num_workers', type=int, default=4, help='number of data loading workers')
argparser.add_argument('--config', type=str, help='config file path (JSON format)')
argparser.add_argument('--dataset_type', type=str, default='auto',
                       choices=['auto', 'pclt', 'pclt20k'],
                       help='dataset type: auto (auto-detect), pclt (old format), or pclt20k (new format)')

args = argparser.parse_args()

# 如果有配置文件，从文件加载参数
if args.config and os.path.exists(args.config):
    print(f"Loading config from {args.config}")
    with open(args.config, 'r') as f:
        config = json.load(f)
    for key, value in config.items():
        if hasattr(args, key):
            setattr(args, key, value)


class GrayscaleTransform:
    def __call__(self, img):
        # Convert the image to grayscale
        if img.shape[0] == 3:
            img = img[0, :, :]
            img = torch.unsqueeze(img, 0)
        return img


def calculate_hd95(pred_binary, target_binary):
    """计算95% Hausdorff距离（使用距离变换优化）
    
    Args:
        pred_binary: 预测二值mask [H, W] (numpy array, uint8)
        target_binary: 真实二值mask [H, W] (numpy array, uint8)
    
    Returns:
        hd95: 95% Hausdorff距离 (float)
    """
    from scipy.ndimage import binary_erosion
    
    # 检查是否有有效的mask
    if pred_binary.sum() == 0 and target_binary.sum() == 0:
        return 0.0
    if pred_binary.sum() == 0 or target_binary.sum() == 0:
        # 如果一个为空，返回一个大的惩罚值
        return float(max(pred_binary.shape))
    
    # 提取边界点
    def get_boundary_points(binary_mask):
        """提取二值mask的边界点坐标（使用腐蚀操作）"""
        boundary = binary_mask.astype(bool) & (~binary_erosion(binary_mask))
        coords = np.array(np.where(boundary)).T
        return coords
    
    # 获取边界点
    pred_boundary = get_boundary_points(pred_binary)
    target_boundary = get_boundary_points(target_binary)
    
    if len(pred_boundary) == 0 or len(target_boundary) == 0:
        return float(max(pred_binary.shape))
    
    # 使用距离变换计算从预测边界到目标mask的距离
    # 计算到目标mask的距离变换（距离背景到前景的距离）
    target_inv = (~target_binary.astype(bool)).astype(float)
    dt_target = distance_transform_edt(target_inv)
    
    # 获取预测边界点到目标mask的距离
    distances_pred_to_target = dt_target[pred_boundary[:, 0], pred_boundary[:, 1]]
    
    # 使用距离变换计算从目标边界到预测mask的距离
    pred_inv = (~pred_binary.astype(bool)).astype(float)
    dt_pred = distance_transform_edt(pred_inv)
    
    # 获取目标边界点到预测mask的距离
    distances_target_to_pred = dt_pred[target_boundary[:, 0], target_boundary[:, 1]]
    
    # 合并所有距离
    all_distances = np.concatenate([distances_pred_to_target, distances_target_to_pred])
    
    if len(all_distances) == 0:
        return 0.0
    
    # 计算95百分位数
    hd95 = np.percentile(all_distances, 95)
    
    return float(hd95)


def calculate_metrics(pred, target, threshold=0.5):
    """计算评估指标 (修改为按样本平均)
    
    Args:
        pred: 预测值 [B, 1, H, W]
        target: 真实值 [B, 1, H, W]
        threshold: 二值化阈值，默认0.5
    
    Returns:
        dice, iou, f1_score, hd95
    """
    # 二值化预测 [B, 1, H, W]
    pred_binary = (torch.sigmoid(pred) > threshold).float()
    
    batch_size = pred_binary.shape[0]
    
    # 按样本计算 (dim=(1, 2, 3) -> [B])
    intersection = (pred_binary * target).sum(dim=(1, 2, 3))
    union_sum = pred_binary.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    union_intersect = union_sum - intersection
    
    # Dice, IoU (per-sample)
    dice_per_sample = (2. * intersection + 1e-6) / (union_sum + 1e-6)
    iou_per_sample = (intersection + 1e-6) / (union_intersect + 1e-6)
    
    # F1 (per-sample)
    tp = intersection
    fp = pred_binary.sum(dim=(1, 2, 3)) - tp
    fn = target.sum(dim=(1, 2, 3)) - tp
    
    precision_per_sample = (tp + 1e-6) / (tp + fp + 1e-6)
    recall_per_sample = (tp + 1e-6) / (tp + fn + 1e-6)
    f1_per_sample = 2 * (precision_per_sample * recall_per_sample) / (precision_per_sample + recall_per_sample + 1e-6)
    
    # 取平均
    dice = dice_per_sample.mean()
    iou = iou_per_sample.mean()
    f1_score = f1_per_sample.mean()
    
    # 计算95% Hausdorff距离（对batch中每个样本分别计算后取平均）
    # (这部分原先就是按样本平均的，保持不变)
    hd95_list = []
    
    for i in range(batch_size):
        pred_np = pred_binary[i, 0].cpu().numpy().astype(np.uint8)
        target_np = target[i, 0].cpu().numpy().astype(np.uint8)
        
        # 修复一个潜在bug：如果pred_np或target_np是空的，hd95计算会出错
        if pred_np.sum() == 0 and target_np.sum() == 0:
            hd95_list.append(0.0)
            continue
        elif pred_np.sum() == 0 or target_np.sum() == 0:
            # 如果只有一个是空的，HD95没有明确定义，这里给一个惩罚值（图像对角线长度）
            hd95_list.append(float(np.sqrt(pred_np.shape[0]**2 + pred_np.shape[1]**2)))
            continue
            
        hd95 = calculate_hd95(pred_np, target_np)
        hd95_list.append(hd95)
    
    hd95_mean = torch.tensor(np.mean(hd95_list), dtype=torch.float32)
    
    return dice, iou, f1_score, hd95_mean


def threshold_sweep(logits, masks, thresholds=None):
    """阈值扫描，找到最佳Dice对应的阈值
    
    Args:
        logits: 模型输出的logits [B, 1, H, W]
        masks: 真实mask [B, 1, H, W]
        thresholds: 要扫描的阈值列表，默认为 0.20~0.60，步长0.05
    
    Returns:
        best_dice: 最佳Dice值
        best_threshold: 对应的最佳阈值
        threshold_results: 所有阈值的结果字典 {threshold: dice}
    """
    if thresholds is None:
        thresholds = np.arange(0.20, 0.61, 0.05)
    
    # 转换为概率
    sig = torch.sigmoid(logits)
    
    best_dice = -1.0
    best_threshold = 0.5
    threshold_results = {}
    
    for t in thresholds:
        # 使用阈值进行二值化
        pred = (sig > t).float()
        
        # 计算每个样本的Dice（按batch维度）
        # pred: [B, 1, H, W], masks: [B, 1, H, W]
        intersection = (pred * masks).sum(dim=(1, 2, 3))  # [B]
        union = pred.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))  # [B]
        dice_per_sample = (2. * intersection / (union + 1e-6))  # [B]
        dice = dice_per_sample.mean().item()  # 平均Dice
        
        threshold_results[float(t)] = dice
        
        if dice > best_dice:
            best_dice = dice
            best_threshold = float(t)
    
    return best_dice, best_threshold, threshold_results


def train_epoch(model, train_loader, optimizer, criterion, device, epoch, ema=None):
    """训练一个epoch
    
    Args:
        model: 模型
        train_loader: 训练数据加载器
        optimizer: 优化器
        criterion: 损失函数
        device: 设备
        epoch: 当前epoch
        ema: EMA对象（可选）
    """
    model.train()
    total_loss = 0
    total_dice = 0
    total_iou = 0
    total_f1 = 0
    total_hd95 = 0
    
    # 使用tqdm显示进度条
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1} [Train]', ncols=100)
    
    for batch_idx, batch in enumerate(pbar):
        ct = batch['ct'].to(device)
        pet = batch['pet'].to(device)
        mask = batch['mask'].to(device)
        
        optimizer.zero_grad()
        
        # 前向传播
        output = model(ct, pet)
        
        # 计算损失
        if isinstance(criterion, CombinedSegLoss):
            loss, dice_loss, ce_loss = criterion(output, mask)
        else:
            loss = criterion(output, mask)
        
        loss.backward()
        optimizer.step()
        
        # 更新EMA（如果启用）
        if ema is not None:
            ema.update()
        
        # 计算指标
        dice, iou, f1_score, hd95 = calculate_metrics(output, mask)
        
        total_loss += loss.item()
        total_dice += dice.item()
        total_iou += iou.item()
        total_f1 += f1_score.item()
        total_hd95 += hd95.item()
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'dice': f'{dice.item():.3f}',
            'f1': f'{f1_score.item():.3f}',
            'hd95': f'{hd95.item():.2f}'
        })
    
    # 平均指标
    avg_loss = total_loss / len(train_loader)
    avg_dice = total_dice / len(train_loader)
    avg_iou = total_iou / len(train_loader)
    avg_f1 = total_f1 / len(train_loader)
    avg_hd95 = total_hd95 / len(train_loader)
    
    return avg_loss, avg_dice, avg_iou, avg_f1, avg_hd95


def validate(model, val_loader, criterion, device, use_threshold_sweep=True):
    """验证函数（支持阈值扫描）
    
    Args:
        model: 模型
        val_loader: 验证数据加载器
        criterion: 损失函数
        device: 设备
        use_threshold_sweep: 是否使用阈值扫描
    
    Returns:
        如果 use_threshold_sweep=True:
            avg_loss, best_dice, best_threshold, avg_dice, avg_iou, avg_f1, avg_hd95, threshold_results
            - avg_loss: 平均损失
            - best_dice: 阈值扫描的最佳Dice值
            - best_threshold: 对应的最佳阈值
            - avg_dice: 使用0.5阈值的平均Dice值（用于对比）
            - avg_iou: 平均IoU
            - avg_f1: 平均F1分数
            - avg_hd95: 平均95% Hausdorff距离
            - threshold_results: 所有阈值扫描结果的字典 {threshold: dice}
        否则:
            avg_loss, avg_dice, avg_iou, avg_f1, avg_hd95
    """
    model.eval()
    total_loss = 0
    
    # 用于阈值扫描：收集所有logits和masks
    all_logits = []
    all_masks = []
    
    # 用于传统指标计算（使用0.5阈值）
    total_dice = 0
    total_iou = 0
    total_f1 = 0
    total_hd95 = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation', ncols=100)
        for batch in pbar:
            ct = batch['ct'].to(device)
            pet = batch['pet'].to(device)
            mask = batch['mask'].to(device)
            
            output = model(ct, pet)
            
            # 计算损失
            if isinstance(criterion, CombinedSegLoss):
                loss, _, _ = criterion(output, mask)
            else:
                loss = criterion(output, mask)
            
            total_loss += loss.item()
            
            # 收集logits和masks用于阈值扫描
            if use_threshold_sweep:
                all_logits.append(output.cpu())
                all_masks.append(mask.cpu())
            
            # 使用0.5阈值计算传统指标
            dice, iou, f1_score, hd95 = calculate_metrics(output, mask, threshold=0.5)
            total_dice += dice.item()
            total_iou += iou.item()
            total_f1 += f1_score.item()
            total_hd95 += hd95.item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'dice': f'{dice.item():.3f}',
                'f1': f'{f1_score.item():.3f}',
                'hd95': f'{hd95.item():.2f}'
            })
    
    # 平均损失和传统指标
    avg_loss = total_loss / len(val_loader)
    avg_dice = total_dice / len(val_loader)
    avg_iou = total_iou / len(val_loader)
    avg_f1 = total_f1 / len(val_loader)
    avg_hd95 = total_hd95 / len(val_loader)
    
    if use_threshold_sweep:
        # 合并所有logits和masks
        all_logits = torch.cat(all_logits, dim=0)  # [N, 1, H, W]
        all_masks = torch.cat(all_masks, dim=0)     # [N, 1, H, W]
        
        # 进行阈值扫描
        best_dice, best_threshold, threshold_results = threshold_sweep(all_logits, all_masks)
        
        print(f"\n[阈值扫描] 最佳 Dice: {best_dice:.4f} @ 阈值: {best_threshold:.2f}")
        print(f"[阈值扫描] 传统指标 (阈值=0.5): Dice={avg_dice:.4f}, IoU={avg_iou:.4f}, F1={avg_f1:.4f}, HD95={avg_hd95:.2f}")
        
        # 返回：损失, 最佳Dice, 最佳阈值, 0.5阈值Dice, IoU, F1, HD95, 阈值结果
        return avg_loss, best_dice, best_threshold, avg_dice, avg_iou, avg_f1, avg_hd95, threshold_results
    else:
        return avg_loss, avg_dice, avg_iou, avg_f1, avg_hd95


def test(args, model, ct_path, pet_path, save_path):
    """测试函数"""
    checkpath = args.ckpt
    print('Loading from {}...'.format(checkpath))

    ct_list = [n for n in os.listdir(ct_path) if n.endswith('.png')]
    pet_list = [n for n in os.listdir(pet_path) if n.endswith('.png')]

    # 确保CT和PET列表匹配
    ct_list = sorted(ct_list)
    pet_list = sorted(pet_list)
    
    if len(ct_list) != len(pet_list):
        print(f"Warning: CT files ({len(ct_list)}) and PET files ({len(pet_list)}) count mismatch")

    device = torch.device('cpu')
    if args.use_gpu:
        device = torch.device('cuda')
    logs = torch.load(checkpath, map_location=device)
    model.load_state_dict(logs['state_dict'])
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        GrayscaleTransform()
    ])

    import time
    Time = []
    pbar = tqdm(zip(ct_list, pet_list), total=len(ct_list), desc='Testing')
    for ct_file, pet_file in pbar:
        fn_pet = os.path.join(pet_path, pet_file)
        fn_ct = os.path.join(ct_path, ct_file)
        start = time.time()

        img_ct = imageio.imread(fn_ct).astype(np.float32)
        img_pet = imageio.imread(fn_pet).astype(np.float32)

        # to tensor and grayscale
        data_ct = transform(img_ct)
        data_pet = transform(img_pet)

        # add batch size dimension
        data_ct = torch.unsqueeze(data_ct, 0).to(device)
        data_pet = torch.unsqueeze(data_pet, 0).to(device)
        
        with torch.no_grad():
            output = model(data_ct, data_pet)

        # 转换为numpy并保存
        output_np = (torch.sigmoid(output) > 0.5).float().squeeze().cpu().numpy()
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        save_fn = fname_presuffix(
            fname=ct_file, prefix='', suffix='', newpath=save_path)
        cv2.imwrite(save_fn.split('.')[0] + '.png', output_np * 255)

        end = time.time()
        Time.append(end - start)

    print("Time: mean:%.3fs, std: %.3fs" % (np.mean(Time), np.std(Time)))


def save_config(args, filepath):
    """保存训练配置"""
    config_dict = vars(args)
    with open(filepath, 'w') as f:
        json.dump(config_dict, f, indent=4)


def detect_dataset_type(dataset_root, split='train'):
    """自动检测数据集类型
    
    Args:
        dataset_root: 数据集根目录
        split: 要检测的split（train/val/test）
        
    Returns:
        'pclt' 或 'pclt20k'
    """
    split_dir = os.path.join(dataset_root, split)
    
    if not os.path.exists(split_dir):
        # 如果没有split目录，检查根目录
        split_dir = dataset_root
    
    # 检查是否是PCLT20K格式：有病人文件夹（数字命名的文件夹）
    # 例如：train/0001/, train/0002/ 等
    if os.path.exists(split_dir):
        patient_dirs = [d for d in os.listdir(split_dir) 
                        if os.path.isdir(os.path.join(split_dir, d)) 
                        and not d.startswith('.')]
        
        # 检查是否有数字命名的文件夹（PCLT20K格式）
        if patient_dirs:
            # 检查第一个文件夹中是否有文件
            first_patient_dir = os.path.join(split_dir, patient_dirs[0])
            files = [f for f in os.listdir(first_patient_dir) 
                    if f.endswith('.png') and not f.startswith('.')]
            
            if files:
                # 检查文件名格式：病人id_切片编号_模态.png
                sample_file = files[0]
                parts = os.path.splitext(sample_file)[0].split('_')
                if len(parts) >= 3:
                    # 可能是PCLT20K格式
                    return 'pclt20k'
    
    # 检查是否是旧格式：有CT/PET/masks子目录
    ct_dir = os.path.join(split_dir, 'CT')
    pet_dir = os.path.join(split_dir, 'PET')
    masks_dir = os.path.join(split_dir, 'masks')
    
    if os.path.exists(ct_dir) and os.path.exists(pet_dir) and os.path.exists(masks_dir):
        return 'pclt'
    
    # 默认返回pclt（向后兼容）
    return 'pclt'


def main():
    print('\n' + '='*60)
    print('CoCoSeg: Modified CoCoNet for Medical Image Segmentation')
    print('='*60)
    print(f'Cuda available: {torch.cuda.is_available()}')
    print(f'Mode: {"Training" if args.train else "Testing"}')
    print('='*60)

    # 设置设备（带CPU fallback机制）
    if args.use_gpu and torch.cuda.is_available():
        try:
            # 尝试使用GPU
            device = torch.device('cuda')
            # 测试GPU是否可用
            test_tensor = torch.randn(2, 2).to(device)
            _ = test_tensor @ test_tensor.T
            print(f"✓ 使用GPU: {torch.cuda.get_device_name(0)}")
            model = Unet_resize_conv().to(device)
        except RuntimeError as e:
            print(f"⚠ GPU初始化失败: {e}")
            print("⚠ 自动降级到CPU模式")
            device = torch.device('cpu')
            model = Unet_resize_conv().to(device)
    else:
        device = torch.device('cpu')
        if args.use_gpu:
            print("⚠ CUDA不可用，使用CPU模式")
        model = Unet_resize_conv().to(device)

    # 计算参数量
    tmp = filter(lambda x: x.requires_grad, model.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(f'Total trainable parameters: {num:,}')

    if args.train:
        # 创建数据集
        print("\n创建数据集...")
        
        # 检测数据集类型
        if args.dataset_type == 'auto':
            dataset_type = detect_dataset_type(args.dataset_root, split='train')
            print(f"自动检测数据集类型: {dataset_type}")
        else:
            dataset_type = args.dataset_type
            print(f"使用指定的数据集类型: {dataset_type}")
        
        # 根据数据集类型选择合适的数据集类
        if dataset_type == 'pclt20k':
            DatasetClass = PCLT20KDataset
            print("使用 PCLT20KDataset")
        else:
            DatasetClass = LungSegmentationDataset
            print("使用 LungSegmentationDataset")
        
        train_dataset = DatasetClass(
            dataset_root=args.dataset_root,
            split='train',
            val_ratio=args.val_ratio,
            random_seed=args.random_seed,
            augment=True  # 训练集启用数据增强
        )
        
        val_dataset = DatasetClass(
            dataset_root=args.dataset_root,
            split='val',
            val_ratio=args.val_ratio,
            random_seed=args.random_seed,
            augment=False  # 验证集不使用数据增强
        )
        
        print(f"训练集: {len(train_dataset)}个样本")
        print(f"验证集: {len(val_dataset)}个样本")
        
        # 创建DataLoader
        train_loader = DataLoader(
            train_dataset, 
            batch_size=args.bs, 
            shuffle=True, 
            num_workers=args.num_workers,
            pin_memory=True if args.use_gpu else False
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=args.bs, 
            shuffle=False, 
            num_workers=args.num_workers,
            pin_memory=True if args.use_gpu else False
        )
        
        # 创建损失函数
        print(f"\n使用损失函数: {args.loss_type}")
        if args.loss_type == 'combined':
            criterion = CombinedSegLoss(
                dice_weight=args.dice_weight,
                ce_weight=args.ce_weight,
                pos_weight=args.pos_weight
            )
            print(f"  正样本权重 (pos_weight): {args.pos_weight}")
        else:
            criterion = get_loss_function(args.loss_type)
        criterion = criterion.to(device)
        
        # 优化器 (使用AdamW，设置weight_decay=1e-4)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        
        # 创建EMA对象（默认衰减率0.999）
        ema = EMA(model, decay=0.999, device=device)
        print("EMA机制已启用（衰减率: 0.999）")
        
        # 学习率调度器
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epoch, eta_min=1e-6
        )
        
        # 创建日志目录
        if not os.path.exists(args.logdir):
            os.makedirs(args.logdir)
        
        # 保存配置
        save_config(args, os.path.join(args.logdir, 'config.json'))
        
        # 创建TensorBoard writer
        tensorboard_dir = os.path.join(args.logdir, 'tensorboard')
        writer = SummaryWriter(tensorboard_dir)
        print(f"TensorBoard日志目录: {tensorboard_dir}")
        print(f"启动TensorBoard: tensorboard --logdir {tensorboard_dir}")
        
        # 训练历史
        history = {
            'train_loss': [], 'train_dice': [], 'train_iou': [], 'train_f1': [], 'train_hd95': [],
            'val_loss': [], 'val_dice': [], 'val_iou': [], 'val_f1': [], 'val_hd95': [],
            'val_best_dice': [], 'val_best_threshold': []  # 阈值扫描结果
        }
        
        best_val_dice = 0  # 使用阈值扫描的最佳Dice
        best_val_threshold = 0.5  # 最佳阈值
        start_epoch = 0
        
        # 恢复训练
        if args.resume:
            if os.path.exists(args.resume_ckpt):
                print(f"\n恢复训练从: {args.resume_ckpt}")
                checkpoint = torch.load(args.resume_ckpt)
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                start_epoch = checkpoint.get('epoch', 0)
                if 'history' in checkpoint:
                    history = checkpoint['history']
                if 'best_val_dice' in checkpoint:
                    best_val_dice = checkpoint['best_val_dice']
                if 'best_val_threshold' in checkpoint:
                    best_val_threshold = checkpoint['best_val_threshold']
                # 恢复学习率调度器状态
                if 'scheduler' in checkpoint:
                    scheduler.load_state_dict(checkpoint['scheduler'])
                    print(f"已恢复学习率调度器状态")
                # 恢复EMA状态
                if 'ema' in checkpoint:
                    ema.load_state_dict(checkpoint['ema'])
                    print(f"已恢复EMA状态")
                print(f"从第 {start_epoch} 个epoch继续训练")
                print(f"恢复的最佳模型: Dice={best_val_dice:.4f} @ 阈值={best_val_threshold:.2f}")
        
        # 训练循环
        print("\n开始训练...")
        print(f"配置: loss={args.loss_type}, dice_weight={args.dice_weight}, ce_weight={args.ce_weight}")
        
        for epoch in range(start_epoch, args.epoch):
            # 训练
            train_loss, train_dice, train_iou, train_f1, train_hd95 = train_epoch(
                model, train_loader, optimizer, criterion, device, epoch, ema=ema
            )
            
            history['train_loss'].append(train_loss)
            history['train_dice'].append(train_dice)
            history['train_iou'].append(train_iou)
            history['train_f1'].append(train_f1)
            history['train_hd95'].append(train_hd95)
            
            # 记录训练指标到TensorBoard
            writer.add_scalar('Train/Loss', train_loss, epoch)
            writer.add_scalar('Train/Dice', train_dice, epoch)
            writer.add_scalar('Train/IoU', train_iou, epoch)
            writer.add_scalar('Train/F1', train_f1, epoch)
            writer.add_scalar('Train/HD95', train_hd95, epoch)
            writer.add_scalar('Train/LearningRate', scheduler.get_last_lr()[0], epoch)
            
            # 验证（使用阈值扫描）
            if (epoch + 1) % args.val_freq == 0:
                # 使用EMA模型进行验证
                ema.apply_shadow()  # 应用EMA参数
                val_loss, best_dice, best_threshold, val_dice_05, val_iou, val_f1, val_hd95, threshold_results = validate(
                    model, val_loader, criterion, device, use_threshold_sweep=True
                )
                ema.restore()  # 恢复原始参数
                
                # val_dice_05是0.5阈值的Dice，best_dice是阈值扫描的最佳Dice
                history['val_loss'].append(val_loss)
                history['val_dice'].append(val_dice_05)  # 保存0.5阈值的Dice用于对比
                history['val_iou'].append(val_iou)
                history['val_f1'].append(val_f1)
                history['val_hd95'].append(val_hd95)
                history['val_best_dice'].append(best_dice)  # 阈值扫描的最佳Dice
                history['val_best_threshold'].append(best_threshold)  # 最佳阈值
                
                # 记录验证指标到TensorBoard
                writer.add_scalar('Val/Loss', val_loss, epoch)
                writer.add_scalar('Val/Dice', val_dice_05, epoch)
                writer.add_scalar('Val/IoU', val_iou, epoch)
                writer.add_scalar('Val/F1', val_f1, epoch)
                writer.add_scalar('Val/HD95', val_hd95, epoch)
                writer.add_scalar('Val/BestDice', best_dice, epoch)
                writer.add_scalar('Val/BestThreshold', best_threshold, epoch)
                
                print(f'\nEpoch {epoch+1}/{args.epoch} 完成:')
                print(f'  Train - Loss: {train_loss:.4f}, Dice: {train_dice:.4f}, IoU: {train_iou:.4f}, F1: {train_f1:.4f}, HD95: {train_hd95:.2f}')
                print(f'  Val   - Loss: {val_loss:.4f}, Dice(0.5): {val_dice_05:.4f}, IoU: {val_iou:.4f}, F1: {val_f1:.4f}, HD95: {val_hd95:.2f}')
                print(f'  Val   - 最佳Dice: {best_dice:.4f} @ 阈值: {best_threshold:.2f}')
                
                # 保存最佳模型（使用阈值扫描的最佳Dice）
                if best_dice > best_val_dice:
                    best_val_dice = best_dice
                    best_val_threshold = best_threshold
                    best_path = os.path.join(args.logdir, 'best_model.pth')
                    # 保存EMA参数作为最佳模型
                    ema.apply_shadow()  # 应用EMA参数
                    torch.save({
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),  # 保存EMA参数
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'ema': ema.state_dict(),  # 保存EMA状态
                        'val_dice': val_dice_05,  # 0.5阈值的Dice
                        'val_best_dice': best_dice,  # 阈值扫描的最佳Dice
                        'val_best_threshold': best_threshold,  # 最佳阈值
                        'threshold_results': threshold_results,  # 所有阈值的结果
                        'history': history,
                        'best_val_dice': best_val_dice,
                        'best_val_threshold': best_val_threshold
                    }, best_path)
                    ema.restore()  # 恢复原始参数用于继续训练
                    print(f'  ✓ 保存最佳模型: 最佳Dice={best_dice:.4f} @ 阈值={best_threshold:.2f}')
            else:
                print(f'\nEpoch {epoch+1}/{args.epoch} 完成:')
                print(f'  Train - Loss: {train_loss:.4f}, Dice: {train_dice:.4f}, IoU: {train_iou:.4f}, F1: {train_f1:.4f}, HD95: {train_hd95:.2f}')
            
            # 保存定期checkpoint
            if (epoch + 1) % args.save_freq == 0:
                checkpoint_path = os.path.join(args.logdir, f'checkpoint_epoch_{epoch+1}.pth')
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'ema': ema.state_dict(),  # 保存EMA状态
                    'history': history,
                    'best_val_dice': best_val_dice,
                    'best_val_threshold': best_val_threshold
                }, checkpoint_path)
                print(f'  Checkpoint已保存: {checkpoint_path}')
            
            # 更新学习率
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            
            # 保存历史记录
            with open(os.path.join(args.logdir, 'history.json'), 'w') as f:
                json.dump(history, f, indent=2)
        
        # 保存最后一个checkpoint
        latest_path = os.path.join(args.logdir, 'latest.pth')
        torch.save({
            'epoch': args.epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'ema': ema.state_dict(),  # 保存EMA状态
            'history': history,
            'best_val_dice': best_val_dice,
            'best_val_threshold': best_val_threshold
        }, latest_path)
        print(f'\n最新模型已保存: {latest_path}')
        print(f'最佳验证Dice: {best_val_dice:.4f} @ 阈值: {best_val_threshold:.2f}')
        
        # 关闭TensorBoard writer
        writer.close()
        print(f'\nTensorBoard日志已保存，可使用以下命令查看:')
        print(f'  tensorboard --logdir {tensorboard_dir}')

    elif args.test:
        # 测试模式
        if args.test_ct and args.test_pet:
            test(args, model, args.test_ct, args.test_pet, args.save_dir)
        else:
            print("Error: Please provide --test_ct and --test_pet directories")
    else:
        print("Error: Please specify --train or --test mode")


if __name__ == '__main__':
    main()
