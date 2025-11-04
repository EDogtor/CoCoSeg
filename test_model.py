#!/usr/bin/env python3
"""
模型测试和评估脚本
支持在有ground truth的情况下计算详细的评估指标
"""

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
from datetime import datetime

from models.model import Unet_resize_conv
from utils import fname_presuffix
from data.dataset import LungSegmentationDataset
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


class GrayscaleTransform:
    def __call__(self, img):
        # Convert the image to grayscale
        if img.shape[0] == 3:
            img = img[0, :, :]
            img = torch.unsqueeze(img, 0)
        return img


def calculate_metrics(pred, target):
    """计算评估指标
    
    Args:
        pred: 预测值 [B, 1, H, W]
        target: 真实值 [B, 1, H, W]
    
    Returns:
        dice, iou, accuracy
    """
    # 二值化预测
    pred_binary = (torch.sigmoid(pred) > 0.5).float()
    
    # Flatten
    pred_flat = pred_binary.view(-1)
    target_flat = target.view(-1)
    
    # 计算交集和并集
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum()
    
    # Dice系数
    dice = (2. * intersection + 1e-6) / (union + 1e-6)
    
    # IoU
    iou = (intersection + 1e-6) / (union - intersection + 1e-6)
    
    # Accuracy
    correct = (pred_flat == target_flat).sum()
    accuracy = correct / len(pred_flat)
    
    return dice, iou, accuracy


def test_with_metrics(model, val_loader, device, save_dir=None):
    """测试模型并计算指标（使用DataLoader）"""
    model.eval()
    total_dice = 0
    total_iou = 0
    total_acc = 0
    num_samples = 0
    
    # 存储每张图片的指标
    per_image_metrics = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Testing', ncols=100)
        for batch_idx, batch in enumerate(pbar):
            ct = batch['ct'].to(device)
            pet = batch['pet'].to(device)
            mask = batch['mask'].to(device)
            filenames = batch.get('name', [f'sample_{batch_idx}_{i}' for i in range(len(ct))])
            
            # 前向传播
            output = model(ct, pet)
            
            # 计算指标
            dice, iou, acc = calculate_metrics(output, mask)
            
            # 累计统计
            batch_size = ct.size(0)
            total_dice += dice.item() * batch_size
            total_iou += iou.item() * batch_size
            total_acc += acc.item() * batch_size
            num_samples += batch_size
            
            # 记录每张图片的指标
            for i in range(batch_size):
                per_image_metrics.append({
                    'filename': filenames[i] if isinstance(filenames[i], str) else filenames[i].item(),
                    'dice': dice.item() if batch_size == 1 else calculate_metrics(
                        output[i:i+1], mask[i:i+1]
                    )[0].item(),
                    'iou': iou.item() if batch_size == 1 else calculate_metrics(
                        output[i:i+1], mask[i:i+1]
                    )[1].item(),
                    'accuracy': acc.item() if batch_size == 1 else calculate_metrics(
                        output[i:i+1], mask[i:i+1]
                    )[2].item()
                })
            
            # 更新进度条
            pbar.set_postfix({
                'dice': f'{dice.item():.3f}',
                'iou': f'{iou.item():.3f}',
                'acc': f'{acc.item():.3f}'
            })
            
            # 保存预测结果
            if save_dir:
                for i in range(batch_size):
                    # 保存预测mask
                    output_np = (torch.sigmoid(output[i]) > 0.5).float().squeeze().cpu().numpy()
                    filename = filenames[i] if isinstance(filenames[i], str) else filenames[i].item()
                    # 确保文件名有.png扩展名
                    if not filename.endswith('.png'):
                        filename = filename + '.png'
                    save_fn = os.path.join(save_dir, filename)
                    cv2.imwrite(save_fn, output_np * 255)
    
    # 平均指标
    avg_dice = total_dice / num_samples if num_samples > 0 else 0
    avg_iou = total_iou / num_samples if num_samples > 0 else 0
    avg_acc = total_acc / num_samples if num_samples > 0 else 0
    
    return avg_dice, avg_iou, avg_acc, per_image_metrics


def test_simple_inference(model, ct_path, pet_path, save_path, device):
    """简单推理模式（无ground truth，只生成预测）"""
    print('开始推理模式...')
    ct_list = [n for n in os.listdir(ct_path) if n.endswith('.png')]
    pet_list = [n for n in os.listdir(pet_path) if n.endswith('.png')]
    
    ct_list = sorted(ct_list)
    pet_list = sorted(pet_list)
    
    if len(ct_list) != len(pet_list):
        print(f"Warning: CT文件数量 ({len(ct_list)}) 和PET文件数量 ({len(pet_list)}) 不匹配")
    
    model.to(device)
    model.eval()
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        GrayscaleTransform()
    ])
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    import time
    Time = []
    pbar = tqdm(zip(ct_list, pet_list), total=len(ct_list), desc='推理中')
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
        
        save_fn = fname_presuffix(
            fname=ct_file, prefix='', suffix='', newpath=save_path)
        cv2.imwrite(save_fn.split('.')[0] + '.png', output_np * 255)
        
        end = time.time()
        Time.append(end - start)
    
    print(f"推理完成！平均时间: {np.mean(Time):.3f}秒/张, 标准差: {np.std(Time):.3f}秒")


def main():
    parser = argparse.ArgumentParser(description='CoCoSeg: 模型测试和评估')
    
    # 模型参数
    parser.add_argument('--ckpt', type=str, required=True, 
                       help='模型checkpoint路径 (如: ./logs/best_model.pth)')
    parser.add_argument('--use_gpu', action='store_true', help='使用GPU')
    
    # 测试模式选择
    parser.add_argument('--mode', type=str, choices=['eval', 'infer'], 
                       default='eval',
                       help='评估模式: eval=有ground truth计算指标, infer=仅生成预测')
    
    # 评估模式参数（使用DataLoader）
    parser.add_argument('--dataset_root', type=str, default='./dataset',
                       help='数据集根目录（评估模式使用）')
    parser.add_argument('--split', type=str, choices=['train', 'val', 'test'], 
                       default='val',
                       help='使用哪个数据分割进行测试')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='批处理大小')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='数据加载进程数')
    
    # 推理模式参数（直接指定路径）
    parser.add_argument('--test_ct', type=str, 
                       help='CT图像目录（推理模式使用）')
    parser.add_argument('--test_pet', type=str,
                       help='PET图像目录（推理模式使用）')
    
    # 输出参数
    parser.add_argument('--save_dir', type=str, default='./test_results/',
                       help='结果保存目录')
    parser.add_argument('--save_predictions', action='store_true',
                       help='是否保存预测结果图像')
    
    args = parser.parse_args()
    
    print('\n' + '='*60)
    print('CoCoSeg: 模型测试和评估')
    print('='*60)
    print(f'GPU可用: {torch.cuda.is_available()}')
    print(f'测试模式: {args.mode}')
    print(f'模型checkpoint: {args.ckpt}')
    print('='*60)
    
    # 设置设备
    if args.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✓ 使用GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        if args.use_gpu:
            print("⚠ CUDA不可用，使用CPU模式")
        else:
            print("✓ 使用CPU模式")
    
    # 加载模型
    print(f"\n加载模型从: {args.ckpt}")
    if not os.path.exists(args.ckpt):
        print(f"错误: Checkpoint文件不存在: {args.ckpt}")
        return
    
    model = Unet_resize_conv()
    checkpoint = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    
    # 显示模型信息
    if 'epoch' in checkpoint:
        print(f"模型epoch: {checkpoint['epoch']}")
    if 'best_val_dice' in checkpoint:
        print(f"训练时最佳验证Dice: {checkpoint['best_val_dice']:.4f}")
    
    # 创建保存目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(args.save_dir, timestamp)
    os.makedirs(save_dir, exist_ok=True)
    print(f"结果保存到: {save_dir}")
    
    # 根据模式执行测试
    if args.mode == 'eval':
        print("\n开始评估模式（使用ground truth计算指标）...")
        print(f"数据集根目录: {args.dataset_root}")
        print(f"数据分割: {args.split}")
        
        # 创建测试数据集
        test_dataset = LungSegmentationDataset(
            dataset_root=args.dataset_root,
            split=args.split,
            val_ratio=0.2,
            random_seed=42
        )
        print(f"测试样本数: {len(test_dataset)}")
        
        # 创建DataLoader
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True if args.use_gpu else False
        )
        
        # 运行评估
        save_pred_dir = os.path.join(save_dir, 'predictions') if args.save_predictions else None
        if save_pred_dir:
            os.makedirs(save_pred_dir, exist_ok=True)
        
        avg_dice, avg_iou, avg_acc, per_image_metrics = test_with_metrics(
            model, test_loader, device, save_pred_dir
        )
        
        # 打印结果
        print('\n' + '='*60)
        print('评估结果汇总')
        print('='*60)
        print(f'平均Dice系数: {avg_dice:.4f}')
        print(f'平均IoU:       {avg_iou:.4f}')
        print(f'平均准确率:   {avg_acc:.4f}')
        print('='*60)
        
        # 保存详细结果
        results = {
            'checkpoint': args.ckpt,
            'dataset_root': args.dataset_root,
            'split': args.split,
            'num_samples': len(test_dataset),
            'overall_metrics': {
                'dice': avg_dice,
                'iou': avg_iou,
                'accuracy': avg_acc
            },
            'per_image_metrics': per_image_metrics,
            'timestamp': timestamp
        }
        
        results_file = os.path.join(save_dir, 'results.json')
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n详细结果已保存到: {results_file}")
        
    elif args.mode == 'infer':
        print("\n开始推理模式（仅生成预测，不计算指标）...")
        
        if not args.test_ct or not args.test_pet:
            print("错误: 推理模式需要指定 --test_ct 和 --test_pet 参数")
            return
        
        print(f"CT目录: {args.test_ct}")
        print(f"PET目录: {args.test_pet}")
        
        save_pred_dir = os.path.join(save_dir, 'predictions')
        os.makedirs(save_pred_dir, exist_ok=True)
        
        test_simple_inference(model, args.test_ct, args.test_pet, save_pred_dir, device)
        print(f"\n预测结果已保存到: {save_pred_dir}")
    
    print("\n测试完成！")


if __name__ == '__main__':
    main()

