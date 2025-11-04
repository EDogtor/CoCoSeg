#!/usr/bin/env python
"""
数据集分割脚本

功能：
1. 将原始数据集按病人ID分割成训练集和测试集
2. 避免数据泄漏（同一病人的所有切片都在同一个集合中）
3. 创建新的目录结构并复制文件（或创建符号链接）
4. 保存分割信息到JSON文件

用法：
    python split_dataset.py --dataset_root ./dataset --output_root ./dataset_split --train_ratio 0.8 --copy_files

参数：
    --dataset_root: 原始数据集根目录（包含CT、PET、masks子目录）
    --output_root: 输出目录根目录
    --train_ratio: 训练集比例（默认0.8，即80%训练，20%测试）
    --random_seed: 随机种子（默认42）
    --copy_files: 是否复制文件（默认创建符号链接）
    --val_ratio: 验证集比例（默认0.2，即从训练集中再分出20%作为验证集）
"""
import os
import shutil
import argparse
import json
import random
from pathlib import Path
from collections import defaultdict
from os.path import join, splitext, exists
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def extract_patient_id(filename):
    """从文件名提取病人ID
    
    Args:
        filename: 文件名，例如 patient_001_slice_001.png
        
    Returns:
        patient_id: 病人ID，例如 patient_001
    """
    name = splitext(filename)[0]
    if '_slice_' in name:
        patient_id = name.rsplit('_slice_', 1)[0]
    else:
        # 如果没有slice编号，使用文件名前缀
        parts = name.split('_')
        if len(parts) >= 2:
            patient_id = '_'.join(parts[:2])  # 假设格式为 patient_001
        else:
            patient_id = parts[0]
    return patient_id


def get_all_patients(dataset_root):
    """获取所有病人ID和对应的文件
    
    Args:
        dataset_root: 数据集根目录
        
    Returns:
        patient_dict: {patient_id: [file_list]} 字典
    """
    ct_dir = join(dataset_root, 'CT')
    pet_dir = join(dataset_root, 'PET')
    mask_dir = join(dataset_root, 'masks')
    
    # 检查目录是否存在
    for dir_path, dir_name in zip([ct_dir, pet_dir, mask_dir], ['CT', 'PET', 'masks']):
        if not exists(dir_path):
            raise FileNotFoundError(f"{dir_name}目录不存在: {dir_path}")
    
    # 获取所有CT文件
    ct_files = [f for f in os.listdir(ct_dir) 
                if f.endswith('.png') and not f.startswith('.')]
    
    # 按病人ID分组
    patient_dict = defaultdict(list)
    for ct_file in ct_files:
        patient_id = extract_patient_id(ct_file)
        patient_dict[patient_id].append(ct_file)
    
    # 验证文件是否存在
    valid_patients = {}
    for patient_id, ct_files_list in patient_dict.items():
        valid_files = []
        for ct_file in ct_files_list:
            ct_path = join(ct_dir, ct_file)
            pet_path = join(pet_dir, ct_file)
            mask_path = join(mask_dir, ct_file)
            
            # 检查所有文件是否存在
            if exists(ct_path) and exists(pet_path) and exists(mask_path):
                valid_files.append(ct_file)
            else:
                missing = []
                if not exists(ct_path):
                    missing.append('CT')
                if not exists(pet_path):
                    missing.append('PET')
                if not exists(mask_path):
                    missing.append('mask')
                logging.warning(
                    f"病人 {patient_id} 的文件 {ct_file} 缺少: {', '.join(missing)}"
                )
        
        if valid_files:
            valid_patients[patient_id] = valid_files
        else:
            logging.warning(f"病人 {patient_id} 没有有效文件，跳过")
    
    return valid_patients


def split_patients(patient_dict, train_ratio, random_seed, val_ratio=None):
    """分割病人ID
    
    Args:
        patient_dict: 病人字典
        train_ratio: 训练集比例
        random_seed: 随机种子
        val_ratio: 验证集比例（从训练集中分出）
        
    Returns:
        train_patients: 训练集病人ID列表
        val_patients: 验证集病人ID列表（如果有）
        test_patients: 测试集病人ID列表
    """
    patient_ids = sorted(patient_dict.keys())
    
    # 固定随机种子
    random.seed(random_seed)
    random.shuffle(patient_ids)
    
    total_patients = len(patient_ids)
    
    # 分割训练集和测试集
    train_split_idx = int(total_patients * train_ratio)
    train_patients = patient_ids[:train_split_idx]
    test_patients = patient_ids[train_split_idx:]
    
    val_patients = None
    if val_ratio and val_ratio > 0:
        # 从训练集中分出验证集
        val_split_idx = int(len(train_patients) * (1 - val_ratio))
        val_patients = train_patients[val_split_idx:]
        train_patients = train_patients[:val_split_idx]
    
    return train_patients, val_patients, test_patients


def create_split_structure(output_root, split_name, patient_dict, patient_ids, 
                          dataset_root, copy_files=False):
    """创建分割后的目录结构并复制/链接文件
    
    Args:
        output_root: 输出根目录
        split_name: 分割名称（train/val/test）
        patient_dict: 病人字典
        patient_ids: 该分割的病人ID列表
        dataset_root: 原始数据集根目录
        copy_files: 是否复制文件（False则创建符号链接）
    """
    split_dir = join(output_root, split_name)
    
    # 创建目录
    for subdir in ['CT', 'PET', 'masks']:
        os.makedirs(join(split_dir, subdir), exist_ok=True)
    
    # 复制或链接文件
    file_count = 0
    for patient_id in patient_ids:
        for filename in patient_dict[patient_id]:
            src_ct = join(dataset_root, 'CT', filename)
            src_pet = join(dataset_root, 'PET', filename)
            src_mask = join(dataset_root, 'masks', filename)
            
            dst_ct = join(split_dir, 'CT', filename)
            dst_pet = join(split_dir, 'PET', filename)
            dst_mask = join(split_dir, 'masks', filename)
            
            if copy_files:
                shutil.copy2(src_ct, dst_ct)
                shutil.copy2(src_pet, dst_pet)
                shutil.copy2(src_mask, dst_mask)
            else:
                # 创建符号链接
                if exists(dst_ct):
                    os.remove(dst_ct)
                if exists(dst_pet):
                    os.remove(dst_pet)
                if exists(dst_mask):
                    os.remove(dst_mask)
                os.symlink(os.path.abspath(src_ct), dst_ct)
                os.symlink(os.path.abspath(src_pet), dst_pet)
                os.symlink(os.path.abspath(src_mask), dst_mask)
            
            file_count += 1
    
    logging.info(f"创建 {split_name} 集: {len(patient_ids)}个病人, {file_count}个文件")
    return file_count


def save_split_info(output_root, train_patients, val_patients, test_patients,
                   patient_dict, train_ratio, val_ratio, random_seed):
    """保存分割信息到JSON文件
    
    Args:
        output_root: 输出根目录
        train_patients: 训练集病人ID列表
        val_patients: 验证集病人ID列表（可能为None）
        test_patients: 测试集病人ID列表
        patient_dict: 病人字典
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        random_seed: 随机种子
    """
    # 统计信息
    info = {
        'split_config': {
            'train_ratio': train_ratio,
            'val_ratio': val_ratio,
            'random_seed': random_seed,
        },
        'statistics': {
            'total_patients': len(train_patients) + len(test_patients) + 
                           (len(val_patients) if val_patients else 0),
            'total_files': sum(len(files) for files in patient_dict.values()),
        },
        'train_set': {
            'patient_count': len(train_patients),
            'file_count': sum(len(patient_dict[p]) for p in train_patients),
            'patient_ids': train_patients,
        },
        'test_set': {
            'patient_count': len(test_patients),
            'file_count': sum(len(patient_dict[p]) for p in test_patients),
            'patient_ids': test_patients,
        }
    }
    
    if val_patients:
        info['val_set'] = {
            'patient_count': len(val_patients),
            'file_count': sum(len(patient_dict[p]) for p in val_patients),
            'patient_ids': val_patients,
        }
        info['statistics']['total_patients'] = (len(train_patients) + 
                                               len(val_patients) + 
                                               len(test_patients))
    
    # 保存到JSON
    json_path = join(output_root, 'split_info.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    
    logging.info(f"分割信息已保存到: {json_path}")
    return info


def print_summary(info):
    """打印分割摘要信息"""
    print("\n" + "=" * 60)
    print("数据集分割摘要")
    print("=" * 60)
    print(f"总病人数: {info['statistics']['total_patients']}")
    print(f"总文件数: {info['statistics']['total_files']}")
    print(f"\n训练集: {info['train_set']['patient_count']}个病人, "
          f"{info['train_set']['file_count']}个文件")
    if 'val_set' in info:
        print(f"验证集: {info['val_set']['patient_count']}个病人, "
              f"{info['val_set']['file_count']}个文件")
    print(f"测试集: {info['test_set']['patient_count']}个病人, "
          f"{info['test_set']['file_count']}个文件")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='分割CT-PET-Mask数据集为训练集和测试集',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 创建符号链接（节省空间）
  python split_dataset.py --dataset_root ./dataset --output_root ./dataset_split
  
  # 复制文件
  python split_dataset.py --dataset_root ./dataset --output_root ./dataset_split --copy_files
  
  # 自定义分割比例（80%训练，20%测试，训练集中20%作为验证集）
  python split_dataset.py --dataset_root ./dataset --output_root ./dataset_split \\
    --train_ratio 0.8 --val_ratio 0.2 --random_seed 42
        """
    )
    
    parser.add_argument('--dataset_root', type=str, required=True,
                       help='原始数据集根目录（包含CT、PET、masks子目录）')
    parser.add_argument('--output_root', type=str, required=True,
                       help='输出目录根目录')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='训练集比例（默认0.8，即80%%训练，20%%测试）')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                       help='验证集比例，从训练集中分出（默认0.2，即训练集的20%%）')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='随机种子（默认42）')
    parser.add_argument('--copy_files', action='store_true',
                       help='是否复制文件（默认创建符号链接以节省空间）')
    
    args = parser.parse_args()
    
    # 验证参数
    if not 0 < args.train_ratio < 1:
        raise ValueError(f"train_ratio必须在0和1之间，当前值: {args.train_ratio}")
    if args.val_ratio and not 0 < args.val_ratio < 1:
        raise ValueError(f"val_ratio必须在0和1之间，当前值: {args.val_ratio}")
    
    print("\n" + "=" * 60)
    print("CoCoSeg 数据集分割工具")
    print("=" * 60)
    print(f"原始数据集: {args.dataset_root}")
    print(f"输出目录: {args.output_root}")
    print(f"训练集比例: {args.train_ratio}")
    if args.val_ratio > 0:
        print(f"验证集比例: {args.val_ratio} (从训练集中分出)")
    print(f"随机种子: {args.random_seed}")
    print(f"文件操作: {'复制' if args.copy_files else '符号链接'}")
    print("=" * 60)
    
    # 检查原始数据集
    if not exists(args.dataset_root):
        raise FileNotFoundError(f"数据集目录不存在: {args.dataset_root}")
    
    # 获取所有病人
    print("\n正在扫描数据集...")
    patient_dict = get_all_patients(args.dataset_root)
    print(f"✓ 找到 {len(patient_dict)} 个病人")
    
    # 分割病人
    print("\n正在分割数据集...")
    train_patients, val_patients, test_patients = split_patients(
        patient_dict, args.train_ratio, args.random_seed, args.val_ratio
    )
    
    print(f"✓ 训练集: {len(train_patients)} 个病人")
    if val_patients:
        print(f"✓ 验证集: {len(val_patients)} 个病人")
    print(f"✓ 测试集: {len(test_patients)} 个病人")
    
    # 创建输出目录
    os.makedirs(args.output_root, exist_ok=True)
    
    # 创建分割结构
    print("\n正在创建目录结构...")
    train_files = create_split_structure(
        args.output_root, 'train', patient_dict, train_patients,
        args.dataset_root, args.copy_files
    )
    
    test_files = create_split_structure(
        args.output_root, 'test', patient_dict, test_patients,
        args.dataset_root, args.copy_files
    )
    
    val_files = 0
    if val_patients:
        val_files = create_split_structure(
            args.output_root, 'val', patient_dict, val_patients,
            args.dataset_root, args.copy_files
        )
    
    # 保存分割信息
    print("\n正在保存分割信息...")
    split_info = save_split_info(
        args.output_root, train_patients, val_patients, test_patients,
        patient_dict, args.train_ratio, args.val_ratio, args.random_seed
    )
    
    # 打印摘要
    print_summary(split_info)
    
    print("\n✓ 数据集分割完成！")
    print(f"输出目录: {args.output_root}")
    print(f"分割信息: {join(args.output_root, 'split_info.json')}")


if __name__ == '__main__':
    main()


