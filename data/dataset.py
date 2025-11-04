import h5py
from torch.utils.data import Dataset
from os.path import splitext, join, exists
from os import listdir
import os
import numpy as np
from glob import glob
import torch
import logging
from PIL import Image
import random
from collections import defaultdict
import albumentations as A
import cv2


class TrainDataSet(Dataset):
    """旧版本的H5数据集加载器（保留用于兼容）"""
    def __init__(self, dataset=None, arg=None):
        super(TrainDataSet, self).__init__()
        self.arg = arg
        data = h5py.File(dataset, 'r')
        data = data['data'][:]
        np.random.shuffle(data)

        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = ((self.data[idx] - 0.5) / 0.5).astype(np.float32)
        return data


class LungSegmentationDataset(Dataset):
    """肺癌CT-PET-Mask分割数据集
    
    支持两种数据格式：
    1. 预先分割的数据集（推荐）:
       - dataset_root/train/CT/, train/PET/, train/masks/
       - dataset_root/val/CT/, val/PET/, val/masks/
       - dataset_root/test/CT/, test/PET/, test/masks/
    
    2. 未分割的数据集（兼容模式）:
       - dataset_root/CT/patient_id_slice_001.png
       - dataset_root/PET/patient_id_slice_001.png
       - dataset_root/masks/patient_id_slice_001.png
    
    按病人ID分割，避免数据泄漏
    """
    
    def __init__(self, dataset_root, split='train', val_ratio=0.2, random_seed=42, 
                 scale=1.0, augment=False):
        """
        Args:
            dataset_root: 数据集根目录
                - 如果包含 train/val/test 子目录，则使用预先分割模式
                - 否则使用运行时分割模式（向后兼容）
            split: 'train' 或 'val' 或 'test'
            val_ratio: 验证集比例（仅在运行时分割模式下使用，默认0.2，即4:1）
            random_seed: 随机种子（仅在运行时分割模式下使用）
            scale: 图像缩放比例（0-1）
            augment: 是否进行数据增强
        """
        super(LungSegmentationDataset, self).__init__()
        
        self.scale = scale
        self.augment = augment
        
        # 初始化增强管道
        if self.augment:
            self.transform = self._get_augmentation_pipeline()
            logging.info("数据增强已启用")
        else:
            self.transform = None
        
        # 检查是否为预先分割的数据集
        split_dir = join(dataset_root, split)
        if exists(split_dir) and os.path.isdir(split_dir):
            # 预先分割模式：直接使用对应split目录
            logging.info(f"使用预先分割的数据集模式: {split_dir}")
            self.ct_dir = join(split_dir, 'CT')
            self.pet_dir = join(split_dir, 'PET')
            self.mask_dir = join(split_dir, 'masks')
            self._load_pre_split_data()
        else:
            # 运行时分割模式：从根目录分割
            logging.info(f"使用运行时分割模式: {dataset_root}")
            self.ct_dir = join(dataset_root, 'CT')
            self.pet_dir = join(dataset_root, 'PET')
            self.mask_dir = join(dataset_root, 'masks')
            self._load_and_split_data(split, val_ratio, random_seed)
    
    def _load_pre_split_data(self):
        """加载预先分割的数据集"""
        # 获取所有图像文件
        if not exists(self.ct_dir):
            raise FileNotFoundError(f"CT目录不存在: {self.ct_dir}")
        
        ct_files = [f for f in listdir(self.ct_dir) 
                   if f.endswith('.png') and not f.startswith('.')]
        
        # 构建数据列表
        self.data_list = []
        for ct_file in ct_files:
            ct_path = join(self.ct_dir, ct_file)
            pet_path = join(self.pet_dir, ct_file)
            mask_path = join(self.mask_dir, ct_file)
            
            # 验证文件是否存在
            if not glob(pet_path):
                logging.warning(f"PET file not found: {pet_path}")
                continue
            if not glob(mask_path):
                logging.warning(f"Mask file not found: {mask_path}")
                continue
            
            self.data_list.append({
                'ct': ct_path,
                'pet': pet_path,
                'mask': mask_path,
                'name': splitext(ct_file)[0]
            })
        
        logging.info(f"Loaded {len(self.data_list)} samples from pre-split dataset")
    
    def _load_and_split_data(self, split, val_ratio, random_seed):
        """加载并运行时分割数据集（向后兼容）"""
        # 获取所有图像文件并提取病人ID
        if not exists(self.ct_dir):
            raise FileNotFoundError(f"CT目录不存在: {self.ct_dir}")
        
        ct_files = [f for f in listdir(self.ct_dir) 
                   if f.endswith('.png') and not f.startswith('.')]
        
        # 按病人ID分组
        patient_dict = defaultdict(list)
        for ct_file in ct_files:
            # 假设命名格式: patient_id_slice_001.png
            filename = splitext(ct_file)[0]
            if '_slice_' in filename:
                patient_id = filename.rsplit('_slice_', 1)[0]
            else:
                # 如果没有slice编号，整个文件名作为patient_id
                patient_id = filename.split('_')[0]
            patient_dict[patient_id].append(ct_file)
        
        # 按病人ID排序并分割数据集
        patient_ids = sorted(patient_dict.keys())
        
        # 固定随机种子以确保可重复性
        random.seed(random_seed)
        random.shuffle(patient_ids)
        
        # 按比例分割
        total_patients = len(patient_ids)
        val_split_idx = int(total_patients * (1 - val_ratio))
        
        if split == 'train':
            self.patient_ids = patient_ids[:val_split_idx]
        elif split == 'val':
            self.patient_ids = patient_ids[val_split_idx:]
        elif split == 'test':
            # 如果没有test目录，将test等同于val
            logging.warning("运行时分割模式不支持test，使用val代替")
            self.patient_ids = patient_ids[val_split_idx:]
        else:
            raise ValueError(f"split must be 'train', 'val' or 'test', got {split}")
        
        # 为选中的病人收集所有切片
        self.data_list = []
        for patient_id in self.patient_ids:
            for ct_file in patient_dict[patient_id]:
                ct_path = join(self.ct_dir, ct_file)
                pet_file = ct_file.replace('.png', '.png')  # CT和PET文件名相同
                pet_path = join(self.pet_dir, pet_file)
                mask_path = join(self.mask_dir, ct_file)
                
                # 验证文件是否存在
                if not glob(pet_path):
                    logging.warning(f"PET file not found: {pet_path}")
                    continue
                if not glob(mask_path):
                    logging.warning(f"Mask file not found: {mask_path}")
                    continue
                
                self.data_list.append({
                    'ct': ct_path,
                    'pet': pet_path,
                    'mask': mask_path,
                    'name': splitext(ct_file)[0]
                })
        
        logging.info(f"Created {split} dataset with {len(self.patient_ids)} patients, "
                    f"{len(self.data_list)} slices")
        logging.info(f"Patient IDs in {split} set: {self.patient_ids[:5]}..." 
                    if len(self.patient_ids) > 5 else f"Patient IDs: {self.patient_ids}")
    
    def __len__(self):
        return len(self.data_list)
    
    def _get_augmentation_pipeline(self):
        """获取数据增强管道
        
        根据数据增强文档，使用Albumentations实现CT-PET-Mask的同步增强。
        组合了几何变换、弹性形变和噪声变换。
        注意：只有几何变换和弹性形变会同步应用，噪声变换只对CT和PET应用。
        """
        # 使用additional_targets定义多个输入
        return A.Compose([
            # 1. 几何变换 - 应对位置、角度变化（所有图像同步）
            A.HorizontalFlip(p=0.5),  # 50%概率水平翻转
            A.VerticalFlip(p=0.5),    # 50%概率垂直翻转
            A.Affine(
                rotate=(-15, 15),     # ±15度旋转
                translate_percent=(0, 0.05),  # 5%平移
                scale=(0.9, 1.1),     # 90%-110%缩放
                p=0.5,
                interpolation=cv2.INTER_LINEAR,
                fit_output=False      # 不调整输出尺寸
            ),
            
            # 2. 弹性形变 - 对抗过拟合的利器（所有图像同步）
            A.ElasticTransform(
                p=0.3,
                alpha=120,
                sigma=120 * 0.05,
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_REFLECT_101
            ),
            
            # 3. 像素/噪声变换 - 应对CT/PET成像差异（只对CT和PET应用）
            A.RandomBrightnessContrast(
                brightness_limit=0.1,   # ±10%亮度
                contrast_limit=0.1,     # ±10%对比度
                p=0.3
            ),
            A.GaussNoise(
                std_range=(0.04, 0.2),  # 高斯噪声标准差范围（归一化到[0,1]）
                p=0.2
            ),
        ], additional_targets={'image1': 'image'})  # 定义PET作为image1，与image（CT）同步变换
    
    @staticmethod
    def preprocess(img, scale=1.0, is_mask=False):
        """预处理图像"""
        # 转换为numpy数组
        img_array = np.array(img)
        
        # 如果是单通道灰度图，增加通道维度
        if len(img_array.shape) == 2:
            img_array = np.expand_dims(img_array, axis=2)
        
        # 缩放（图片和 mask 都要按各自的插值方式缩放）
        if scale != 1.0:
            from PIL import Image
            new_w = int(img_array.shape[1] * scale)
            new_h = int(img_array.shape[0] * scale)
            if is_mask:
                # 最近邻避免产生灰阶
                img = Image.fromarray(img_array.squeeze())
                img = img.resize((new_w, new_h), Image.NEAREST)
                img_array = np.array(img)[..., None] if img_array.ndim == 3 else np.array(img)
            else:
                # 双线性插值用于图像
                img = Image.fromarray(img_array.squeeze())
                img = img.resize((new_w, new_h), Image.BILINEAR)
                img_array = np.array(img)[..., None] if img_array.ndim == 3 else np.array(img)
        
        # 转换为CHW格式
        img_trans = img_array.transpose((2, 0, 1))
        
        # 归一化（mask不需要归一化）
        if not is_mask and img_trans.max() > 1:
            img_trans = img_trans / 255.0
        
        return img_trans
    
    def __getitem__(self, idx):
        """获取单个样本"""
        data_info = self.data_list[idx]
        
        # 加载图像
        try:
            ct_img = Image.open(data_info['ct']).convert('L')
            pet_img = Image.open(data_info['pet']).convert('L')
            mask_img = Image.open(data_info['mask']).convert('L')
        except Exception as e:
            logging.error(f"Error loading images: {data_info['name']}, {e}")
            # 返回下一个样本
            return self.__getitem__((idx + 1) % len(self.data_list))
        
        # 验证图像尺寸一致性
        assert ct_img.size == pet_img.size == mask_img.size, \
            f"Image sizes mismatch for {data_info['name']}"
        
        # 预处理
        ct_array = self.preprocess(ct_img, self.scale, is_mask=False)
        pet_array = self.preprocess(pet_img, self.scale, is_mask=False)
        mask_array = self.preprocess(mask_img, self.scale, is_mask=True)
        
        # 归一化到[-1, 1]（CT和PET）
        ct_array = (ct_array - 0.5) / 0.5
        pet_array = (pet_array - 0.5) / 0.5
        
        # Mask二值化处理
        mask_array = (mask_array > 0.5).astype(np.float32)
        
        # 应用数据增强（如果需要）
        if self.transform is not None:
            # 将CHW转换为HWC以便Albumentations处理
            # 注意：preprocess返回的是CHW格式，shape为(1, H, W)
            ct_hwc = ct_array.transpose(1, 2, 0).squeeze()  # (H, W)
            pet_hwc = pet_array.transpose(1, 2, 0).squeeze()  # (H, W)
            mask_hwc = mask_array.transpose(1, 2, 0).squeeze()  # (H, W)
            
            # Albumentations需要numpy数组格式，并且需要是uint8类型
            # 注意：Albumentations默认输入范围是[0, 255]，我们需要先调整
            # 由于我们已经归一化到[-1, 1]，需要先转换回[0, 255]
            ct_hwc = np.clip(((ct_hwc + 1.0) / 2.0 * 255.0), 0, 255).astype(np.uint8)
            pet_hwc = np.clip(((pet_hwc + 1.0) / 2.0 * 255.0), 0, 255).astype(np.uint8)
            mask_hwc = np.clip(mask_hwc * 255.0, 0, 255).astype(np.uint8)
            
            # 应用增强：CT作为image，PET作为image1，mask作为mask
            augmented = self.transform(
                image=ct_hwc,
                image1=pet_hwc,  # PET图像，会同步变换
                mask=mask_hwc    # Mask图像，会同步变换
            )
            
            ct_hwc = augmented['image']
            pet_hwc = augmented['image1']
            mask_hwc = augmented['mask']
            
            # 转回[0, 1]浮点型范围
            ct_hwc = ct_hwc.astype(np.float32) / 255.0
            pet_hwc = pet_hwc.astype(np.float32) / 255.0
            mask_hwc = mask_hwc.astype(np.float32) / 255.0
            
            # 转回[-1, 1]范围
            ct_hwc = ct_hwc * 2.0 - 1.0
            pet_hwc = pet_hwc * 2.0 - 1.0
            # Mask保持[0, 1]范围
            
            # 转回CHW格式
            ct_array = np.expand_dims(ct_hwc, axis=0)  # (1, H, W)
            pet_array = np.expand_dims(pet_hwc, axis=0)  # (1, H, W)
            mask_array = np.expand_dims(mask_hwc, axis=0).astype(np.float32)  # (1, H, W)
        
        # 转换为tensor
        ct_tensor = torch.from_numpy(ct_array).type(torch.FloatTensor)
        pet_tensor = torch.from_numpy(pet_array).type(torch.FloatTensor)
        mask_tensor = torch.from_numpy(mask_array).type(torch.FloatTensor)
        
        return {
            'ct': ct_tensor,
            'pet': pet_tensor,
            'mask': mask_tensor,
            'name': data_info['name']
        }


# 保留旧代码用于兼容
class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale):

        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))

        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(self.masks_dir + idx + '.*')
        img_file = glob(self.imgs_dir + idx + '.*')

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale)
        mask = self.preprocess(mask, self.scale)
        img = (img - 0.5) / 0.5
        mask = (mask - 0.5) / 0.5

        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'mask': torch.from_numpy(mask).type(torch.FloatTensor),
            'name': idx
        }


class MaskDataset(Dataset):
    """旧版本的双模态数据集（保留用于兼容）"""
    def __init__(self, vis_dir, ir_dir, masks_dir, scale=1):
        self.vis_dir = vis_dir
        self.ir_dir = ir_dir
        self.masks_dir = masks_dir
        self.scale = scale
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(vis_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale):

        # w, h = pil_img.size
        # newW, newH = int(scale * w), int(scale * h)
        # assert newW > 0 and newH > 0, 'Scale is too small'
        # pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))

        if img_trans.max() > 1:
            img_trans = img_trans / 255

        # resize

        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        vis_file = glob(self.vis_dir + idx + '.*')
        ir_file = glob(self.ir_dir + idx + '.*')
        mask_file = glob(self.masks_dir + idx + '.*')

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(ir_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {ir_file}'
        assert len(vis_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {vis_file}'
        mask = Image.open(mask_file[0])
        img_vis = Image.open(vis_file[0]).convert('L')
        img_ir = Image.open(ir_file[0]).convert('L')

        assert img_vis.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img_vis.size} and {mask.size}'

        img_vis = self.preprocess(img_vis, self.scale)
        mask = self.preprocess(mask, self.scale)
        img_ir = self.preprocess(img_ir, self.scale)

        img_vis = (img_vis - 0.5) / 0.5
        # mask = (mask - 0.5)/0.5
        img_ir = (img_ir - 0.5) / 0.5

        return {
            'vis': torch.from_numpy(img_vis).type(torch.FloatTensor),
            'ir': torch.from_numpy(img_ir).type(torch.FloatTensor),
            'mask': torch.from_numpy(mask).type(torch.FloatTensor),
            'name': idx
        }


class PCLT20KDataset(Dataset):
    """PCLT20K数据集加载器
    
    数据集结构：
    - dataset_root/
      - 0001/
        - 0001_001_CT.png
        - 0001_001_PET.png
        - 0001_001_mask.png
        - 0001_002_CT.png
        - ...
      - 0002/
        - ...
    
    每个病人一个文件夹，文件中包含该病人的所有CT、PET和mask图片。
    文件命名格式：病人id_切片编号_模态.png
    
    支持两种数据格式：
    1. 预先分割的数据集（推荐）:
       - dataset_root/train/0001/, train/0002/, ...
       - dataset_root/val/0001/, val/0002/, ...
       - dataset_root/test/0001/, test/0002/, ...
    
    2. 未分割的数据集（兼容模式）:
       - dataset_root/0001/, 0002/, ...
    
    按病人ID分割，避免数据泄漏
    """
    
    def __init__(self, dataset_root, split='train', val_ratio=0.2, test_ratio=0.2, 
                 random_seed=42, scale=1.0, augment=False):
        """
        Args:
            dataset_root: 数据集根目录
                - 如果包含 train/val/test 子目录，则使用预先分割模式
                - 否则使用运行时分割模式（向后兼容）
            split: 'train' 或 'val' 或 'test'
            val_ratio: 验证集比例（仅在运行时分割模式下使用，默认0.2）
            test_ratio: 测试集比例（仅在运行时分割模式下使用，默认0.2）
            random_seed: 随机种子（仅在运行时分割模式下使用）
            scale: 图像缩放比例（0-1）
            augment: 是否进行数据增强
        """
        super(PCLT20KDataset, self).__init__()
        
        self.scale = scale
        self.augment = augment
        self.dataset_root = dataset_root
        
        # 初始化增强管道
        if self.augment:
            self.transform = self._get_augmentation_pipeline()
            logging.info("数据增强已启用")
        else:
            self.transform = None
        
        # 检查是否为预先分割的数据集
        split_dir = join(dataset_root, split)
        if exists(split_dir) and os.path.isdir(split_dir):
            # 预先分割模式：直接使用对应split目录
            logging.info(f"使用预先分割的PCLT20K数据集模式: {split_dir}")
            self._load_pre_split_data(split_dir)
        else:
            # 运行时分割模式：从根目录分割
            logging.info(f"使用运行时分割的PCLT20K数据集模式: {dataset_root}")
            self._load_and_split_data(split, val_ratio, test_ratio, random_seed)
    
    def _load_pre_split_data(self, split_dir):
        """加载预先分割的数据集"""
        # 获取所有病人文件夹
        patient_dirs = [d for d in os.listdir(split_dir) 
                       if os.path.isdir(join(split_dir, d)) and not d.startswith('.')]
        patient_dirs = sorted(patient_dirs)
        
        # 构建数据列表
        self.data_list = []
        for patient_id in patient_dirs:
            patient_dir = join(split_dir, patient_id)
            files = [f for f in os.listdir(patient_dir) 
                    if f.endswith('.png') and not f.startswith('.')]
            
            # 按切片编号分组
            slice_dict = {}
            for filename in files:
                # 解析文件名：病人id_切片编号_模态.png
                parts = splitext(filename)[0].split('_')
                if len(parts) >= 3:
                    slice_id = '_'.join(parts[:-1])  # 病人id_切片编号
                    modality = parts[-1].lower()
                    
                    if slice_id not in slice_dict:
                        slice_dict[slice_id] = {}
                    
                    file_path = join(patient_dir, filename)
                    if modality == 'ct':
                        slice_dict[slice_id]['ct'] = file_path
                    elif modality == 'pet':
                        slice_dict[slice_id]['pet'] = file_path
                    elif modality == 'mask':
                        slice_dict[slice_id]['mask'] = file_path
            
            # 验证并添加到数据列表
            for slice_id, files_dict in slice_dict.items():
                if 'ct' in files_dict and 'pet' in files_dict and 'mask' in files_dict:
                    self.data_list.append({
                        'ct': files_dict['ct'],
                        'pet': files_dict['pet'],
                        'mask': files_dict['mask'],
                        'name': slice_id
                    })
                else:
                    missing = []
                    if 'ct' not in files_dict:
                        missing.append('CT')
                    if 'pet' not in files_dict:
                        missing.append('PET')
                    if 'mask' not in files_dict:
                        missing.append('mask')
                    logging.warning(f"病人 {patient_id} 的切片 {slice_id} 缺少文件: {', '.join(missing)}")
        
        logging.info(f"Loaded {len(self.data_list)} samples from pre-split PCLT20K dataset")
    
    def _load_and_split_data(self, split, val_ratio, test_ratio, random_seed):
        """加载并运行时分割数据集（向后兼容）"""
        # 获取所有病人文件夹
        patient_dirs = [d for d in os.listdir(self.dataset_root) 
                       if os.path.isdir(join(self.dataset_root, d)) and not d.startswith('.')]
        patient_ids = sorted(patient_dirs)
        
        # 固定随机种子以确保可重复性
        random.seed(random_seed)
        random.shuffle(patient_ids)
        
        # 按比例分割
        total_patients = len(patient_ids)
        train_ratio = 1.0 - val_ratio - test_ratio
        
        train_end = int(total_patients * train_ratio)
        val_end = train_end + int(total_patients * val_ratio)
        
        if split == 'train':
            self.patient_ids = patient_ids[:train_end]
        elif split == 'val':
            self.patient_ids = patient_ids[train_end:val_end]
        elif split == 'test':
            self.patient_ids = patient_ids[val_end:]
        else:
            raise ValueError(f"split must be 'train', 'val' or 'test', got {split}")
        
        # 为选中的病人收集所有切片
        self.data_list = []
        for patient_id in self.patient_ids:
            patient_dir = join(self.dataset_root, patient_id)
            if not exists(patient_dir):
                logging.warning(f"病人目录不存在: {patient_dir}")
                continue
            
            files = [f for f in os.listdir(patient_dir) 
                    if f.endswith('.png') and not f.startswith('.')]
            
            # 按切片编号分组
            slice_dict = {}
            for filename in files:
                # 解析文件名：病人id_切片编号_模态.png
                parts = splitext(filename)[0].split('_')
                if len(parts) >= 3:
                    slice_id = '_'.join(parts[:-1])  # 病人id_切片编号
                    modality = parts[-1].lower()
                    
                    if slice_id not in slice_dict:
                        slice_dict[slice_id] = {}
                    
                    file_path = join(patient_dir, filename)
                    if modality == 'ct':
                        slice_dict[slice_id]['ct'] = file_path
                    elif modality == 'pet':
                        slice_dict[slice_id]['pet'] = file_path
                    elif modality == 'mask':
                        slice_dict[slice_id]['mask'] = file_path
            
            # 验证并添加到数据列表
            for slice_id, files_dict in slice_dict.items():
                if 'ct' in files_dict and 'pet' in files_dict and 'mask' in files_dict:
                    self.data_list.append({
                        'ct': files_dict['ct'],
                        'pet': files_dict['pet'],
                        'mask': files_dict['mask'],
                        'name': slice_id
                    })
                else:
                    missing = []
                    if 'ct' not in files_dict:
                        missing.append('CT')
                    if 'pet' not in files_dict:
                        missing.append('PET')
                    if 'mask' not in files_dict:
                        missing.append('mask')
                    logging.warning(f"病人 {patient_id} 的切片 {slice_id} 缺少文件: {', '.join(missing)}")
        
        logging.info(f"Created {split} dataset with {len(self.patient_ids)} patients, "
                    f"{len(self.data_list)} slices")
        logging.info(f"Patient IDs in {split} set: {self.patient_ids[:5]}..." 
                    if len(self.patient_ids) > 5 else f"Patient IDs: {self.patient_ids}")
    
    def __len__(self):
        return len(self.data_list)
    
    def _get_augmentation_pipeline(self):
        """获取数据增强管道（与LungSegmentationDataset相同）"""
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Affine(
                rotate=(-15, 15),
                translate_percent=(0, 0.05),
                scale=(0.9, 1.1),
                p=0.5,
                interpolation=cv2.INTER_LINEAR,
                fit_output=False
            ),
            A.ElasticTransform(
                p=0.3,
                alpha=120,
                sigma=120 * 0.05,
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_REFLECT_101
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.1,
                contrast_limit=0.1,
                p=0.3
            ),
            A.GaussNoise(
                std_range=(0.04, 0.2),
                p=0.2
            ),
        ], additional_targets={'image1': 'image'})
    
    @staticmethod
    def preprocess(img, scale=1.0, is_mask=False):
        """预处理图像（与LungSegmentationDataset相同）"""
        img_array = np.array(img)
        
        if len(img_array.shape) == 2:
            img_array = np.expand_dims(img_array, axis=2)
        
        if scale != 1.0:
            from PIL import Image
            new_w = int(img_array.shape[1] * scale)
            new_h = int(img_array.shape[0] * scale)
            if is_mask:
                img = Image.fromarray(img_array.squeeze())
                img = img.resize((new_w, new_h), Image.NEAREST)
                img_array = np.array(img)[..., None] if img_array.ndim == 3 else np.array(img)
            else:
                img = Image.fromarray(img_array.squeeze())
                img = img.resize((new_w, new_h), Image.BILINEAR)
                img_array = np.array(img)[..., None] if img_array.ndim == 3 else np.array(img)
        
        img_trans = img_array.transpose((2, 0, 1))
        
        if not is_mask and img_trans.max() > 1:
            img_trans = img_trans / 255.0
        
        return img_trans
    
    def __getitem__(self, idx):
        """获取单个样本（与LungSegmentationDataset相同）"""
        data_info = self.data_list[idx]
        
        # 加载图像
        try:
            ct_img = Image.open(data_info['ct']).convert('L')
            pet_img = Image.open(data_info['pet']).convert('L')
            mask_img = Image.open(data_info['mask']).convert('L')
        except Exception as e:
            logging.error(f"Error loading images: {data_info['name']}, {e}")
            return self.__getitem__((idx + 1) % len(self.data_list))
        
        # 验证图像尺寸一致性
        assert ct_img.size == pet_img.size == mask_img.size, \
            f"Image sizes mismatch for {data_info['name']}"
        
        # 预处理
        ct_array = self.preprocess(ct_img, self.scale, is_mask=False)
        pet_array = self.preprocess(pet_img, self.scale, is_mask=False)
        mask_array = self.preprocess(mask_img, self.scale, is_mask=True)
        
        # 归一化到[-1, 1]（CT和PET）
        ct_array = (ct_array - 0.5) / 0.5
        pet_array = (pet_array - 0.5) / 0.5
        
        # Mask二值化处理
        mask_array = (mask_array > 0.5).astype(np.float32)
        
        # 应用数据增强（如果需要）
        if self.transform is not None:
            ct_hwc = ct_array.transpose(1, 2, 0).squeeze()
            pet_hwc = pet_array.transpose(1, 2, 0).squeeze()
            mask_hwc = mask_array.transpose(1, 2, 0).squeeze()
            
            ct_hwc = np.clip(((ct_hwc + 1.0) / 2.0 * 255.0), 0, 255).astype(np.uint8)
            pet_hwc = np.clip(((pet_hwc + 1.0) / 2.0 * 255.0), 0, 255).astype(np.uint8)
            mask_hwc = np.clip(mask_hwc * 255.0, 0, 255).astype(np.uint8)
            
            augmented = self.transform(
                image=ct_hwc,
                image1=pet_hwc,
                mask=mask_hwc
            )
            
            ct_hwc = augmented['image']
            pet_hwc = augmented['image1']
            mask_hwc = augmented['mask']
            
            ct_hwc = ct_hwc.astype(np.float32) / 255.0
            pet_hwc = pet_hwc.astype(np.float32) / 255.0
            mask_hwc = mask_hwc.astype(np.float32) / 255.0
            
            ct_hwc = ct_hwc * 2.0 - 1.0
            pet_hwc = pet_hwc * 2.0 - 1.0
            
            ct_array = np.expand_dims(ct_hwc, axis=0)
            pet_array = np.expand_dims(pet_hwc, axis=0)
            mask_array = np.expand_dims(mask_hwc, axis=0).astype(np.float32)
        
        # 转换为tensor
        ct_tensor = torch.from_numpy(ct_array).type(torch.FloatTensor)
        pet_tensor = torch.from_numpy(pet_array).type(torch.FloatTensor)
        mask_tensor = torch.from_numpy(mask_array).type(torch.FloatTensor)
        
        return {
            'ct': ct_tensor,
            'pet': pet_tensor,
            'mask': mask_tensor,
            'name': data_info['name']
        }
