"""
医疗图像分割专用损失函数
包含Dice Loss, CrossEntropy Loss等
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Dice损失函数
    适用于二值分割任务，对小目标友好
    """
    
    def __init__(self, smooth=1e-6):
        """
        Args:
            smooth: 平滑因子，避免分母为0
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        """
        Args:
            pred: 预测值 [B, 1, H, W] (logits或经过sigmoid)
            target: 真实值 [B, 1, H, W] (0或1)
        
        Returns:
            dice loss
        """
        # 将logits转换为概率
        pred = torch.sigmoid(pred)
        
        # Flatten
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        # 计算交集和并集
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        
        # Dice系数
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
        return 1 - dice


class CombinedSegLoss(nn.Module):
    """组合损失函数：Dice Loss + CrossEntropy Loss
    
    这是医疗图像分割中最常用的组合
    - Dice Loss: 处理类别不平衡，对小目标友好
    - CE Loss: 稳定梯度，帮助收敛
    """
    
    def __init__(self, 
                 dice_weight=0.5, 
                 ce_weight=0.5,
                 use_class_weights=False,
                 pos_weight=None):
        """
        Args:
            dice_weight: Dice损失的权重
            ce_weight: CrossEntropy损失的权重
            use_class_weights: 是否使用类别权重
            pos_weight: 正样本权重（用于处理类别不平衡）
        """
        super(CombinedSegLoss, self).__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.use_class_weights = use_class_weights
        
        # Dice Loss
        self.dice_loss = DiceLoss()
        
        # BCE with Logits Loss (等同于二分类CrossEntropy)
        # 将 pos_weight 注册为 buffer，确保在设备移动时自动处理
        if pos_weight is not None:
            pos_weight_tensor = torch.tensor([pos_weight], dtype=torch.float32)
            self.register_buffer('pos_weight', pos_weight_tensor)
            self.ce_loss = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        else:
            self.register_buffer('pos_weight', None)
            self.ce_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, pred, target):
        """
        Args:
            pred: 预测值 [B, 1, H, W]
            target: 真实值 [B, 1, H, W]
        
        Returns:
            total_loss, dice_loss, ce_loss
        """
        # 计算两个损失
        dice = self.dice_loss(pred, target)
        ce = self.ce_loss(pred, target)
        
        # 组合损失
        total_loss = self.dice_weight * dice + self.ce_weight * ce
        
        return total_loss, dice, ce


class IoULoss(nn.Module):
    """IoU损失函数
    与Dice类似，对小目标友好
    """
    
    def __init__(self, smooth=1e-6):
        super(IoULoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        # 将logits转换为概率
        pred = torch.sigmoid(pred)
        
        # Flatten
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        # 计算交集和并集
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        iou = (intersection + self.smooth) / (union - intersection + self.smooth)
        
        return 1 - iou


class TverskyLoss(nn.Module):
    """Tversky损失
    Dice Loss的泛化版本，可以调整假阳性/假阴性的权重
    """
    
    def __init__(self, alpha=0.7, beta=0.3, smooth=1e-6):
        """
        Args:
            alpha: 假阴性的权重（FN）
            beta: 假阳性的权重（FP）
            smooth: 平滑因子
        """
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
    
    def forward(self, pred, target):
        # 将logits转换为概率
        pred = torch.sigmoid(pred)
        
        # Flatten
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        # 计算TP, FP, FN
        intersection = (pred_flat * target_flat).sum()
        fp = (pred_flat * (1 - target_flat)).sum()
        fn = ((1 - pred_flat) * target_flat).sum()
        
        # Tversky系数
        tversky = (intersection + self.smooth) / (intersection + self.alpha * fn + self.beta * fp + self.smooth)
        
        return 1 - tversky


class FocalLoss(nn.Module):
    """Focal Loss
    解决难易样本不平衡问题
    """
    
    def __init__(self, alpha=1.0, gamma=2.0):
        """
        Args:
            alpha: 平衡因子
            gamma: 聚焦参数，越大越关注难样本
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        # BCE Loss
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        
        # 计算概率
        p = torch.sigmoid(pred)
        pt = p * target + (1 - p) * (1 - target)
        
        # Focal term
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        
        # Focal Loss
        focal_loss = focal_weight * bce
        
        return focal_loss.mean()


def get_loss_function(loss_type='combined', **kwargs):
    """获取损失函数的便捷函数
    
    Args:
        loss_type: 损失类型
            - 'dice': Dice Loss
            - 'ce': CrossEntropy Loss
            - 'combined': Dice + CE
            - 'iou': IoU Loss
            - 'tversky': Tversky Loss
            - 'focal': Focal Loss
        
    Returns:
        loss function
    """
    if loss_type == 'dice':
        return DiceLoss(**kwargs)
    elif loss_type == 'ce':
        return nn.BCEWithLogitsLoss(**kwargs)
    elif loss_type == 'combined':
        return CombinedSegLoss(**kwargs)
    elif loss_type == 'iou':
        return IoULoss(**kwargs)
    elif loss_type == 'tversky':
        return TverskyLoss(**kwargs)
    elif loss_type == 'focal':
        return FocalLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


