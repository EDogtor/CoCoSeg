"""
指数移动平均（Exponential Moving Average, EMA）工具类

EMA通过维护模型参数的移动平均来提高模型的稳定性和性能。
在训练过程中，EMA模型通常能提供更好的泛化能力。
"""

import torch
import torch.nn as nn
import copy
from typing import Optional


class EMA:
    """
    指数移动平均（EMA）类
    
    在训练过程中维护模型参数的指数移动平均，通常能提供更好的验证性能。
    
    Attributes:
        model: 要应用EMA的模型
        decay: EMA衰减率（0-1之间，通常0.999或0.9999）
        device: 设备（CPU或CUDA）
        shadow: EMA模型的参数副本
        backup: 原始模型参数的备份（用于恢复）
        num_updates: 更新次数（用于动态衰减率计算）
    """
    
    def __init__(self, model: nn.Module, decay: float = 0.999, device: Optional[torch.device] = None):
        """
        初始化EMA
        
        Args:
            model: 要应用EMA的模型
            decay: EMA衰减率（默认0.999）
                   - 0.999: 更快的更新，对近期权重更敏感
                   - 0.9999: 更慢的更新，更平滑但可能需要更多步数
            device: 设备，如果为None则自动检测
        """
        self.decay = decay
        self.device = device if device is not None else next(model.parameters()).device
        self.model = model.to(self.device)
        
        # 创建EMA模型的参数副本
        self.shadow = {}
        self.backup = {}
        
        # 注册所有参数的EMA版本
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().detach().to(self.device)
        
        self.num_updates = 0
    
    def update(self):
        """
        更新EMA参数
        
        在每个训练步骤后调用此方法来更新EMA模型参数。
        EMA更新公式: shadow = decay * shadow + (1 - decay) * param
        """
        self.num_updates += 1
        
        # 动态衰减率（可选，让早期训练时更新更快）
        decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        
        # 更新所有参数的EMA版本
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow, f"参数 {name} 不在EMA影子参数中"
                # 确保shadow参数在正确的设备上
                shadow_param = self.shadow[name].to(param.device)
                new_val = decay * shadow_param + (1.0 - decay) * param.data
                self.shadow[name] = new_val.clone().detach().to(self.device)
    
    def apply_shadow(self):
        """
        应用EMA参数到模型（将模型参数替换为EMA参数）
        
        在验证/测试前调用此方法，使用EMA模型进行评估。
        注意：这会修改原始模型的参数。
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                # 确保shadow参数在正确的设备上
                param.data = self.shadow[name].to(param.device)
    
    def restore(self):
        """
        恢复原始模型参数
        
        在验证/测试后调用此方法，恢复原始训练模型参数。
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
    
    def state_dict(self):
        """
        获取EMA状态字典（用于保存检查点）
        
        Returns:
            包含shadow参数和更新次数的字典
        """
        return {
            'shadow': self.shadow,
            'num_updates': self.num_updates,
            'decay': self.decay
        }
    
    def load_state_dict(self, state_dict):
        """
        加载EMA状态字典（用于恢复检查点）
        
        Args:
            state_dict: 包含shadow参数和更新次数的字典
        """
        self.shadow = state_dict['shadow']
        self.num_updates = state_dict.get('num_updates', 0)
        self.decay = state_dict.get('decay', self.decay)
        
        # 确保参数在正确的设备上
        for name in self.shadow:
            self.shadow[name] = self.shadow[name].to(self.device)
    
    def copy_to(self, model: nn.Module):
        """
        将EMA参数复制到另一个模型
        
        Args:
            model: 目标模型
        """
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                param.data = self.shadow[name].clone().detach().to(param.device)


def create_ema_model(model: nn.Module, decay: float = 0.999, device: Optional[torch.device] = None) -> EMA:
    """
    便捷函数：创建EMA对象
    
    Args:
        model: 要应用EMA的模型
        decay: EMA衰减率（默认0.999）
        device: 设备
    
    Returns:
        EMA对象
    """
    return EMA(model, decay=decay, device=device)

