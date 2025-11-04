import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from utils import pad_tensor
from utils import pad_tensor_back
from utils.attention import CAM_Module


class Vgg19_Encoder(torch.nn.Module):
    """VGG19编码器 - 提取多尺度特征用于双模态融合
    
    从预训练VGG19提取三个层次特征：
    - h_relu1: 64 channels (浅层边缘特征)
    - h_relu2: 128 channels (中层纹理特征)
    - h_relu3: 256 channels (深层语义特征)
    """
    
    def __init__(self, requires_grad: bool = False):
        super(Vgg19_Encoder, self).__init__()
        
        # 加载预训练VGG19
        vgg_model = models.vgg19(weights='DEFAULT')
        pretrain_dict = vgg_model.state_dict()
        
        # 将第一层RGB卷积转换为灰度
        layer1 = pretrain_dict['features.0.weight']  # [64, 3, 3, 3]
        new = torch.zeros(64, 1, 3, 3)
        for i, output_channel in enumerate(layer1):
            # Grey = 0.299R + 0.587G + 0.114B, RGB2GREY
            new[i] = 0.299 * output_channel[0] + 0.587 * output_channel[1] + 0.114 * output_channel[2]
        pretrain_dict['features.0.weight'] = new
        vgg_model.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        vgg_model.load_state_dict(pretrain_dict)
        
        vgg_pretrained_features = vgg_model.features

        # 提取三个层次的特征（与原CoCoNet保持一致）
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()

        # Slice 1: layers 0-4 -> 64 channels
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        self.slice1.add_module(str(2), nn.MaxPool2d(2, 2))
        for x in range(2, 4):
            self.slice1.add_module(str(x+1), vgg_pretrained_features[x])
            
        # Slice 2: layers 5-8 -> 128 channels
        for x in range(4, 9):
            self.slice2.add_module(str(x+1), vgg_pretrained_features[x])
            
        # Slice 3: layers 9-17 -> 256 channels
        for x in range(9, 18):
            self.slice3.add_module(str(x+1), vgg_pretrained_features[x])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        """前向传播
        
        Args:
            x: 输入图像 [B, 1, H, W]
            
        Returns:
            h_relu1: 浅层特征 [B, 64, H/2, W/2]
            h_relu2: 中层特征 [B, 128, H/4, W/4]  
            h_relu3: 深层特征 [B, 256, H/8, W/8]
        """
        h_relu1 = self.slice1(x)      # 64 channels
        h_relu2 = self.slice2(h_relu1) # 128 channels
        h_relu3 = self.slice3(h_relu2) # 256 channels
        return h_relu1, h_relu2, h_relu3


class UNetEncoder(nn.Module):
    """独立的UNet编码器
    
    用于CT或PET模态的独立编码，保持模态特异性
    """
    
    def __init__(self):
        super(UNetEncoder, self).__init__()
        p = 1
        
        # 第1层：1 -> 32 channels
        self.conv1_1 = nn.Conv2d(1, 32, 3, padding=p)
        self.LReLU1_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn1_1 = nn.InstanceNorm2d(32)
        self.conv1_2 = nn.Conv2d(32, 32, 3, padding=p)
        self.LReLU1_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn1_2 = nn.InstanceNorm2d(32)
        self.max_pool1 = nn.MaxPool2d(2)
        
        # 第2层：32 -> 64 channels
        self.conv2_1 = nn.Conv2d(32, 64, 3, padding=p)
        self.LReLU2_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn2_1 = nn.InstanceNorm2d(64)
        self.conv2_2 = nn.Conv2d(64, 64, 3, padding=p)
        self.LReLU2_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn2_2 = nn.InstanceNorm2d(64)
        self.max_pool2 = nn.MaxPool2d(2)
        
        # 第3层：64 -> 128 channels
        self.conv3_1 = nn.Conv2d(64, 128, 3, padding=p)
        self.LReLU3_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn3_1 = nn.InstanceNorm2d(128)
        self.conv3_2 = nn.Conv2d(128, 128, 3, padding=p)
        self.LReLU3_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn3_2 = nn.InstanceNorm2d(128)
        self.max_pool3 = nn.MaxPool2d(2)
        
        # 第4层：128 -> 256 channels
        self.conv4_1 = nn.Conv2d(128, 256, 3, padding=p)
        self.LReLU4_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn4_1 = nn.InstanceNorm2d(256)
        self.conv4_2 = nn.Conv2d(256, 256, 3, padding=p)
        self.LReLU4_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn4_2 = nn.InstanceNorm2d(256)
    
    def forward(self, x):
        """前向传播
        
        Args:
            x: 输入 [B, 1, H, W]
            
        Returns:
            conv1: [B, 32, H, W]
            conv2: [B, 64, H/2, W/2]
            conv3: [B, 128, H/4, W/4]
            conv4: [B, 256, H/8, W/8]
        """
        # 第1层
        x = self.bn1_1(self.LReLU1_1(self.conv1_1(x)))
        conv1 = self.bn1_2(self.LReLU1_2(self.conv1_2(x)))
        x = self.max_pool1(conv1)
        
        # 第2层
        x = self.bn2_1(self.LReLU2_1(self.conv2_1(x)))
        conv2 = self.bn2_2(self.LReLU2_2(self.conv2_2(x)))
        x = self.max_pool2(conv2)
        
        # 第3层
        x = self.bn3_1(self.LReLU3_1(self.conv3_1(x)))
        conv3 = self.bn3_2(self.LReLU3_2(self.conv3_2(x)))
        x = self.max_pool3(conv3)
        
        # 第4层
        x = self.bn4_1(self.LReLU4_1(self.conv4_1(x)))
        conv4 = self.bn4_2(self.LReLU4_2(self.conv4_2(x)))
        
        return conv1, conv2, conv3, conv4


class DualIndependentEncoderUNet(nn.Module):
    """双独立编码器UNet用于CT-PET分割
    
    架构设计：
    1. CT和PET分别输入两个独立的UNet编码器（非共享权重）
    2. VGG编码器已关闭（可选，当前不使用）
    3. 在跳跃连接中使用MAM模块（CAM注意力）增强特征
       - 对CT和PET的UNet特征分别应用通道注意力
       - 自适应选择重要通道，提升融合效果
    4. 采用中期融合（Mid-Level Fusion）
       - 在4个层级（32, 64, 128, 256通道）进行CT-PET特征融合
    5. 通过解码器上采样并输出分割结果
    """
    
    def __init__(self):
        super(DualIndependentEncoderUNet, self).__init__()
        
        # ========== VGG19编码器（已关闭）==========
        # self.ct_vgg_encoder = Vgg19_Encoder(requires_grad=False)
        # self.pet_vgg_encoder = Vgg19_Encoder(requires_grad=False)
        
        # 独立的UNet编码器（主要特征提取）
        self.ct_unet_encoder = UNetEncoder()
        self.pet_unet_encoder = UNetEncoder()
        
        # ========== CAM注意力模块（MAM）- 用于跳跃连接中的注意力增强 ==========
        self.cam_32 = CAM_Module(32)
        self.cam_64 = CAM_Module(64)
        self.cam_128 = CAM_Module(128)
        self.cam_256 = CAM_Module(256)
        
        p = 1
        
        # ========== 中间融合层和解码器（修改为仅使用UNet特征）==========
        # 最深层融合：CT_UNet(256) + PET_UNet(256) -> 256（从2路改为4路->2路）
        self.deconv4 = nn.Conv2d(256 * 2, 256, 3, padding=p)
        self.bn4_3 = nn.InstanceNorm2d(256)
        
        # 解码器第1层
        self.deconv6 = nn.Conv2d(256, 128, 3, padding=p)
        # 中层融合：2路特征 -> 128（从4路改为2路）
        self.att_deconv7 = nn.Conv2d(128 * 2, 128, 3, padding=p)
        self.conv7_1 = nn.Conv2d(256, 128, 3, padding=p)  # 解码器上采样128 + 融合128 -> 256
        self.LReLU7_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn7_1 = nn.InstanceNorm2d(128)
        self.conv7_2 = nn.Conv2d(128, 128, 3, padding=p)
        self.LReLU7_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn7_2 = nn.InstanceNorm2d(128)
        
        # 解码器第2层
        self.deconv7 = nn.Conv2d(128, 64, 3, padding=p)
        # 浅层融合：2路特征 -> 64（从4路改为2路）
        self.att_deconv8 = nn.Conv2d(64 * 2, 64, 3, padding=p)
        self.conv8_1 = nn.Conv2d(128, 64, 3, padding=p)
        self.LReLU8_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn8_1 = nn.InstanceNorm2d(64)
        self.conv8_2 = nn.Conv2d(64, 64, 3, padding=p)
        self.LReLU8_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn8_2 = nn.InstanceNorm2d(64)
        
        # 解码器第3层
        self.deconv8 = nn.Conv2d(64, 32, 3, padding=p)
        self.conv9_1 = nn.Conv2d(96, 32, 3, padding=p)  # 32+32+32=96 channels (deconv32 + ct32 + pet32)
        self.LReLU9_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn9_1 = nn.InstanceNorm2d(32)
        self.conv9_2 = nn.Conv2d(32, 32, 3, padding=p)
        self.LReLU9_2 = nn.LeakyReLU(0.2, inplace=True)
        
        # 输出层
        self.conv10 = nn.Conv2d(32, 1, 1)
        
    def forward(self, ct, pet):
        """前向传播
        
        Args:
            ct: CT图像 [B, 1, H, W]
            pet: PET图像 [B, 1, H, W]
            
        Returns:
            output: 分割结果 [B, 1, H, W]
        """
        # 准备输入（处理大图像）
        ct_pad, pad_left, pad_right, pad_top, pad_bottom = pad_tensor(ct)
        pet_pad, _, _, _, _ = pad_tensor(pet)
        
        # 检查是否需要downsample
        flag = 0
        if ct_pad.size()[3] > 2200:
            avg = nn.AvgPool2d(2)
            ct_pad = avg(ct_pad)
            pet_pad = avg(pet_pad)
            flag = 1
            ct_pad, _, _, _, _ = pad_tensor(ct_pad)
            pet_pad, _, _, _, _ = pad_tensor(pet_pad)
        
        # ========== 编码阶段 ==========
        # ========== VGG编码器（已关闭，不使用）==========
        # ct_vgg1, ct_vgg2, ct_vgg3 = self.ct_vgg_encoder(ct_pad)  # 64, 128, 256
        # pet_vgg1, pet_vgg2, pet_vgg3 = self.pet_vgg_encoder(pet_pad)  # 64, 128, 256
        
        # 独立的UNet编码器提取主要特征
        ct_conv1, ct_conv2, ct_conv3, ct_conv4 = self.ct_unet_encoder(ct_pad)  # 32, 64, 128, 256
        pet_conv1, pet_conv2, pet_conv3, pet_conv4 = self.pet_unet_encoder(pet_pad)  # 32, 64, 128, 256
        
        # ========== 中期融合层1（最深层，256通道）==========
        # 使用CAM注意力增强UNet特征
        ct_unet_out, _ = self.cam_256(ct_conv4)  # CT UNet 256
        pet_unet_out, _ = self.cam_256(pet_conv4)  # PET UNet 256
        
        # 融合：concat + deconv（使用注意力增强后的UNet特征）
        att_4 = torch.cat([ct_unet_out, pet_unet_out], 1)  # 256*2 = 512
        x = self.deconv4(att_4)  # 512 -> 256
        conv4 = self.bn4_3(x)  # 256
        
        # 上采样
        conv6 = F.interpolate(conv4, scale_factor=2, mode='bilinear', align_corners=False)  # 256
        
        # ========== 中期融合层2（中层，128通道）==========
        # 使用CAM注意力增强UNet特征
        ct_unet_out, _ = self.cam_128(ct_conv3)  # CT UNet 128
        pet_unet_out, _ = self.cam_128(pet_conv3)  # PET UNet 128
        
        # 2路特征融合（使用注意力增强后的UNet特征）
        att_7 = torch.cat([ct_unet_out, pet_unet_out], 1)  # 128*2 = 256
        att_7 = self.att_deconv7(att_7)  # 256 -> 128
        
        # 解码器第1层：上采样特征 + 融合特征
        up7 = torch.cat([self.deconv6(conv6), att_7], 1)  # 128 + 128 = 256
        x = self.bn7_1(self.LReLU7_1(self.conv7_1(up7)))  # 256 -> 128
        conv7 = self.bn7_2(self.LReLU7_2(self.conv7_2(x)))  # 128
        
        # 上采样
        conv7 = F.interpolate(conv7, scale_factor=2, mode='bilinear', align_corners=False)  # 128
        
        # ========== 中期融合层3（浅层，64通道）==========
        # 使用CAM注意力增强UNet特征
        ct_unet_out, _ = self.cam_64(ct_conv2)  # CT UNet 64
        pet_unet_out, _ = self.cam_64(pet_conv2)  # PET UNet 64
        
        # 2路特征融合（使用注意力增强后的UNet特征）
        att_8 = torch.cat([ct_unet_out, pet_unet_out], 1)  # 64*2 = 128
        att_8 = self.att_deconv8(att_8)  # 128 -> 64
        
        # 解码器第2层：上采样特征 + 融合特征
        up8 = torch.cat([self.deconv7(conv7), att_8], 1)  # 64 + 64 = 128
        x = self.bn8_1(self.LReLU8_1(self.conv8_1(up8)))  # 128 -> 64
        conv8 = self.bn8_2(self.LReLU8_2(self.conv8_2(x)))  # 64
        
        # 上采样
        conv8 = F.interpolate(conv8, scale_factor=2, mode='bilinear', align_corners=False)  # 64
        
        # ========== 最浅层融合（32通道）==========
        # 使用CAM注意力增强UNet特征
        ct_unet_out, _ = self.cam_32(ct_conv1)  # CT UNet 32
        pet_unet_out, _ = self.cam_32(pet_conv1)  # PET UNet 32
        
        # 融合CT和PET的最浅层特征（使用注意力增强后的UNet特征）
        up9 = torch.cat([self.deconv8(conv8), ct_unet_out, pet_unet_out], 1)  # 32 + 32 + 32 = 96
        x = self.bn9_1(self.LReLU9_1(self.conv9_1(up9)))  # 96 -> 32
        conv9 = self.LReLU9_2(self.conv9_2(x))  # 32
        
        # 输出层
        output = self.conv10(conv9)  # 32 -> 1
        
        # 恢复padding
        output = pad_tensor_back(output, pad_left, pad_right, pad_top, pad_bottom)
        
        # 如果之前downsample了，需要upsample回来
        if flag == 1:
            output = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=False)
        
        return output


# 为了兼容性，保留旧类名
class DualEncoderUNet(DualIndependentEncoderUNet):
    """别名"""
    pass


class Unet_resize_conv(DualEncoderUNet):
    """兼容性别名，指向新的双独立编码器架构"""
    pass
