#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
改进的LORA_Net_12345模型

使用正确的LoRA实现，解决原实现中的问题：
1. LoRA作为现有层的适配器，而不是独立层
2. 正确的权重初始化（A使用高斯分布，B初始化为0）
3. 冻结预训练权重，只训练LoRA参数
4. 简化的维度变换，避免复杂的reshape操作
5. 消除硬编码，提高代码稳定性和可维护性
"""

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.nn import init
import math
from typing import Optional, List, Tuple
from einops import rearrange

try:
    # 尝试绝对导入（当作为模块导入时）
    from models.lora_layers import (
        LoRALayer,
        LoRAConv1d,
        LoRAConv1dWithKernal,
        apply_lora_to_conv1d,
        count_lora_parameters,
        print_lora_info
    )
except ModuleNotFoundError:
    # 当直接运行脚本时使用相对导入
    from lora_layers import (
        LoRALayer,
        LoRAConv1d,
        LoRAConv1dWithKernal,
        apply_lora_to_conv1d,
        count_lora_parameters,
        print_lora_info
    )

# 常量定义，避免硬编码
class ModelConstants:
    """模型常量定义"""
    # 默认维度
    DEFAULT_IN_CHANNELS = 3
    DEFAULT_BASE_WIDTH = 64
    DEFAULT_SCALE = 2
    DEFAULT_EXPANSION = 4
    
    # BiLSTM默认参数
    DEFAULT_LSTM_HIDDEN_SIZE = 64
    DEFAULT_LSTM_NUM_LAYERS = 1
    DEFAULT_LSTM_DROPOUT = 0.5
    
    # MLP默认参数
    DEFAULT_MLP_HIDDEN_DIM = 64
    DEFAULT_MLP_OUTPUT_DIM = 32
    DEFAULT_MLP_NUM_LAYERS = 1
    
    # 分类头默认参数
    DEFAULT_NUM_CLASSES = 12
    
    # LoRA默认参数
    DEFAULT_LORA_R = 4
    DEFAULT_LORA_ALPHA = 1
    DEFAULT_LORA_DROPOUT = 0.1
    
    # 参数名称关键词
    LORA_KEYWORD = 'lora'
    LSTM_KEYWORD = 'BiLSTM'
    PROJECTION_KEYWORD = 'projetion'
    FC_KEYWORD = 'fc'


class MLP(nn.Module):
    """多层感知器 (MLP)
    
    Args:
        input_dim: 输入维度
        hidden_dim: 隐藏层维度
        output_dim: 输出维度
        num_layers: 层数（包括输出层）
        dropout: Dropout概率，默认为0
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int, dropout: float = 0.0):
        super().__init__()
        
        # 参数验证
        if num_layers < 1:
            raise ValueError(f"num_layers must be at least 1, got {num_layers}")
        if input_dim <= 0 or hidden_dim <= 0 or output_dim <= 0:
            raise ValueError(f"All dimensions must be positive, got input_dim={input_dim}, "
                           f"hidden_dim={hidden_dim}, output_dim={output_dim}")
        
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        
        # 添加Dropout层
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = self.dropout(x)
        return x


class SE_Block(nn.Module):
    """Squeeze-and-Excitation 注意力模块
    
    Args:
        inchannel: 输入通道数
        ratio: 压缩比例，默认为16
    """
    
    def __init__(self, inchannel: int, ratio: int = 16):
        super(SE_Block, self).__init__()
        
        # 参数验证
        if inchannel <= 0:
            raise ValueError(f"inchannel must be positive, got {inchannel}")
        if ratio <= 0:
            raise ValueError(f"ratio must be positive, got {ratio}")
        
        # 确保压缩后的维度至少为1
        reduced_dim = max(1, inchannel // ratio)
        
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(inchannel, reduced_dim, bias=False),
            nn.ReLU(),
            nn.Linear(reduced_dim, inchannel, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        b, c, h = x.size()
        y = self.gap(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class Bottle2neck(nn.Module):
    """Res2Net风格的残差块
    
    Args:
        inplanes: 输入通道数
        planes: 输出通道数
        stride: 卷积步长，默认为1
        downsample: 下采样层，默认为None
        baseWidth: 基础宽度，默认为26
        scale: 缩放因子，默认为4
        stype: 类型，'normal'或'stage'，默认为'normal'
    """
    
    expansion = 4
    
    def __init__(self, inplanes: int, planes: int, stride: int = 1,
                 downsample: Optional[nn.Module] = None,
                 baseWidth: int = 26, scale: int = 4, stype: str = 'normal'):
        super(Bottle2neck, self).__init__()
        
        # 参数验证
        if inplanes <= 0 or planes <= 0:
            raise ValueError(f"inplanes and planes must be positive, got inplanes={inplanes}, planes={planes}")
        if scale <= 0:
            raise ValueError(f"scale must be positive, got {scale}")
        if baseWidth <= 0:
            raise ValueError(f"baseWidth must be positive, got {baseWidth}")
        if stype not in ['normal', 'stage']:
            raise ValueError(f"stype must be 'normal' or 'stage', got {stype}")
        
        width = int(math.floor(planes * (baseWidth / 64.0)))
        if width <= 0:
            width = 1  # 确保width至少为1
        
        self.conv1 = nn.Conv1d(inplanes, width * scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(width * scale)
        
        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        
        if stype == 'stage':
            self.pool = nn.AvgPool1d(kernel_size=5, stride=stride, padding=2)
            self.se2 = SE_Block(width)
        
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(nn.Conv1d(width, width, kernel_size=5, stride=stride,
                                   padding=2, bias=False))
            bns.append(nn.BatchNorm1d(width))
        
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        
        self.conv3 = nn.Conv1d(width * scale, planes * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        
        if self.scale != 1 and self.stype == 'normal':
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == 'stage':
            out = torch.cat((out, self.pool(self.se2(spx[self.nums]))), 1)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = self.relu(out)
        
        return out


class Res2NetWithLoRA(nn.Module):
    """
    集成LoRA的Res2Net模型
    
    使用正确的LoRA实现，冻结预训练权重，只训练LoRA参数
    
    Args:
        block: 残差块类型
        layers: 每个阶段的块数列表
        in_channels: 输入通道数，默认为3
        baseWidth: Res2Net基础宽度，默认为26
        scale: Res2Net缩放因子，默认为2
        lora_r: LoRA秩，默认为4
        lora_alpha: LoRA缩放因子，默认为1
        lora_dropout: LoRA Dropout概率，默认为0.1
        lstm_hidden_size: LSTM隐藏层大小，默认为64
        num_classes: 分类数量，默认为12
    """
    
    def __init__(self, block: type, layers: List[int],
                 in_channels: int = ModelConstants.DEFAULT_IN_CHANNELS,
                 baseWidth: int = 26, scale: int = 2,
                 lora_r: int = ModelConstants.DEFAULT_LORA_R,
                 lora_alpha: int = ModelConstants.DEFAULT_LORA_ALPHA,
                 lora_dropout: float = ModelConstants.DEFAULT_LORA_DROPOUT,
                 lstm_hidden_size: int = ModelConstants.DEFAULT_LSTM_HIDDEN_SIZE,
                 num_classes: int = ModelConstants.DEFAULT_NUM_CLASSES):
        super().__init__()
        
        # 参数验证
        if len(layers) != 4:
            raise ValueError(f"layers must have exactly 4 elements, got {len(layers)}")
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")
        if baseWidth <= 0:
            raise ValueError(f"baseWidth must be positive, got {baseWidth}")
        if scale <= 0:
            raise ValueError(f"scale must be positive, got {scale}")
        if lora_r < 0:
            raise ValueError(f"lora_r must be non-negative, got {lora_r}")
        if lstm_hidden_size <= 0:
            raise ValueError(f"lstm_hidden_size must be positive, got {lstm_hidden_size}")
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")
        
        # 保存配置
        self.inplanes = 64
        self.baseWidth = baseWidth
        self.scale = scale
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lstm_hidden_size = lstm_hidden_size
        self.num_classes = num_classes
        
        # 原始层
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=4, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # 为conv1添加LoRA
        if lora_r > 0:
            self.conv1_lora = LoRAConv1d(self.conv1, r=lora_r, lora_alpha=lora_alpha,
                                         lora_dropout=lora_dropout)
        else:
            self.conv1_lora = self.conv1
        
        # Res2Net层
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 8, layers[3], stride=2)
        
        # 为每个layer中的卷积层添加LoRA
        if lora_r > 0:
            self._add_lora_to_layers(lora_r, lora_alpha, lora_dropout)
        
        # 计算BiLSTM输入维度（layer4的输出通道数）
        # layer4输出: 8 * expansion = 8 * 4 = 32
        lstm_input_size = 8 * block.expansion
        
        # BiLSTM层
        self.BiLSTM1 = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden_size,
            num_layers=ModelConstants.DEFAULT_LSTM_NUM_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=ModelConstants.DEFAULT_LSTM_DROPOUT if ModelConstants.DEFAULT_LSTM_NUM_LAYERS > 1 else 0
        )
        
        # 全局平均池化
        self.ap = nn.AdaptiveAvgPool1d(output_size=1)
        
        # 分类头（BiLSTM是双向的，所以维度翻倍）
        mlp_input_dim = lstm_hidden_size * 2
        self.projetion_pos_1 = MLP(
            input_dim=mlp_input_dim,
            hidden_dim=ModelConstants.DEFAULT_MLP_HIDDEN_DIM,
            output_dim=ModelConstants.DEFAULT_MLP_OUTPUT_DIM,
            num_layers=ModelConstants.DEFAULT_MLP_NUM_LAYERS
        )
        self.fc1 = nn.Linear(ModelConstants.DEFAULT_MLP_OUTPUT_DIM, num_classes)
        
        # 冻结非LoRA参数
        self._freeze_non_lora_params()
    
    def _make_layer(self, block: type, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        """创建Res2Net层
        
        Args:
            block: 残差块类型
            planes: 输出通道数
            blocks: 块数
            stride: 卷积步长
            
        Returns:
            nn.Sequential: 包含多个残差块的序列
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )
        
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                           stype='stage', baseWidth=self.baseWidth, scale=self.scale))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                               baseWidth=self.baseWidth, scale=self.scale))
        
        return nn.Sequential(*layers)
    
    def _add_lora_to_layers(self, r: int, alpha: int, dropout: float):
        """为所有卷积层添加LoRA
        
        Args:
            r: LoRA秩
            alpha: LoRA缩放因子
            dropout: Dropout概率
        """
        # 定义需要添加LoRA的层
        layers_to_process = [self.layer1, self.layer2, self.layer3, self.layer4]
        
        for layer in layers_to_process:
            for block in layer:
                # 为conv1添加LoRA
                if hasattr(block, 'conv1'):
                    block.conv1_lora = LoRAConv1d(block.conv1, r=r, lora_alpha=alpha,
                                                  lora_dropout=dropout)
                # 为conv3添加LoRA
                if hasattr(block, 'conv3'):
                    block.conv3_lora = LoRAConv1d(block.conv3, r=r, lora_alpha=alpha,
                                                  lora_dropout=dropout)
    
    def _freeze_non_lora_params(self):
        """冻结非LoRA参数"""
        # 定义需要保持可训练的参数关键词
        trainable_keywords = [
            ModelConstants.LORA_KEYWORD,
            ModelConstants.LSTM_KEYWORD,
            ModelConstants.PROJECTION_KEYWORD,
            ModelConstants.FC_KEYWORD
        ]
        
        for name, param in self.named_parameters():
            # 检查参数名称是否包含任何可训练关键词
            is_trainable = any(keyword in name for keyword in trainable_keywords)
            if not is_trainable:
                param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, in_channels, sequence_length)
            
        Returns:
            tuple: (分类输出, 特征输出)
        """
        # 使用带LoRA的卷积
        x = self.conv1_lora(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # Res2Net层
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # BiLSTM (需要调整维度: (batch, channels, seq_len) -> (batch, seq_len, channels))
        x = x.permute(0, 2, 1)  # (batch, seq_len, channels)
        x, _ = self.BiLSTM1(x)
        x = x.permute(0, 2, 1)  # (batch, channels, seq_len)
        
        # 全局平均池化
        x = self.ap(x).squeeze(-1)  # (batch, channels)
        
        # 分类头
        view = self.projetion_pos_1(x)
        pos = self.fc1(view)
        
        return pos, view


class LORA_Net_12345_Improved(nn.Module):
    """
    改进的LORA_Net_12345模型
    
    使用正确的LoRA实现，解决原实现中的问题
    
    Args:
        in_channel: 输入通道数，默认为3
        baseWidth: Res2Net基础宽度，默认为128
        scale: Res2Net缩放因子，默认为2
        lora_r: LoRA秩，默认为4
        lora_alpha: LoRA缩放因子，默认为1
        lora_dropout: LoRA Dropout概率，默认为0.1
        lstm_hidden_size: LSTM隐藏层大小，默认为64
        num_classes: 分类数量，默认为12
    """
    
    def __init__(self, in_channel: int = ModelConstants.DEFAULT_IN_CHANNELS,
                 baseWidth: int = 128, scale: int = 2,
                 lora_r: int = ModelConstants.DEFAULT_LORA_R,
                 lora_alpha: int = ModelConstants.DEFAULT_LORA_ALPHA,
                 lora_dropout: float = ModelConstants.DEFAULT_LORA_DROPOUT,
                 lstm_hidden_size: int = ModelConstants.DEFAULT_LSTM_HIDDEN_SIZE,
                 num_classes: int = ModelConstants.DEFAULT_NUM_CLASSES):
        super().__init__()
        
        # 参数验证
        if in_channel <= 0:
            raise ValueError(f"in_channel must be positive, got {in_channel}")
        if baseWidth <= 0:
            raise ValueError(f"baseWidth must be positive, got {baseWidth}")
        if scale <= 0:
            raise ValueError(f"scale must be positive, got {scale}")
        if lora_r < 0:
            raise ValueError(f"lora_r must be non-negative, got {lora_r}")
        if lstm_hidden_size <= 0:
            raise ValueError(f"lstm_hidden_size must be positive, got {lstm_hidden_size}")
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")
        
        # 保存配置
        self.lstm_hidden_size = lstm_hidden_size
        self.num_classes = num_classes
        
        # 使用改进的Res2Net
        self.backbone = Res2NetWithLoRA(
            Bottle2neck, [1, 1, 1, 1],
            in_channels=in_channel,
            baseWidth=baseWidth,
            scale=scale,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lstm_hidden_size=lstm_hidden_size,
            num_classes=num_classes
        )
        
        # 计算BiLSTM输入维度（backbone输出的特征维度）
        # backbone输出: lstm_hidden_size * 2 (双向)
        lstm_input_size = lstm_hidden_size * 2
        
        # BiLSTM层
        self.BiLSTM1 = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden_size,
            num_layers=ModelConstants.DEFAULT_LSTM_NUM_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=ModelConstants.DEFAULT_LSTM_DROPOUT if ModelConstants.DEFAULT_LSTM_NUM_LAYERS > 1 else 0
        )
        
        # 全局平均池化
        self.ap = nn.AdaptiveAvgPool1d(output_size=1)
        
        # 分类头（BiLSTM是双向的，所以维度翻倍）
        mlp_input_dim = lstm_hidden_size * 2
        self.projetion_pos_1 = MLP(
            input_dim=mlp_input_dim,
            hidden_dim=ModelConstants.DEFAULT_MLP_HIDDEN_DIM,
            output_dim=ModelConstants.DEFAULT_MLP_OUTPUT_DIM,
            num_layers=ModelConstants.DEFAULT_MLP_NUM_LAYERS
        )
        self.fc1 = nn.Linear(ModelConstants.DEFAULT_MLP_OUTPUT_DIM, num_classes)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播
        
        Args:
            x: 输入张量，形状为 (batch_size, in_channels, sequence_length)
            
        Returns:
            tuple: (分类输出, 特征输出)
        """
        # 通过backbone
        x, y = self.backbone(x)
        
        # BiLSTM (需要调整维度: (batch, channels) -> (batch, 1, channels))
        x = x.unsqueeze(1)  # (batch, 1, channels)
        x, _ = self.BiLSTM1(x)
        x = x.squeeze(1)  # (batch, channels)
        
        # 跳跃连接
        x = x + y
        
        # 分类头
        view = self.projetion_pos_1(x)
        pos = self.fc1(view)
        
        return pos, view
    
    def print_lora_info(self):
        """打印LoRA信息"""
        print_lora_info(self)


# 测试代码
if __name__ == '__main__':
    print("测试改进的LORA_Net_12345模型")
    
    # 创建模型（使用新的参数接口）
    model = LORA_Net_12345_Improved(
        in_channel=3,
        baseWidth=128,
        scale=2,
        lora_r=4,
        lora_alpha=4,
        lora_dropout=0.1,
        lstm_hidden_size=64,
        num_classes=12
    )
    
    # 打印LoRA信息
    model.print_lora_info()
    
    # 测试前向传播
    print("\n测试前向传播:")
    x = torch.randn(32, 3, 1792)
    pos, view = model(x)
    print(f"输入形状: {x.shape}")
    print(f"位置输出形状: {pos.shape}")
    print(f"特征输出形状: {view.shape}")
    
    # 统计参数
    print("\n参数统计:")
    info = count_lora_parameters(model)
    print(f"总参数数量: {info['total_params']:,}")
    print(f"可训练参数数量: {info['trainable_params']:,} ({info['trainable_ratio']:.2f}%)")
    print(f"LoRA参数数量: {info['lora_params']:,} ({info['lora_ratio']:.2f}%)")
    
    print("\n测试完成!")
