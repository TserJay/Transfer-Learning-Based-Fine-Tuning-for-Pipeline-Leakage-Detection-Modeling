"""
SimpleLoraNet: Pipeline leakage detection with LoRA on deep layers only

Architecture: ConvBlock → Res2Net stages → BiLSTM → LoRA at deep layer → Classifier
LoRA只放在Layer4后的深层位置，不分散放置
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple, Optional


class ModelConfig:
    INPUT_CHANNELS = 3
    INITIAL_CONV_CHANNELS = 64
    BASE_WIDTH = 128
    SCALE = 2
    EXPANSION = 4
    
    LSTM_HIDDEN_SIZE = 128
    LSTM_NUM_LAYERS = 1
    LSTM_DROPOUT = 0.5
    
    MLP_HIDDEN_DIM = 64
    MLP_OUTPUT_DIM = 32
    
    NUM_CLASSES = 12
    
    LORA_R = 16
    LORA_ALPHA = 4
    LORA_DROPOUT = 0.1


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class SE_Block(nn.Module):
    def __init__(self, inchannel: int, ratio: int = 16):
        super().__init__()
        reduced_dim = max(1, inchannel // ratio)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(inchannel, reduced_dim, bias=False),
            nn.ReLU(),
            nn.Linear(reduced_dim, inchannel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _ = x.size()
        y = self.gap(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class LoRAConv1d(nn.Module):
    def __init__(self, conv: nn.Conv1d, r: int = 16, alpha: int = 4, dropout: float = 0.1):
        super().__init__()
        self.original = conv
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        
        if r > 0:
            self.lora_A = nn.Conv1d(conv.in_channels, r, kernel_size=1, bias=False)
            self.lora_B = nn.Conv1d(r, conv.out_channels, kernel_size=1, bias=False)
            nn.init.normal_(self.lora_A.weight, std=1.0 / r)
            nn.init.zeros_(self.lora_B.weight)
        else:
            self.lora_A = None
            self.lora_B = None
        
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_output = self.original(x)
        if self.r > 0 and self.lora_A is not None and self.lora_B is not None:
            lora_output = self.dropout(self.lora_B(self.lora_A(x))) * self.scaling
            return original_output + lora_output
        return original_output


class Bottle2neck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        baseWidth: int = 128,
        scale: int = 2,
        stype: str = 'normal',
        use_lora: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 4,
        lora_dropout: float = 0.1,
    ):
        super().__init__()

        width = int(math.floor(planes * (baseWidth / 64.0)))
        width = max(1, width)

        self.conv1 = nn.Conv1d(inplanes, width * scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(width * scale)

        self.nums = 1 if scale == 1 else scale - 1
        self.stype = stype
        self.scale = scale
        self.width = width

        self.relu = nn.ReLU(inplace=True)

        if stype == 'stage':
            self.pool = nn.AvgPool1d(kernel_size=5, stride=stride, padding=2)
            self.se2 = SE_Block(width)

        self.convs = nn.ModuleList([
            nn.Conv1d(width, width, kernel_size=5, stride=stride, padding=2, bias=False)
            for _ in range(self.nums)
        ])
        self.bns = nn.ModuleList([
            nn.BatchNorm1d(width) for _ in range(self.nums)
        ])

        self.conv3 = nn.Conv1d(width * scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)

        if downsample is None and (inplanes != planes * self.expansion or stride != 1):
            self.downsample = nn.Sequential(
                nn.Conv1d(inplanes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * self.expansion)
            )
        else:
            self.downsample = downsample

        self.use_lora = use_lora
        if use_lora:
            self.conv1_lora = LoRAConv1d(self.conv1, r=lora_r, alpha=lora_alpha, dropout=lora_dropout)
            self.conv3_lora = LoRAConv1d(self.conv3, r=lora_r, alpha=lora_alpha, dropout=lora_dropout)
        else:
            self.conv1_lora = None
            self.conv3_lora = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        if self.use_lora and self.conv1_lora is not None:
            out = self.relu(self.bn1(self.conv1_lora(x)))
        else:
            out = self.relu(self.bn1(self.conv1(x)))
            
        spx = torch.split(out, self.width, 1)
        sp = spx[0]

        for i in range(self.nums):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.relu(self.bns[i](self.convs[i](sp)))
            out = sp if i == 0 else torch.cat((out, sp), 1)

        if self.scale != 1 and self.stype == 'normal':
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == 'stage':
            out = torch.cat((out, self.pool(self.se2(spx[self.nums]))), 1)

        if self.use_lora and self.conv3_lora is not None:
            out = self.bn3(self.conv3_lora(out))
        else:
            out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            residual = self.downsample(x)

        return self.relu(out + residual)


class SimpleLoraNet(nn.Module):
    """
    Architecture: ConvBlock -> Res2Net (no LoRA) -> BiLSTM -> Deep Layer with LoRA -> Classifier
    LoRA只放在最后深层位置，其他层无LoRA
    """

    def __init__(
        self,
        pretrained: bool = False,
        in_channels: int = ModelConfig.INPUT_CHANNELS,
        baseWidth: int = ModelConfig.BASE_WIDTH,
        scale: int = ModelConfig.SCALE,
        lora_r: int = ModelConfig.LORA_R,
        lora_alpha: int = ModelConfig.LORA_ALPHA,
        lora_dropout: float = ModelConfig.LORA_DROPOUT,
        num_classes: int = ModelConfig.NUM_CLASSES,
    ):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=7, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )

        self.layer1 = self._make_layer(64, 64, 1, baseWidth, scale, use_lora=False)
        self.layer2 = self._make_layer(256, 128, 1, baseWidth, scale, use_lora=False)
        self.layer3 = self._make_layer(512, 64, 1, baseWidth, scale, use_lora=False)
        self.layer4 = self._make_layer(128, 8, 1, baseWidth, scale, use_lora=True, 
                                        lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)

        self.BiLSTM1 = nn.LSTM(
            input_size=256,
            hidden_size=ModelConfig.LSTM_HIDDEN_SIZE,
            num_layers=ModelConfig.LSTM_NUM_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=ModelConfig.LSTM_DROPOUT if ModelConfig.LSTM_NUM_LAYERS > 1 else 0
        )

        self.BiLSTM2 = nn.LSTM(
            input_size=512,
            hidden_size=256,
            num_layers=ModelConfig.LSTM_NUM_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=ModelConfig.LSTM_DROPOUT if ModelConfig.LSTM_NUM_LAYERS > 1 else 0
        )

        self.BiLSTM3 = nn.LSTM(
            input_size=256,
            hidden_size=ModelConfig.LSTM_HIDDEN_SIZE // 2,
            num_layers=ModelConfig.LSTM_NUM_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=ModelConfig.LSTM_DROPOUT if ModelConfig.LSTM_NUM_LAYERS > 1 else 0
        )

        self.ap = nn.AdaptiveAvgPool1d(output_size=1)
        self.projetion_pos_1 = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32)
        )
        self.fc1 = nn.Linear(32, num_classes)

        self._init_weights()

    def _make_layer(
        self, inplanes: int, planes: int, blocks: int,
        baseWidth: int, scale: int, use_lora: bool = False,
        lora_r: int = 16, lora_alpha: int = 4, lora_dropout: float = 0.1,
        stride: int = 1
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or inplanes != planes * Bottle2neck.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(inplanes, planes * Bottle2neck.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * Bottle2neck.expansion),
            )

        layers_list = [
            Bottle2neck(
                inplanes, planes, stride, downsample=downsample,
                stype='stage', baseWidth=baseWidth, scale=scale,
                use_lora=use_lora, lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout
            )
        ]

        for _ in range(1, blocks):
            layers_list.append(
                Bottle2neck(
                    planes * Bottle2neck.expansion, planes,
                    baseWidth=baseWidth, scale=scale,
                    use_lora=use_lora, lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout
                )
            )

        return nn.Sequential(*layers_list)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.conv_block(x)

        x = self.layer1(x)
        x, _ = self.BiLSTM1(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)

        x = self.layer2(x)
        x, _ = self.BiLSTM2(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)

        x = self.layer3(x)
        x, _ = self.BiLSTM3(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)

        x = self.layer4(x)

        x = self.ap(x).squeeze(-1)
        view = self.projetion_pos_1(x)
        pos = self.fc1(view)
        return pos, view

    def print_lora_info(self):
        lora_params = 0
        total_params = 0
        for name, param in self.named_parameters():
            total_params += param.numel()
            if 'lora' in name.lower():
                lora_params += param.numel()
        
        print("=" * 60)
        print("LoRA Parameter Statistics")
        print("=" * 60)
        print(f"Total parameters:        {total_params:,}")
        print(f"LoRA parameters:         {lora_params:,} ({lora_params/total_params*100:.2f}%)")
        print("=" * 60)


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


if __name__ == '__main__':
    model = SimpleLoraNet(pretrained=False)
    
    print("Model: SimpleLoraNet (LoRA only at deep layer - Layer4)")
    print("=" * 60)
    print(f"Total parameters: {count_parameters(model):,}")
    print(f"Trainable parameters: {count_parameters(model, trainable_only=True):,}")
    
    model.print_lora_info()
    
    x = torch.randn(2, 3, 1792)
    pos, feat = model(x)
    print(f"\nInput shape: {list(x.shape)}")
    print(f"Position output shape: {list(pos.shape)}")
    print(f"Feature shape: {list(feat.shape)}")