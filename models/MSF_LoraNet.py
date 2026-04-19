"""
MSF_LoraNet: Multi-Scale Fusion with LoRA Fine-tuning

Combines:
- Multi-Scale Transformer Fusion architecture
- LoRA fine-tuning at conv layers
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
    
    LORA_R = 4
    LORA_ALPHA = 1
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


class CoordinateAttention(nn.Module):
    def __init__(self, inchannel: int, reduction_ratio: int = 16):
        super().__init__()
        reduced_channels = max(inchannel // reduction_ratio, 8)
        self.conv1d_h = nn.Conv1d(inchannel, reduced_channels, kernel_size=1)
        self.conv1d_w = nn.Conv1d(inchannel, reduced_channels, kernel_size=1)
        self.fc = nn.Sequential(
            nn.Conv1d(reduced_channels * 2, inchannel, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h = x.size()
        x_h = F.adaptive_avg_pool1d(x, 1)
        x_w = x.mean(dim=2, keepdim=True)
        x_h = self.conv1d_h(x_h)
        x_w = self.conv1d_w(x_w)
        y = torch.cat([x_h, x_w], dim=1)
        y = self.fc(y)
        return x * y.expand_as(x)


class LoRAConv1d(nn.Module):
    """LoRA adaptation for Conv1d layers."""
    def __init__(self, conv: nn.Conv1d, r: int = 4, alpha: int = 1, dropout: float = 0.1):
        super().__init__()
        self.original = conv
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        
        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False
        
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


class Bottle2neckMSF(nn.Module):
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
        drop_prob: float = 0.2,
        lora_r: int = 4,
        lora_alpha: int = 1,
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
        self.drop_prob = drop_prob

        self.relu = nn.ReLU(inplace=True)

        if stype == 'stage':
            self.pool = nn.AvgPool1d(kernel_size=5, stride=stride, padding=2)
            self.se2 = SE_Block(width)
            self.co_atten = CoordinateAttention(width)

        self.convs = nn.ModuleList([
            nn.Conv1d(width, width, kernel_size=5, stride=stride, padding=2, bias=False)
            for _ in range(self.nums)
        ])
        self.bns = nn.ModuleList([
            nn.BatchNorm1d(width) for _ in range(self.nums)
        ])

        self.conv3 = nn.Conv1d(width * scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.dropout = nn.Dropout(p=drop_prob)

        if downsample is None and (inplanes != planes * self.expansion or stride != 1):
            self.downsample = nn.Sequential(
                nn.Conv1d(inplanes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * self.expansion)
            )
        else:
            self.downsample = downsample

        if lora_r > 0:
            self.conv1_lora = LoRAConv1d(self.conv1, r=lora_r, alpha=lora_alpha, dropout=lora_dropout)
            self.conv3_lora = LoRAConv1d(self.conv3, r=lora_r, alpha=lora_alpha, dropout=lora_dropout)
        else:
            self.conv1_lora = None
            self.conv3_lora = None

    def _drop_path(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        if self.conv1_lora is not None:
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
            attention_input = self.se2(spx[self.nums])
            attention_input = self.co_atten(attention_input)
            gate = torch.sigmoid(attention_input.mean(dim=-1, keepdim=True))
            attention_input = attention_input * (1 - gate) + spx[self.nums] * gate
            out = torch.cat((out, self.pool(attention_input)), 1)

        out = self.dropout(self.bn3(self.conv3(out)))

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self._drop_path(out) + residual
        return self.relu(out)


class LayerNormLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, 
                 bidirectional: bool = True, dropout: float = 0.5):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers=num_layers,
            batch_first=True, bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )
        lstm_output_size = hidden_size * (2 if bidirectional else 1)
        self.layer_norm = nn.LayerNorm(lstm_output_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_perm = x.permute(0, 2, 1)
        x, _ = self.lstm(x_perm)
        x = self.layer_norm(x)
        y = x.permute(0, 2, 1)
        return y, x


class MultiScaleTransformerFusion(nn.Module):
    def __init__(self, input_dims: List[int], d_model: int = 64, nhead: int = 2,
                 num_layers: int = 1, dim_feedforward: int = 512, dropout: float = 0.1):
        super().__init__()
        self.projections = nn.ModuleList([nn.Linear(in_dim, d_model) for in_dim in input_dims])

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fusion_fc = nn.Linear(d_model, d_model)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        batch_size = features[0].shape[0]
        proj_feats = []
        lengths = []

        for x, proj in zip(features, self.projections):
            x_proj = proj(x)
            proj_feats.append(x_proj)
            lengths.append(x.shape[1])

        max_len = max(lengths)
        padded_feats = []
        masks = []

        for x, l in zip(proj_feats, lengths):
            pad_len = max_len - l
            if pad_len > 0:
                x = F.pad(x, (0, 0, 0, pad_len))
            padded_feats.append(x)
            mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=x.device)
            mask[:, l:] = True
            masks.append(mask)

        stacked_feats = torch.stack(padded_feats, dim=1)
        stacked_feats = stacked_feats.view(batch_size * len(features), max_len, -1)
        all_masks = torch.cat(masks, dim=0)

        transformer_out = self.transformer(stacked_feats, src_key_padding_mask=all_masks)
        pooled_output = transformer_out[:, 0, :]

        pooled_output = pooled_output.view(batch_size, len(features), -1)
        fused = pooled_output.mean(dim=1)
        return self.fusion_fc(fused)


class MSF_LoraNet(nn.Module):
    """
    Main model: Multi-Scale Fusion with LoRA fine-tuning.
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

        self.conv1 = nn.Conv1d(3, 64, kernel_size=7, stride=2, padding=4, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(64, 64, 1, baseWidth, scale, lora_r, lora_alpha, lora_dropout, stride=1)
        self.layer2 = self._make_layer(256, 128, 1, baseWidth, scale, lora_r, lora_alpha, lora_dropout, stride=2)
        self.layer3 = self._make_layer(512, 64, 1, baseWidth, scale, lora_r, lora_alpha, lora_dropout, stride=2)
        self.layer4 = self._make_layer(256, 8, 1, baseWidth, scale, lora_r, lora_alpha, lora_dropout, stride=2)

        self.BiLSTM1 = LayerNormLSTM(256, 128)
        self.BiLSTM2 = LayerNormLSTM(512, 256)
        self.BiLSTM3 = LayerNormLSTM(256, 128)
        self.BiLSTM4 = LayerNormLSTM(32, 56)

        self.fusion_module = MultiScaleTransformerFusion(
            input_dims=[256, 512, 256, 112],
            d_model=64, nhead=2, num_layers=1
        )

        self.ap = nn.AdaptiveAvgPool1d(output_size=1)
        self.projetion_pos_1 = MLP(input_dim=176, hidden_dim=64, output_dim=32, num_layers=1)
        self.fc1 = nn.Linear(32, num_classes)

        self._init_weights()

    def _make_layer(
        self, inplanes: int, planes: int, blocks: int,
        baseWidth: int, scale: int, lora_r: int, lora_alpha: float, lora_dropout: float,
        stride: int = 1
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or inplanes != planes * Bottle2neckMSF.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(inplanes, planes * Bottle2neckMSF.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * Bottle2neckMSF.expansion),
            )

        layers_list = [
            Bottle2neckMSF(
                inplanes, planes, stride, downsample=downsample,
                stype='stage', baseWidth=baseWidth, scale=scale,
                lora_r=lora_r, lora_alpha=int(lora_alpha), lora_dropout=lora_dropout
            )
        ]

        for _ in range(1, blocks):
            layers_list.append(
                Bottle2neckMSF(
                    planes * Bottle2neckMSF.expansion, planes,
                    baseWidth=baseWidth, scale=scale,
                    lora_r=lora_r, lora_alpha=int(lora_alpha), lora_dropout=lora_dropout
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
        x = self.relu(self.bn1(self.conv1(x)))

        x = self.layer1(x)
        x, x1 = self.BiLSTM1(x)

        x = self.layer2(x)
        x, x2 = self.BiLSTM2(x)

        x = self.layer3(x)
        x, x3 = self.BiLSTM3(x)

        x = self.layer4(x)
        x, x4 = self.BiLSTM4(x)

        fused = self.fusion_module([x1, x2, x3, x4])
        x = self.ap(x).squeeze(-1)
        x = torch.cat([x, fused], dim=1)
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
    model = MSF_LoraNet(pretrained=False)
    
    print("Model: MSF_LoraNet")
    print("=" * 60)
    print(f"Total parameters: {count_parameters(model):,}")
    print(f"Trainable parameters: {count_parameters(model, trainable_only=True):,}")
    
    model.print_lora_info()
    
    x = torch.randn(32, 3, 1792)
    pos, feat = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Position output shape: {pos.shape}")
    print(f"Feature shape: {feat.shape}")