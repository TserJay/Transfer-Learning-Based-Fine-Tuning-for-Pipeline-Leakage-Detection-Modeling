import math

import torch
from torch import nn
import torch.nn.functional as F


def _as_tuple(values, expected_length, name):
    if len(values) != expected_length:
        raise ValueError(f"{name} must contain {expected_length} values, got {len(values)}")
    return tuple(values)


class MLP(nn.Module):
    """Simple multi-layer perceptron used by the prediction heads."""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        hidden_dims = [hidden_dim] * (num_layers - 1)
        dims = [input_dim] + hidden_dims + [output_dim]
        self.layers = nn.ModuleList(
            nn.Linear(in_dim, out_dim) for in_dim, out_dim in zip(dims[:-1], dims[1:])
        )

    def forward(self, x):
        for index, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if index < self.num_layers - 1 else layer(x)
        return x


class SE_Block(nn.Module):
    def __init__(self, inchannel, ratio=16):
        super().__init__()
        reduced_channels = max(1, inchannel // ratio)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(inchannel, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, inchannel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        batch_size, channels, _ = x.size()
        y = self.gap(x).view(batch_size, channels)
        y = self.fc(y).view(batch_size, channels, 1)
        return x * y.expand_as(x)


class Bottle2neck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        baseWidth=26,
        scale=4,
        stype='normal',
        kernel_size=5,
    ):
        super().__init__()

        width = int(math.floor(planes * (baseWidth / 64.0)))
        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(inplanes, width * scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(width * scale)
        self.se1 = SE_Block(width * scale)

        self.nums = 1 if scale == 1 else scale - 1
        if stype == 'stage':
            self.pool = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=padding)
            self.se2 = SE_Block(width)

        self.convs = nn.ModuleList(
            nn.Conv1d(width, width, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
            for _ in range(self.nums)
        )
        self.bns = nn.ModuleList(nn.BatchNorm1d(width) for _ in range(self.nums))

        self.conv3 = nn.Conv1d(width * scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.se3 = SE_Block(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x):
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
            out = sp if i == 0 else torch.cat((out, sp), 1)

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


class Res2Net(nn.Module):
    def __init__(
        self,
        block,
        layers,
        input_channels=3,
        stem_channels=64,
        stage_planes=(64, 128, 64, 8),
        baseWidth=26,
        scale=2,
        lstm_input_sizes=(897, 448, 224),
        lstm_hidden_sizes=(448, 224, 112),
        dropout=0,
    ):
        super().__init__()
        self.inplanes = stem_channels
        self.baseWidth = baseWidth
        self.scale = scale
        self.stage_planes = _as_tuple(stage_planes, 4, "stage_planes")
        self.lstm_input_sizes = _as_tuple(lstm_input_sizes, 3, "lstm_input_sizes")
        self.lstm_hidden_sizes = _as_tuple(lstm_hidden_sizes, 3, "lstm_hidden_sizes")

        self.conv1 = nn.Conv1d(
            input_channels,
            stem_channels,
            kernel_size=3,
            stride=2,
            padding=2,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(stem_channels)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, self.stage_planes[0], layers[0])
        self.BiLSTM1 = nn.LSTM(
            input_size=self.lstm_input_sizes[0],
            hidden_size=self.lstm_hidden_sizes[0],
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.layer2 = self._make_layer(block, self.stage_planes[1], layers[1], stride=2)
        self.BiLSTM2 = nn.LSTM(
            input_size=self.lstm_input_sizes[1],
            hidden_size=self.lstm_hidden_sizes[1],
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.layer3 = self._make_layer(block, self.stage_planes[2], layers[2], stride=2)
        self.BiLSTM3 = nn.LSTM(
            input_size=self.lstm_input_sizes[2],
            hidden_size=self.lstm_hidden_sizes[2],
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.layer4 = self._make_layer(block, self.stage_planes[3], layers[3], stride=2)

        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = [
            block(
                self.inplanes,
                planes,
                stride,
                downsample=downsample,
                stype='stage',
                baseWidth=self.baseWidth,
                scale=self.scale,
            )
        ]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, baseWidth=self.baseWidth, scale=self.scale))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x, _ = self.BiLSTM1(x)

        x = self.layer2(x)
        x, _ = self.BiLSTM2(x)

        x = self.layer3(x)
        x, _ = self.BiLSTM3(x)

        x = self.layer4(x)
        return x


class DualPiecesHead(nn.Module):
    def __init__(
        self,
        sequence_input_size=112,
        sequence_hidden_size=64,
        position_hidden_dim=64,
        position_classes=12,
        class_input_dim=113,
        class_hidden_dim=64,
        class_classes=4,
        dropout=0,
    ):
        super().__init__()
        self.sequence_encoder = nn.LSTM(
            input_size=sequence_input_size,
            hidden_size=sequence_hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.pool = nn.AdaptiveAvgPool1d(output_size=1)
        self.position_projection = MLP(
            input_dim=sequence_hidden_size * 2,
            hidden_dim=position_hidden_dim,
            output_dim=position_classes,
            num_layers=1,
        )
        self.class_projection = MLP(
            input_dim=class_input_dim,
            hidden_dim=class_hidden_dim,
            output_dim=class_classes,
            num_layers=1,
        )

    def forward(self, x):
        x, _ = self.sequence_encoder(x.permute(1, 0, 2))
        features = self.pool(x.permute(1, 2, 0)).squeeze(-1)
        pos = self.position_projection(features)
        return pos, features


class DualPiecesNet(nn.Module):
    def __init__(
        self,
        pretrained=False,
        in_channel=3,
        kernel_size=3,
        in_embd=64,
        d_model=32,
        in_head=8,
        num_block=1,
        dropout=0,
        d_ff=128,
        out_num=12,
        stem_channels=64,
        backbone_layers=(1, 1, 1, 1),
        backbone_stage_planes=(64, 128, 64, 8),
        backbone_base_width=128,
        backbone_scale=2,
        backbone_lstm_input_sizes=(897, 448, 224),
        backbone_lstm_hidden_sizes=(448, 224, 112),
        head_lstm_input_size=112,
        head_lstm_hidden_size=64,
        head_hidden_dim=64,
        cls_out_num=4,
        cls_input_dim=113,
    ):
        super().__init__()

        if not isinstance(pretrained, bool):
            in_channel = pretrained
            pretrained = False

        self.pretrained = pretrained
        self.kernel_size = kernel_size
        self.in_embd = in_embd
        self.d_model = d_model
        self.in_head = in_head
        self.num_block = num_block
        self.d_ff = d_ff

        backbone_layers = _as_tuple(backbone_layers, 4, "backbone_layers")
        backbone_stage_planes = _as_tuple(backbone_stage_planes, 4, "backbone_stage_planes")
        backbone_lstm_input_sizes = _as_tuple(backbone_lstm_input_sizes, 3, "backbone_lstm_input_sizes")
        backbone_lstm_hidden_sizes = _as_tuple(backbone_lstm_hidden_sizes, 3, "backbone_lstm_hidden_sizes")

        self.backbone = Res2Net(
            Bottle2neck,
            backbone_layers,
            input_channels=in_channel,
            stem_channels=stem_channels,
            stage_planes=backbone_stage_planes,
            baseWidth=backbone_base_width,
            scale=backbone_scale,
            lstm_input_sizes=backbone_lstm_input_sizes,
            lstm_hidden_sizes=backbone_lstm_hidden_sizes,
            dropout=dropout,
        )

        self.head = DualPiecesHead(
            sequence_input_size=head_lstm_input_size,
            sequence_hidden_size=head_lstm_hidden_size,
            position_hidden_dim=head_hidden_dim,
            position_classes=out_num,
            class_input_dim=cls_input_dim,
            class_hidden_dim=head_hidden_dim,
            class_classes=cls_out_num,
            dropout=dropout,
        )

        # Compatibility aliases for existing checkpoints or external references.
        self.BiLSTM1 = self.head.sequence_encoder
        self.ap = self.head.pool
        self.projetion_pos_1 = self.head.position_projection
        self.projetion_cls = self.head.class_projection

    def forward(self, x):
        x = self.backbone(x)
        return self.head(x)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DualPiecesNet(pretrained=False, in_channel=3).to(device)
    x = torch.randn(32, 3, 1792).to(device)
    pos, features = model(x)
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Model total parameter: %.2fkb\n' % (model_params / 1024))
    print(pos.shape, features.shape)
