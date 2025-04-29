
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.nn import init
import math
# from einops import rearrange
from torch.utils.hooks import RemovableHandle
from timm.layers import DropPath




import torch.nn.functional as F
# from linformer import Linformer
from performer_pytorch import Performer

from linformer_pytorch import Linformer

# from models.embed import DepthwiseSeparableConv1d, DataEmbedding, MLP
# import summary
# from torchsummary import summary



# 改进Bottle2neck内部结构
# 改进整体主干结构，拆分LSTM,并进行多尺度特征融合



class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)   多层感知器FFN"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

#全局平均池化+1*1卷积核+ReLu+1*1卷积核+Sigmoid
class SE_Block(nn.Module):
    def __init__(self, inchannel, ratio=16):
        super(SE_Block, self).__init__()
        # 全局平均池化(Fsq操作)
        self.gap = nn.AdaptiveAvgPool1d(1)
        # 两个全连接层(Fex操作)
        self.fc = nn.Sequential(
            nn.Linear(inchannel, inchannel // ratio, bias=False),  # 从 c -> c/r
            nn.ReLU(),
            nn.Linear(inchannel // ratio, inchannel, bias=False),  # 从 c/r -> c
            nn.Sigmoid()
        )

    def forward(self, x):
            # 读取批数据图片数量及通道数
            b, c, h = x.size()
            # Fsq操作：经池化后输出b*c的矩阵
            y = self.gap(x).view(b, c)
            # Fex操作：经全连接层输出（b，c，1，1）矩阵
            y = self.fc(y).view(b, c, 1)
            
            # Fscale操作：将得到的权重乘以原来的特征图x
            return x * y.expand_as(x)
    
# co-atten
class CoordinateAttention(nn.Module):
    def __init__(self, inchannel, reduction_ratio=16):
        super(CoordinateAttention, self).__init__()
        self.inchannel = inchannel
        self.reduction_ratio = reduction_ratio
        self.reduced_channels = max(inchannel // reduction_ratio, 8)  # 确保至少8个通道

        # 1D 卷积用于捕获坐标信息
        self.conv1d_h = nn.Conv1d(inchannel, self.reduced_channels, kernel_size=1)
        self.conv1d_w = nn.Conv1d(inchannel, self.reduced_channels, kernel_size=1)

        # 全连接层用于生成注意力权重
        self.fc = nn.Sequential(
            nn.Conv1d(self.reduced_channels * 2, inchannel, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h = x.size()
        #print(f"Input shape: {x.shape}")  # 打印输入形状

        # 全局平均池化（高度方向）
        x_h = F.adaptive_avg_pool1d(x, 1)  # (b, c, 1)
        #print(f"x_h shape: {x_h.shape}")  # 打印 x_h 形状

        # 全局平均池化（宽度方向）
        x_w = x.mean(dim=2, keepdim=True)  # (b, c, 1)
        # print(f"x_w shape: {x_w.shape}")  # 打印 x_w 形状

        # 1D 卷积捕获坐标信息
        x_h = self.conv1d_h(x_h)  # (b, c/r, 1)
        x_w = self.conv1d_w(x_w)  # (b, c/r, 1)

        # 拼接高度和宽度方向的特征
        y = torch.cat([x_h, x_w], dim=1)  # (b, 2*c/r, 1)
        # 生成注意力权重
        y = self.fc(y)  # (b, c, 1)

        # 将注意力权重应用到输入特征图上
        return x * y.expand_as(x)
    


# 假设你已经定义了 SE_Block 和 CoordinateAttention
class Bottle2neck_v200(nn.Module): 
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=128, scale=2, stype='normal', drop_prob=0.2, lora_rank=4):
        super(Bottle2neck_v200, self).__init__()

        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.conv1 = nn.Conv1d(inplanes, width * scale, kernel_size=1, bias=False)  
        self.bn1 = nn.BatchNorm1d(width * scale)

        self.nums = 1 if scale == 1 else scale - 1
        self.stype = stype
        self.scale = scale
        self.width = width
        self.relu = nn.ReLU(inplace=True)
        self.drop_path = DropPath(drop_prob) if drop_prob > 0. else nn.Identity()

        if stype == 'stage':
            self.pool = nn.AvgPool1d(kernel_size=5, stride=stride, padding=2)            
            self.se2 = SE_Block(width)
            self.co_atten = CoordinateAttention(width)

        convs = []
        bns = []
        loras = []
        for _ in range(self.nums):
            convs.append(nn.Conv1d(width, width, kernel_size=5, stride=stride, padding=2, bias=False))
            bns.append(nn.BatchNorm1d(width))
            loras.append(self._make_lora(width, rank=lora_rank))

        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.loras = nn.ModuleList(loras)

        self.conv3 = nn.Conv1d(width * scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.dropout = nn.Dropout(p=drop_prob)

        # 自动构造 downsample 分支（如果维度不一致）
        if downsample is None and (inplanes != planes * self.expansion or stride != 1):
            self.downsample = nn.Sequential(
                nn.Conv1d(inplanes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * self.expansion)
            )
        else:
            self.downsample = downsample

    def _make_lora(self, width, rank=4):
        return nn.Sequential(
            nn.Conv1d(width, rank, kernel_size=1, bias=False),
            nn.Conv1d(rank, width, kernel_size=1, bias=False)
        )

    def forward(self, x):
        residual = x

        out = self.relu(self.bn1(self.conv1(x)))
        spx = torch.split(out, self.width, 1)

        for i in range(self.nums):
            sp = spx[i] if (i == 0 or self.stype == 'stage') else sp + spx[i]
            sp = self.convs[i](sp)
            # sp = self.loras[i](sp)
            sp = self.relu(self.bns[i](sp))
            out = sp if i == 0 else torch.cat((out, sp), 1)

        if self.scale != 1 and self.stype == 'normal':
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == 'stage':
            attention_input = self.se2(spx[self.nums])
            attention_input = self.co_atten(attention_input)
            # 动态门控制，减弱注意力的权重
            gate = torch.sigmoid(attention_input.mean(dim=-1, keepdim=True))  # (B, C, 1)
            attention_input = attention_input * (1 - gate) + spx[self.nums] * gate  # 逐通道动态混合
            out = torch.cat((out, self.pool(attention_input)), 1)
             
            
        out = self.dropout(self.bn3(self.conv3(out)))

        #  确保 residual 通道数与 out 一致
        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.drop_path(out) + residual
        out = self.relu(out)
        return out



class LoraLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(LoraLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = {}
        self.lora_alpha = {}
        self.lora_dropout = nn.ModuleDict()
        self.lora_A = nn.ModuleDict()
        self.lora_B = nn.ModuleDict()
        self.scaling = {}

    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights):
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))

        if r > 0:
            self.lora_A.update(nn.ModuleDict({adapter_name: nn.Linear(self.in_features, r, bias=False)}))
            self.lora_B.update(nn.ModuleDict({adapter_name: nn.Linear(r, self.out_features, bias=False)}))
            self.scaling[adapter_name] = lora_alpha / r
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)

    def reset_lora_parameters(self, adapter_name):
        # 这里可以定义权重初始化的方法
         # 获取当前适配器的 A 和 B 层
        lora_A = self.lora_A[adapter_name]
        lora_B = self.lora_B[adapter_name]

        # 初始化 A 层的权重
        init.kaiming_uniform_(lora_A.weight, a=math.sqrt(5))  # Kaiming 初始化
        # 如果有偏置，初始化偏置
        if lora_A.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(lora_A.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(lora_A.bias, -bound, bound)

        # 初始化 B 层的权重
        init.kaiming_uniform_(lora_B.weight, a=math.sqrt(5))  # Kaiming 初始化
        # 如果有偏置，初始化偏置
        if lora_B.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(lora_B.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(lora_B.bias, -bound, bound)
        pass
    def forward(self,x , adapter_name):
        lora_A = self.lora_A[adapter_name]
        lora_B = self.lora_B[adapter_name]
        lora_dropout = self.lora_dropout[adapter_name]

        x = lora_A(x)
        x = lora_dropout(x)
        x = lora_B(x)

        x = x * self.scaling[adapter_name]


        return x



class FusedConvBNReLU(nn.Module):
    def __init__(self, conv, bn):
        super().__init__()
        self.conv = conv
        self.bn = bn

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x)



class LayerNormLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bidirectional=True, dropout=0.5):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, 
                            batch_first=True, bidirectional=bidirectional, dropout=dropout)
        self.layer_norm = nn.LayerNorm(hidden_size * (2 if bidirectional else 1))  # 归一化 LSTM 输出

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = self.layer_norm(x)  # 对 LSTM 输出归一化
        y = x.permute(0, 2, 1)
        return x, y
    
class MultiScaleTransformerFusion(nn.Module):
    def __init__(self, input_dims, d_model=64, nhead=2, num_layers=2, dim_feedforward=512, dropout=0.1, use_performer=False, use_linformer=False):
        super().__init__()

        self.projections = nn.ModuleList([nn.Linear(in_dim, d_model) for in_dim in input_dims])
        self.use_performer = use_performer
        self.use_linformer = use_linformer
        
        # Choose lightweight transformer variant
        if use_performer:
            self.transformer = Performer(
                hidden_size=d_model, 
                num_attention_heads=nhead, 
                num_hidden_layers=num_layers, 
                intermediate_size=dim_feedforward
            )
        elif use_linformer:
            self.transformer = Linformer(
                dim=d_model, 
                num_heads=nhead, 
                num_layers=num_layers,
                dropout=dropout
            )
        else:
            encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fusion_fc = nn.Linear(d_model, d_model)

    def forward(self, features):
        """
        features: list[Tensor], 每个 Tensor 形状为 [B, T_i, C_i]
        """
        batch_size = features[0].shape[0]
        proj_feats, lengths = [], []

        for x, proj in zip(features, self.projections):
            B, T, C = x.shape
            x_proj = proj(x)  # [B, T, d_model]
            proj_feats.append(x_proj)
            lengths.append(T)

        max_len = max(lengths)
        padded_feats, masks = [], []

        for x, l in zip(proj_feats, lengths):
            pad_len = max_len - l
            if pad_len > 0:
                x = F.pad(x, (0, 0, 0, pad_len))  # pad 时间维
            padded_feats.append(x)

            mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=x.device)
            mask[:, l:] = True  # True 为 masked
            masks.append(mask)

        stacked_feats = torch.stack(padded_feats, dim=1)  # [B, N, T, d_model]
        stacked_feats = stacked_feats.view(batch_size * len(features), max_len, -1)
        all_masks = torch.cat(masks, dim=0)  # [B*N, T]

        # Use Cross-Attention via Transformer Decoder
        transformer_out = self.transformer(stacked_feats, src_key_padding_mask=all_masks)
        
        # Apply attention pooling or mean pooling
        if self.use_performer or self.use_linformer:
            # Perform pooling by averaging or weighted attention mechanism
            pooled_output = transformer_out.mean(dim=1)  # Mean pooling
        else:
            pooled_output = transformer_out[:, 0, :]  # CLS token as output

        pooled_output = pooled_output.view(batch_size, len(features), -1)
        fused = pooled_output.mean(dim=1)  # [B, d_model] (mean pooling across branches)
        fused = self.fusion_fc(fused)        # [B, d_model]

        return fused

class Res2Net(nn.Module):

    def __init__(self, block, layers, baseWidth=26, scale=2):
        self.inplanes = 64
        super(Res2Net, self).__init__()
        self.baseWidth = baseWidth
        self.scale = scale

        self.conv1 = nn.Conv1d(3, 64, kernel_size=7, stride=2, padding=4, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.fused_conv_bn_relu1 = FusedConvBNReLU(self.conv1, self.bn1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 8, layers[3], stride=2)

        # 拆分的 LSTM 层，全部输出为64
        self.BiLSTM1 = LayerNormLSTM(256, 128)
        self.BiLSTM2 = LayerNormLSTM(512, 256)
        self.BiLSTM3 = LayerNormLSTM(256, 128)
        self.BiLSTM4 = LayerNormLSTM(32, 56)

        self.fusion_module = MultiScaleTransformerFusion(
            input_dims=[256, 512, 256, 112],  # 你4个分支的C
            d_model=64,
            nhead=2,
            num_layers=1
        )

      



        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
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
            layers.append(block(self.inplanes, planes, baseWidth=self.baseWidth, scale=self.scale))

        return nn.Sequential(*layers)

    def forward(self, x):
        # [32, 3, 1792]

        x = self.fused_conv_bn_relu1(x) 
    
        x = self.layer1(x) # [32, 256, 897]
        x1, x = self.BiLSTM1(x)  # x: [32, 256, 897] x1:[32, 897, 256]
        
        x = self.layer2(x)  # [32, 512, 449]
        x2, x = self.BiLSTM2(x)  # x: [32, 512, 449] x2:[32, 449, 512]
       

        x = self.layer3(x)
        x3, x = self.BiLSTM3(x)  # x: [32, 256, 225] x3:[32, 225, 256]
 
       
        x = self.layer4(x)
        x4, x = self.BiLSTM4(x)  # x: [32,32,113] x4: [32, 113, 112]
   


        fused = self.fusion_module([x1, x2, x3, x4])  # [B, d_model]
        
        #print(x.shape)   # x: [32, 112, 113])
        


        # # Attention 融合
        # fused = self.lstm_fusion([x1, x2, x3, x4])  # [T, B, 64]

        return x, fused


class Net_v2012(nn.Module):
    def __init__(self, in_channel=3, kernel_size=3, in_embd=64, d_model=32, in_head=8, num_block=1, dropout=0, d_ff=128, out_num=12):
        super(Net_v2012, self).__init__()
        self.backbone = Res2Net(Bottle2neck_v200, [1, 1, 1, 1], baseWidth=128, scale=2)
        self.ap = nn.AdaptiveAvgPool1d(output_size=1)
        self.projetion_pos_1 = MLP(input_dim=176, hidden_dim=32, output_dim=32, num_layers=1)
        self.fc1 = nn.Linear(32, 12)

    def forward(self, x):
        x, fused = self.backbone(x)  # 输出为 BCT[32,32,113]
        # print(x.shape)  [32,256]
        # print(fused.shape)
        x = self.ap(x).squeeze(-1)  # [32, 32]
        # print(x.shape)
        x = torch.cat([x, fused], dim=1)  # [32, 176]
        
        view = self.projetion_pos_1(x)
        pos = self.fc1(view)
        return pos, view



if __name__ == '__main__':

    parameter1 = 32
    x = torch.randn(parameter1, 2, 1792).to(0)
    

    model = Res2Net(Bottle2neck_v200, [1, 1, 1, 1], baseWidth =32, scale = 2).to(0)  # [3,4,23,3]
    print(model)
    # model = model(in_channel=3, kernel_size=3, in_embd=64, d_model=112, in_head=2, num_block=1, d_ff=64, dropout=0.2, out_c=4).to(0)
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Model total parameter: %.2fMb\n' % (model_params/1024/2024))
    # print(model)

    # x = model(x)
    # print(x[0].shape)
    # print(x[1].shape)



    # summary(model,input_size=(64,64))
   

    # print(model)
    
    # summary(model, input_size=(2, 1792))
    # summary(model, input_size=(x))
