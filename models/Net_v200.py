
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.nn import init
import math
from einops import rearrange
from torch.utils.hooks import RemovableHandle
from timm.layers import DropPath

import torch.nn.functional as F

# from models.embed import DepthwiseSeparableConv1d, DataEmbedding, MLP
# import summary
# from torchsummary import summary



# 改进Bottle2neck内部结构



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
    
# class Bottle2neck(nn.Module): 
#     expansion = 4

#     def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=26, scale = 4, stype='normal'):
#         """ Constructor
#         Args:
#             inplanes: input channel dimensionality
#             planes: output channel dimensionality
#             stride: conv stride. Replaces pooling layer.
#             downsample: None when stride = 1
#             baseWidth: basic width of conv3x3
#             scale: number of scale.
#             type: 'normal': normal set. 'stage': first block of a new stage.
#         """
#         super(Bottle2neck, self).__init__()

#         width = int(math.floor(planes * (baseWidth/64.0)))
#         self.conv1 = nn.Conv1d(inplanes, width*scale, kernel_size=1, bias=False)  
#         self.bn1 = nn.BatchNorm1d(width*scale)
#         # self.se1 = SE_Block(width*scale)
        
        
#         if scale == 1:
#           self.nums = 1
#         else:
#           self.nums = scale -1
#         if stype == 'stage':
#             self.pool = nn.AvgPool1d(kernel_size=5, stride = stride, padding=2)            
#             # self.atten = MixedAttention(width)
#             self.se2 = SE_Block(width)
#             # co_atten 模块
#             self.co_atten = CoordinateAttention (width)
#         convs = []
#         bns = []
#         for i in range(self.nums):
#           convs.append(nn.Conv1d(width, width, kernel_size=5, stride = stride, padding=2, bias=False))
#           bns.append(nn.BatchNorm1d(width))

#         self.convs = nn.ModuleList(convs)
#         self.bns = nn.ModuleList(bns)

#         self.conv3 = nn.Conv1d(width*scale, planes * self.expansion, kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm1d(planes * self.expansion)
#         # self.se3 = SE_Block(planes * self.expansion)

#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#         self.stype = stype
#         self.scale = scale
#         self.width  = width

#     def forward(self, x):
#         residual = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         # out = self.se1(out)  #se模块
     
#         out = self.relu(out)

#         spx = torch.split(out, self.width, 1)
#         for i in range(self.nums):
#           if i==0 or self.stype=='stage':
#             sp = spx[i]
#           else:
#             sp = sp + spx[i]
#           sp = self.convs[i](sp)
          
#           sp = self.relu(self.bns[i](sp))
#           if i==0:
#             out = sp
#           else:
#             out = torch.cat((out, sp), 1)
#         if self.scale != 1 and self.stype=='normal':
#           out = torch.cat((out, spx[self.nums]),1)
#         elif self.scale != 1 and self.stype=='stage':
#           out = torch.cat((out, self.pool(self.co_atten(self.se2(spx[self.nums]) )  )),1)
#         #   out = torch.cat((out, self.atten(spx[self.nums])),1)
         

#         out = self.conv3(out)
#         out = self.bn3(out)
#         # out = self.se3(out)

#         if self.downsample is not None:
#             residual = self.downsample(x)

#         out += residual
#         out = self.relu(out)

#         return out


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


    
# class MyModel(nn.Module):
#     def __init__(self, input_channels, in_features, out_features):
#         super(MyModel, self).__init__()
#         self.conv = nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1)  # 示例卷积层
#         self.lora_layer = LoraLayer(in_features, out_features)

#     def forward(self, x, adapter_name):
#         x = self.conv(x)  # 经过卷积层
#         x = x.view(x.size(0), -1)  # 展平，准备输入到 LoraLayer
#         return self.lora_layer(x, adapter_name)  # 调用 LoRA 层


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
        x, _ = self.lstm(x)
        x = self.layer_norm(x)  # 对 LSTM 输出归一化
        return x


class Res2Net(nn.Module):

    def __init__(self, block, layers, baseWidth = 26, scale = 2):
        self.inplanes = 64
        super(Res2Net, self).__init__()
        self.baseWidth = baseWidth
        self.scale = scale

        self.conv1 = nn.Conv1d(3, 64, kernel_size=7 , stride=2, padding=4,  #self.conv1 = nn.Conv1d(3, 64, kernel_size=7 , stride=2, padding=3,  
                               bias=False)
        self.bn1 = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(3, 32, kernel_size=7 , stride=2, padding=4, bias=False)
        self.bn2 = nn.BatchNorm1d(32)

        self.relu = nn.ReLU(inplace=True)

        # 初始化
        self.fused_conv_bn_relu1 = FusedConvBNReLU(self.conv1, self.bn1)
        self.fused_conv_bn_relu2 = FusedConvBNReLU(self.conv2, self.bn2)


        
        # self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        # self.shrinkage = Shrinkage(out_channels, gap_size=(1), reduction=reduction)
        # self.bn = nn.Sequential(      
        #     nn.BatchNorm1d(256 * block.expansion),
        #     nn.ReLU(inplace=True),
        #     self.shrinkage #特征缩减  
        # )
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.BiLSTM1 = LayerNormLSTM(897,448)

        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.BiLSTM2 = LayerNormLSTM(448,224)
      
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.BiLSTM3 = LayerNormLSTM(224,112)
    
        self.layer4 = self._make_layer(block, 8, layers[3], stride=2)
        self.BiLSTM4 = LayerNormLSTM(112,64)

        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
     



        self.LoraLayer_1 = LoraLayer(897,897)
        self.LoraLayer_1.update_layer('adapter1', r=4  , lora_alpha=0.1, lora_dropout=0.5, init_lora_weights=True)
       

        self.LoraLayer_2 = LoraLayer(897,896)
        self.LoraLayer_2.update_layer('adapter1', r=4, lora_alpha=0.1, lora_dropout=0.5, init_lora_weights=True)
        self.lora_fc_2 = nn.Linear(64,256)

        self.LoraLayer_3 = LoraLayer(897,448)
        self.LoraLayer_3.update_layer('adapter1', r=4, lora_alpha=0.1, lora_dropout=0.5, init_lora_weights=True)
        self.lora_fc_3 = nn.Linear(64,512)

        self.LoraLayer_4 = LoraLayer(897,224)
        self.LoraLayer_4.update_layer('adapter1', r=4, lora_alpha=0.1, lora_dropout=0.5, init_lora_weights=True)
        self.lora_fc_4 = nn.Linear(64,256)

        self.LoraLayer_5 = LoraLayer(897,128)
        self.LoraLayer_5.update_layer('adapter1', r=4, lora_alpha=0.1, lora_dropout=0.5, init_lora_weights=True)
        #self.lora_fc_4 = nn.Linear(64,256)


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
            )   # 检查是否进行下采样操作

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample, 
                        stype='stage', baseWidth = self.baseWidth, scale=self.scale))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, baseWidth = self.baseWidth, scale=self.scale))

        return nn.Sequential(*layers)
    
    def forward(self, x, adapter_name):
        y = x
        
        x = self.fused_conv_bn_relu1(x)  # 融合 Conv + BN + ReLU
        y = self.fused_conv_bn_relu2(y)
 
        lora_outputs = [
            self.LoraLayer_1(x, adapter_name),
            self.LoraLayer_2(x, adapter_name),
            self.LoraLayer_3(x, adapter_name),
            self.LoraLayer_4(x, adapter_name),
            self.LoraLayer_5(y, adapter_name),
            ]         # 计算 LoraLayer 并缓存结果

        # 融合点1
        x = x + lora_outputs[0]
        x = self.layer1(x)
        x = self.BiLSTM1(x)

        # 融合点2
        lora_fc_out_2 = self.lora_fc_2(lora_outputs[1].reshape(-1, 64)).reshape(32, 256, 896)
        x = x + lora_fc_out_2
        x = self.layer2(x)
        x = self.BiLSTM2(x)

        # 融合点3
        lora_fc_out_3 = self.lora_fc_3(lora_outputs[2].reshape(-1, 64)).reshape(32, 512, 448)
        x = x + lora_fc_out_3
        x = self.layer3(x)
        x = self.BiLSTM3(x)

        # 融合点4  
        lora_fc_out_4 = self.lora_fc_4(lora_outputs[3].reshape(-1, 64)).reshape(32, 256, 224)
        x = x + lora_fc_out_4
        x = self.layer4(x)
        x = self.BiLSTM4(x.permute(1, 0, 2))


        # pos = self.fc1(x)
        # cls = self.fc2(x)

        y = lora_outputs[4]

        return x,y


class Net_v200(nn.Module): 
    def __init__(self,  in_channel=3, kernel_size=3, in_embd=64, d_model=32, in_head=8, num_block=1, dropout=0, d_ff=128, out_num=12):
        # def __init__(self, in_channel=3, kernel_size=3, in_embd=128, d_model=512, in_head=8, num_block=1, dropout=0.3, d_ff=128, out_c=1):
        ##################  改动！！！

        '''
        :param in_embd: embedding
        :param d_model: embedding of transformer encoder
        :param in_head: mutil-heat attention
        :param dropout:
        :param d_ff: feedforward of transformer
        :param out_c: class_num
        '''
        super(Net_v200, self).__init__()
        

        self.backbone = Res2Net(Bottle2neck_v200, [1, 1, 1, 1], baseWidth = 128, scale = 2)  # [3,4,23,3]
        # basewidth= 448   1792/448=4
        
    

        # self.enc_embedding_en = DataEmbedding(in_embd, d_model, dropout=dropout)
        # DataEmbedding数据嵌入模块，对输入的数据进行嵌入处理
        

        layers = []
        # for _ in range(num_block):
        #     layers.append(
        #         Encoder_exformer(
        #             layer=Attention(dim=d_model, num_heads=in_head, attn_drop=dropout, proj_drop=0.2),
        #             norm_layer=torch.nn.LayerNorm(d_model),
        #             d_model=d_model,
        #             dropout=dropout,
        #             d_ff=d_ff
        #         )
        #     )
        self.transformer_encoder = nn.Sequential(*layers)


        self.ap = nn.AdaptiveAvgPool1d(output_size=1)

        self.projetion_pos_1 = MLP(input_dim=128, hidden_dim=64, output_dim=32, num_layers=1)
        # self.projetion_pos_2 = MLP(input_dim=128, hidden_dim=64, output_dim=12, num_layers=1)
        self.projetion_cls = MLP(input_dim=113, hidden_dim=64, output_dim=4, num_layers=1)

        self.fc1 = nn.Linear(32 , 12)  # 
        # self.fc2 = nn.Linear(128 , 4)
        self.fc_lora =nn.Linear(64,32)
        


    def forward(self, x ):

        # x = self.backbone1(x)
        x,y = self.backbone(x,'adapter1')  #[32, 128, 113]  源域分支
           
        x = x+y   # 绕过4层DPN网络
        x_1 = self.ap(x.permute(1, 2, 0)).squeeze(-1)  #注意：它和nn.Linear一样，如果你输入了一个三维的数据，他只会对最后一维的数据进行处理
        view = self.projetion_pos_1(x_1)  #孔径位置信息
        pos = self.fc1(view)
	

        return pos,view


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
