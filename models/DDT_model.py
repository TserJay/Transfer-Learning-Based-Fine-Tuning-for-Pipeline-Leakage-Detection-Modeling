import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.nn import init

# from embed import DepthwiseSeparableConv1d, DataEmbedding, MLP
# from embed import DepthwiseSeparableConv1d, DataEmbedding, MLP
from torchsummary import summary
import math


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                    kernel_size=3, padding=padding, padding_mode='circular')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1,2)
        return x
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super(DataEmbedding, self).__init__()
        '''
        * * * *
        * * * *
        * * * *
        * * * *
        不可学习位置编码，没一个batch都是以上的tesnor，连续的几个样本
        '''
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)


    def forward(self, x, x_mark):
        # x_mark是时间搞的特征，目前用不着，用着了再重写
        x = self.value_embedding(x) + self.position_embedding(x)

        return self.dropout(x)

class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, bias=False, activate=True):
        super(DepthwiseSeparableConv1d, self).__init__()
        self.DepthwiseSeparableConv1d = nn.Sequential(
            nn.Conv1d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=1, bias=bias),
            nn.BatchNorm1d(out_channels),
            nn.GELU() if activate else nn.Identity(),
            nn.Conv1d(in_channels=out_channels,
                      out_channels=out_channels,
                      kernel_size=kernel_size,
                      stride=1,
                      padding=kernel_size // 2,
                      groups=out_channels, bias=bias),
            nn.BatchNorm1d(out_channels),
            nn.GELU() if activate else nn.Identity(),

            nn.MaxPool1d(kernel_size=stride),
        )

    def forward(self, x):
        return self.DepthwiseSeparableConv1d(x)

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


class BasicBlock(nn.Module):
    expansion = 1
    # 类属性，基本块的扩展倍数，1为输入和输出通道相同
    def __init__(self, in_channels, out_channels, stride=1, reduction=8):
        # reduction为缩减维度的参数

        super().__init__()
        self.shrinkage = Shrinkage(out_channels, gap_size=(1), reduction=reduction)
        # 特征维度缩减；gap_size全局平均池化窗口大小为1

        # residual function
        self.residual_function = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=3//2, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=3//2, bias=False),
            nn.BatchNorm1d(out_channels * BasicBlock.expansion),
            self.shrinkage #特征缩减
        )

        # shortcut
        self.shortcut = nn.Sequential()

        # the shortcut output dimension is not the same with residual function
        # use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels * BasicBlock.expansion)
            )
            # 如果输入通道和输出通道不同时需要进行维度匹配

    def forward(self, x):
         return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))
    # inplace=True 在原输入张量上进行操作，会覆盖原始的输入张量，有利于节省内存

class Shrinkage(nn.Module):
    def __init__(self, channel, gap_size, reduction):
        super(Shrinkage, self).__init__()

        self.maxpool = nn.AdaptiveMaxPool1d(gap_size)
        self.avgpool = nn.AdaptiveAvgPool1d(gap_size)
        # 

        self.se = nn.Sequential(
            nn.Conv1d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            # nn.ELU(),
            nn.Conv1d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_raw = x
        x = torch.abs(x) #全部取绝对值变为正数
        x_abs = x

        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)

        mean_max_result = torch.flatten(max_result, 1)
        mean_avg_result = torch.flatten(avg_result, 1)

        average_max_result = torch.mean(mean_max_result, dim=1, keepdim=True)
        average_avg_result = torch.mean(mean_avg_result, dim=1, keepdim=True)

        average = (average_max_result + average_avg_result)/2.

        # x = torch.flatten(x, 1) #batch_size, channel
        # average = torch.mean(x, dim=1, keepdim=True)  
        # CS batch_size, 1,每个样本n个特征，在n个特征上求一个均值。其实是n个通道值就均值。
        ## average = x    #CW
        # x = self.fc(x) #对每个样本的通道求sigmoid注意力机制

        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        x = self.sigmoid(max_out + avg_out)

        x = torch.mul(average, x.squeeze(-1)) #激活之后每个值都被嵌入为0和1，0和1并不能表征信号的幅值信息，乘以这个信号的平均值恢复回来。
        x = x.unsqueeze(2) #2, 5, 1
        # soft thresholding 比阈值x大的都保留下来，比阈值x小的全部归为0实现去噪。
        sub = x_abs - x # 在序列长度方向上减去其对应的通道噪声值，这是去噪之后的值
        zeros = sub - sub
        n_sub = torch.max(sub, zeros)
        x = torch.mul(torch.sign(x_raw), n_sub) #符号函数，把n_sub当中原本为负值的重新归置为负值。
        return x

class Encoder_exformer(nn.Module):
    def __init__(self, layer, norm_layer, d_model, dropout, d_ff):
        super(Encoder_exformer, self).__init__()

        self.layer = layer
        self.norm_layer = norm_layer

        self.conv1 = BasicBlock(d_model, d_ff, 1, 4)
        self.conv2 = BasicBlock(d_ff, d_model, 1, 4)
        # d_ff是前馈网络中的维度

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # 两个归一化层
        self.dropout = nn.Dropout(dropout)
        # dropou是丢弃的概率
        self.activation = F.gelu
        # 激活函数gelu

    def forward(self, x):
        exattn = self.layer(x)
        x = x + self.dropout(exattn)
        y = x = self.norm1(x)
        #这个位置为Feed forward。

        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        
        return self.norm2(x + y)

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        # dim输入特征维数；num_heads默认注意力头数；akv_bias表示在查询、键、值的线性变换中使用偏置项
        # qk_scale缩放因子，用于缩放查询-键的点积，默认为 None; attn_drop注意力权重丢弃概率；
        # proj_drop控制投影结果丢弃的概率。
        super().__init__()
        self.num_heads = num_heads
        assert dim % num_heads == 0

        self.coef = 4

        self.trans_dims = nn.Linear(dim, dim * self.coef) # 256, 4*256
        self.num_heads = self.num_heads * self.coef # 8*4=32 
        
        self.k = 64 // self.coef # 64   =16？

        self.linear_0 = nn.Linear(dim * self.coef // self.num_heads, self.k) # 256*4//32, 64=(32, 64)
        self.linear_1 = nn.Linear(self.k, dim * self.coef // self.num_heads) # (64, 32)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim * self.coef, dim) # (256*4, 256)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape

        x = self.trans_dims(x)  # B, N, C
        # 线性层变换，转为B,N,C的张量
        x = x.view(B, N, self.num_heads, -1).permute(0, 2, 1, 3)

        attn = self.linear_0(x)
        # 线性层变换 32->64
        attn = attn.softmax(dim=-2) 
        # 在序列长度方向求softmax，求在不同的token之间的权重占比。
        # 
        attn = attn / (1e-9 + attn.sum(dim=-1, keepdim=True)) 
        # 对于每一个token来说，它的每个通道的权重值都是对比了其他token的，那么对此token的整个通道值求和然后重新分配
        

        attn = self.attn_drop(attn)

        x = self.linear_1(attn).permute(0, 2, 1, 3).reshape(B, N, -1)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class DDT_models(nn.Module):
    def __init__(self, in_channel=3, kernel_size=3, in_embd=64, d_model=32, in_head=8, num_block=1, dropout=0, d_ff=128, out_c=1):
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
        super(DDT_models, self).__init__()
        
        # self.sos = nn.Embedding(3, 1792)
        # self.flag = torch.LongTensor([0, 1, 2]).to(0)

        # self.backbone = nn.Sequential(
        #     DepthwiseSeparableConv1d(in_channels=in_channel, out_channels=32, kernel_size=kernel_size, stride=1, activate=True, bias=False),
        #     DepthwiseSeparableConv1d(in_channels=32, out_channels=64, kernel_size=kernel_size, stride=2, activate=True, bias=False),
        #     DepthwiseSeparableConv1d(in_channels=64, out_channels=in_embd, kernel_size=kernel_size, stride=2, activate=True, bias=False),
        # )

        self.backbone1 = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=kernel_size, stride=4, padding=kernel_size // 2, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),

            nn.Conv1d(32, 64, kernel_size=kernel_size, stride=4, padding=kernel_size // 2, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Conv1d(64, 64, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),


            # nn.Conv1d(64, 64, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=False),
            # nn.BatchNorm1d(64),
            # nn.ReLU(inplace=True),
            # nn.Conv1d(64, 64, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=False),
            # nn.BatchNorm1d(64),
            # nn.ReLU(inplace=True),
            # nn.Conv1d(64, 64, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=False),
            # nn.BatchNorm1d(64),
            # nn.ReLU(inplace=True),
        )

        # self.backbone2 = nn.Sequential(
        #     nn.Conv1d(2, 32, kernel_size=kernel_size, stride=4, padding=kernel_size // 2, bias=False),
        #     nn.BatchNorm1d(32),
        #     nn.ReLU(inplace=True),
        #
        #     nn.Conv1d(32, 64, kernel_size=kernel_size, stride=4, padding=kernel_size // 2, bias=False),
        #     nn.BatchNorm1d(64),
        #     nn.ReLU(inplace=True),
        #
        #     nn.Conv1d(64, 64, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=False),
        #     nn.BatchNorm1d(64),
        #     nn.ReLU(inplace=True),
        # )

        self.enc_embedding_en = DataEmbedding(in_embd, d_model, dropout=dropout)
        # DataEmbedding数据嵌入模块，对输入的数据进行嵌入处理
        

        layers = []
        for _ in range(num_block):
            layers.append(
                Encoder_exformer(
                    layer=Attention(dim=d_model, num_heads=in_head, attn_drop=dropout, proj_drop=0),
                    norm_layer=torch.nn.LayerNorm(d_model),
                    d_model=d_model,
                    dropout=dropout,
                    d_ff=d_ff
                )
            )
        self.transformer_encoder = nn.Sequential(*layers)

        self.BiLSTM1 = nn.LSTM(input_size=d_model,
                               hidden_size=d_model//2,
                               num_layers=1,
                               batch_first=False,
                               bidirectional=True,
                               dropout=0)

        self.ap = nn.AdaptiveAvgPool1d(output_size=1)

        self.projetion_huigui = MLP(input_dim=d_model, hidden_dim=64, output_dim=12, num_layers=1)
        self.projetion_leibie = MLP(input_dim=d_model, hidden_dim=64, output_dim=out_c, num_layers=1)

        self.fc1 = nn.Linear(32 , 12)
        self.fc2 = nn.Linear(32 , 4)

    def forward(self, x):
        x = self.backbone1(x)
        # x = self.backbone1(x[:, 0, ].unsqueeze(1) + x[:, 1, ].unsqueeze(1) + x[:, 2, ].unsqueeze(1))
        print(x.shape)
        

        x = x.permute(0, 2, 1)
        # x = x + self.sos(self.flag)[None, :, :]
        # x = self.backbone(x).permute(0, 2, 1)
        x = self.enc_embedding_en(x, None)
        print(x.shape)
       
        x = self.transformer_encoder(x)
        print(x.shape)
  

        x, _ = self.BiLSTM1(x.permute(1, 0, 2))
        print(x.shape)
  
        x_1 = self.ap(x.permute(1, 2, 0)).squeeze(-1)

        print(x_1.shape)


        pos = self.fc1(x_1)  #孔径位置信息
        cls = self.fc2(x_1)  #孔径大小信息

 

        # x = self.ap(x.permute(0, 2, 1)).squeeze(-1)

        # out_x = self.projetion_huigui(x_1)
        # out_fenlei = self.projetion_leibie(x_1)



        # return out_huigui, out_fenlei, x.permute(1, 2, 0)[:, :, ::16]
        return pos,cls    


if __name__ == '__main__':

    parameter1 = 32
    x = torch.randn(parameter1, 3, 1792).to(0)

    model = DDT_models(in_channel=3, kernel_size=3, in_embd=64, d_model=64, in_head=2, num_block=1, d_ff=64, dropout=0.2, out_c=4).to(0)
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Model total parameter: %.2fkb\n' % (model_params/1024))

    x = model(x)
    print(x[0].shape)
    print(x[1].shape)
    # print(x[2].shape)
    summary(model, input_size=(3, 1792))



