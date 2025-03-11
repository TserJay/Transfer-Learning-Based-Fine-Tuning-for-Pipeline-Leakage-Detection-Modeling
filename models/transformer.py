import torch
from torch import nn
import math
import torch.nn.functional as F

# from embed import MLP, DataEmbedding
class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=64, out_channels=d_model,
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


class transformer(nn.Module):
    def __init__(self, in_channel=3, kernel_size=3, in_embd=128, d_model=512, in_head=8, num_block=1, dropout=0.3, d_ff=128, out_c=1):
        super(transformer, self).__init__()

        self.residual_function = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=7, stride=4, padding=7//2, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),

            nn.Conv1d(32, 64, kernel_size=7, stride=4, padding=7 // 2, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Conv1d(64, 64, kernel_size=7, stride=1, padding=7 // 2, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        self.enc_embedding_en = DataEmbedding(64, d_model, dropout=0.1)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=in_head, dim_feedforward=d_ff)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.ap = nn.AdaptiveAvgPool1d(output_size=1)

        self.BiLSTM = nn.LSTM(input_size=512,
                                    hidden_size=256,
                                    num_layers=1,
                                    batch_first=True,
                                    bidirectional=True,
                                    dropout=0)

        self.projetion_huigui = MLP(input_dim=d_model, hidden_dim=64, output_dim=12, num_layers=1)
        # self.projetion_leibie = MLP(input_dim=d_model, hidden_dim=64, output_dim=out_c, num_layers=1)

    def forward(self, x):
        # x = self.residual_function(x[:, 0, ].unsqueeze(1) + x[:, 1, ].unsqueeze(1) + x[:, 2, ].unsqueeze(1))
        x = self.residual_function(x)

        x = self.enc_embedding_en(x.transpose(2, 1), None)
        x = self.transformer_encoder(x)
        x , _  = self.BiLSTM(x)
        #print(x.shape)

        x = self.ap(x.transpose(2,1)).squeeze(-1)
        #x, _  = self.BiLSTM(x)
        #print(x.shape)

        out_huigui = self.projetion_huigui(x)
        #out_fenlei = self.projetion_leibie(x)

        return out_huigui, x
