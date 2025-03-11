import torch
from torch import nn 
from models.oneD_Meta_ACON import MetaAconC

import warnings




class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        # 初始化，输入和输出通道数、计算中间通道数的参数
        super(CoordAtt, self).__init__()
        # self.pool_w = nn.AdaptiveAvgPool1d(1)
        self.pool_w = nn.AdaptiveMaxPool1d(1)
        # 最大自适应池化层
        mip = max(6, inp // reduction)
        self.conv1 = nn.Conv1d(inp, mip, kernel_size=1, stride=1, padding=0)
        # 输入通道数，输出通道数，卷积核大小，步长，填充

        self.bn1 = nn.BatchNorm1d(mip, track_running_stats=False)
        # 输入的通道数mip即为上卷积层的输出mip，更新策略选择，使用当前batch来归一化

        self.act = MetaAconC(mip)
        # 激活函数（导入激活函数文件）

        self.conv_w = nn.Conv1d(mip, oup, kernel_size=1, stride=1, padding=0)
        

    def forward(self, x):
        identity = x
        # 
        n, c, w = x.size()
        x_w = self.pool_w(x)
        # 池化
        y = torch.cat([identity, x_w], dim=2)
        # 拼接，沿着w（dim = 2）
        y = self.conv1(y)
        # 卷积
        y = self.bn1(y)
        # 归一化
        y = self.act(y)
        # 激活函数
        x_ww, x_c = torch.split(y, [w, 1], dim=2)
        # 分割
        a_w = self.conv_w(x_ww)
        a_w = a_w.sigmoid()
        out = identity * a_w
        return out


class Net(nn.Module):
    def __init__(self , pretrained=False):
        super(Net, self).__init__()
        if pretrained == True:
            warnings.warn("Pretrained model is not available")

        self.p1_1 = nn.Sequential(nn.Conv1d(3, 50, kernel_size=2, stride=2, padding=1),
                                  nn.BatchNorm1d(50),
                                  MetaAconC(50))
        self.p1_2 = nn.Sequential(nn.Conv1d(50, 30, kernel_size=2, stride=2),
                                  nn.BatchNorm1d(30),
                                  MetaAconC(30))
        
        
        self.p1_3 = nn.MaxPool1d(2, 1)


        self.p2_1 = nn.Sequential(nn.Conv1d(3, 50, kernel_size=2, stride=2,padding=1),
                                  nn.BatchNorm1d(50),
                                  MetaAconC(50))
        self.p2_2 = nn.Sequential(nn.Conv1d(50, 40, kernel_size=2, stride=1),
                                  nn.BatchNorm1d(40),
                                  MetaAconC(40))
        self.p2_3 = nn.MaxPool1d(2, 1)

        self.p2_4 = nn.Sequential(nn.Conv1d(40, 30, kernel_size=3, stride=1),
                                  nn.BatchNorm1d(30),
                                  MetaAconC(30))
        self.p2_5 = nn.Sequential(nn.Conv1d(30, 30, kernel_size=3, stride=2),
                                  nn.BatchNorm1d(30),
                                  MetaAconC(30))
        self.p2_6 = nn.MaxPool1d(2, 1)


        self.p3_0 = CoordAtt(30, 30)
        # Attention
        self.p3_1 = nn.Sequential(nn.GRU(124, 64, bidirectional=True)) #BIGUR
        # self.p3_2 = nn.Sequential(nn.LSTM(128, 512))
        self.p3_3 = nn.Sequential(nn.AdaptiveAvgPool1d(1))
        # GAP
        # self.p4 = nn.Sequential(nn.Linear(30, 10))
        self.p4 = nn.Sequential(nn.Linear(30, 12))
        # FC

    def forward(self, x):
        p1 = self.p1_2(self.p1_1(x))
        

        print(p1.shape)
        p1 = self.p1_3(p1)





        p2 = self.p2_6(self.p2_5(self.p2_4(self.p2_3(self.p2_2(self.p2_1(x))))))

        encode = torch.mul(p1, p2)

        # p3 = self.p3_2(self.p3_1(encode))
        p3_0 = self.p3_0(encode).permute(1, 0, 2)
        # Attention
        p3_2, _ = self.p3_1(p3_0)
        
        # p3_2, _ = self.p3_2(p3_1)
        p3_11 = p3_2.permute(1, 0, 2)  
        p3_12 = self.p3_3(p3_11).squeeze()
        # p3_11 = h1.permute(1,0,2)
        # p3 = self.p3(encode)
        # p3 = p3.squeeze()
        # p4 = self.p4(p3_11)  # LSTM(seq_len, batch, input_size)
        # p4 = self.p4(encode)
        p4 = self.p4(p3_12)

        return p4
