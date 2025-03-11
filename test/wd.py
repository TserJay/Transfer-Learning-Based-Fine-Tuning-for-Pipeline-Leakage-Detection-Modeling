import torch
import torch.nn as nn
import pywt


#小波变换
# class WaveletTransform(nn.Module):
#     def __init__(self, wavelet_name='db1', level=3):
#         super(WaveletTransform, self).__init__()
#         self.wavelet = pywt.Wavelet(wavelet_name)
#         self.level = level

#     def forward(self, x):
#         # Apply discrete wavelet transform (DWT) on input
#         coeffs = pywt.wavedec(x.cpu().numpy(), self.wavelet, level=self.level)
#         coeffs_torch = [torch.tensor(c).to(x.device) for c in coeffs]
#         return coeffs_torch

# class InverseWaveletTransform(nn.Module):
#     def __init__(self, wavelet_name='db1'):
#         super(InverseWaveletTransform, self).__init__()
#         self.wavelet = pywt.Wavelet(wavelet_name)

#     def forward(self, coeffs):
#         # Apply inverse discrete wavelet transform (IDWT) to reconstruct signal
#         coeffs_np = [c.cpu().numpy() for c in coeffs]
#         reconstructed = pywt.waverec(coeffs_np, self.wavelet)
#         return torch.tensor(reconstructed).to(coeffs[0].device)



    
# 小波包分解与重构
# class WaveletPacketTransform(nn.Module):
#     def __init__(self, wavelet_name='db1', level=3):
#         super(WaveletPacketTransform, self).__init__()
#         self.wavelet = wavelet_name
#         self.level = level

#     def forward(self, x):
#         # Wavelet Packet Decomposition (WPD)
#         wp = pywt.WaveletPacket(data=x.cpu().numpy(), wavelet=self.wavelet, mode='symmetric', maxlevel=self.level)
        
#         nodes = wp.get_level(self.level, order='freq')

#         # Error check for the number of nodes
#         if not nodes:
#             raise ValueError(f"No nodes found at level {self.level}. Please check the level and input signal.")

#         # Extract coefficients from each node
#         coeffs = [torch.tensor(node.data).to(x.device) for node in nodes]
#         return coeffs

import torch
import torch.nn as nn
import pywt

class WaveletPacketTransform(nn.Module):
    def __init__(self, wavelet_name='db1', level=3):
        super(WaveletPacketTransform, self).__init__()
        self.wavelet = wavelet_name
        self.level = level

    def forward(self, x):
        channels, length = x.shape  # 假设输入为 (channels, length)
        coeffs = []
        for c in range(channels):
            # 对每个通道的信号进行小波包分解
            wp = pywt.WaveletPacket(data=x[c].cpu().numpy(), wavelet=self.wavelet, mode='symmetric', maxlevel=self.level)
            nodes = wp.get_level(self.level, order='freq')
            coeffs.append([torch.tensor(node.data).to(x.device) for node in nodes])
        return coeffs


class InverseWaveletPacketTransform(nn.Module):
    def __init__(self, wavelet_name='db1', level=3):
        super(InverseWaveletPacketTransform, self).__init__()
        self.wavelet = wavelet_name
        self.level = level

    def forward(self, coeffs):
        channels = len(coeffs)
        reconstructed = []
        for c in range(channels):
            dummy_data = [0] * (2 ** self.level * 7)
            wp = pywt.WaveletPacket(data=dummy_data, wavelet=self.wavelet, mode='symmetric', maxlevel=self.level)
            nodes = wp.get_level(self.level, order='freq')

            if len(nodes) != len(coeffs[c]):
                raise ValueError(f"系数数量 ({len(coeffs[c])}) 与节点数量 ({len(nodes)}) 不匹配，级别为 {self.level}")

            for i, node in enumerate(nodes):
                node.data = coeffs[c][i].detach().cpu().numpy().flatten()
            reconstructed_signal = wp.reconstruct(update=True)
            reconstructed.append(torch.tensor(reconstructed_signal).to(coeffs[0][0].device))
        return torch.stack(reconstructed, dim=0)


# class AdaptiveGatedUnit(nn.Module):
#     def __init__(self, signal_length, level):
#         super(AdaptiveGatedUnit, self).__init__()
#         input_dim = signal_length // (2 ** level)
#         self.gate = nn.Sequential(
#             nn.Linear(input_dim, input_dim),
#             #nn.Conv1d(1, 1, kernel_size=3, padding=1),
#             nn.Sigmoid()
#         )
#         self.threshold = nn.Parameter(torch.tensor(0.1), requires_grad=True)  # 学习阈值

#     def forward(self, coeffs):
#         gated_coeffs = []
#         for channel_coeffs in coeffs:
#             channel_gated = []
#             for c in channel_coeffs:
#                 gate_weight = self.gate(c)
#                 gated_coeff = c * gate_weight

#                 # 应用自适应阈值
#                 gated_coeff[torch.abs(gated_coeff) < self.threshold] = 0
#                 channel_gated.append(gated_coeff)
#             gated_coeffs.append(channel_gated)
#         return gated_coeffs
    
class AdaptiveGatedUnit(nn.Module):
    def __init__(self, level, in_channels=1):
        super(AdaptiveGatedUnit, self).__init__()
        self.conv_gate = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=3 // 2),
            nn.Sigmoid()
        )
        self.threshold = nn.Parameter(torch.tensor(0.1), requires_grad=True)  # 学习阈值

    def forward(self, coeffs):
        gated_coeffs = []
        for channel_coeffs in coeffs:
            channel_gated = []
            for c in channel_coeffs:
                c = c.unsqueeze(0).unsqueeze(0)  # 调整形状为 (batch=1, channels=1, length)
                gate_weight = self.conv_gate(c).squeeze(0)  # 应用卷积门控
                gated_coeff = c.squeeze(0) * gate_weight  # 应用门控

                # 自适应阈值更新
                threshold_value = torch.mean(gated_coeff.abs()) * self.threshold
                gated_coeff[torch.abs(gated_coeff) < threshold_value] = 0  # 应用自适应阈值
                channel_gated.append(gated_coeff.squeeze(0))
            gated_coeffs.append(channel_gated)
        return gated_coeffs


class WaveletGatedNet(nn.Module):
    def __init__(self, signal_length, wavelet_name='db1', level=3):
        super(WaveletGatedNet, self).__init__()
        self.wavelet_transform = WaveletPacketTransform(wavelet_name, level)
        self.gated_unit = AdaptiveGatedUnit(signal_length=signal_length, level=level)
        self.inverse_wavelet_transform = InverseWaveletPacketTransform(wavelet_name, level)

    def forward(self, x):
        # Step 1: Wavelet Packet Decomposition
        coeffs = self.wavelet_transform(x)

        # Step 2: Apply Gated Unit
        gated_coeffs = self.gated_unit(coeffs)

        # Step 3: Inverse Wavelet Packet Transform
        reconstructed_signal = self.inverse_wavelet_transform(gated_coeffs)

        return reconstructed_signal





# Example usage
if __name__ == "__main__":
    # Generate a random 1D signal
    import torch
    import torch.nn as nn
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    file_path = r'E:\projects\UDTL-LoRA\data\leak_signals\0\7\(7).csv'  # 替换为你的文件路径
    data = pd.read_csv(file_path, usecols=[1, 3, 5])  # 读取第 1, 3, 5 列
    data = data.iloc[:1792]

    # 提取列数据
    X = np.linspace(0, 1, 1792)[:, None]  # 输入信号 (1792 个点)
    y1 = data.iloc[:, 0].values  # 第 1 列作为第一个目标变量（传感器 1）
    y2 = data.iloc[:, 1].values  # 第 2 列作为第二个目标变量（传感器 2）
    y3 = data.iloc[:, 2].values  # 第 3 列作为第三个目标变量（传感器 3）

    #signal = torch.randn([32, 3, 1792])
    # 组合信号为一个三通道的输入
    signal = torch.tensor(np.stack([y1, y2, y3]), dtype=torch.float32)  # 形状为 (3, 1792)

    # 定义模型参数
    signal_length = 1792
    level = 8
    model = WaveletGatedNet(signal_length=signal_length, wavelet_name='db1', level=level)

    # 前向传播
    output_signal = model(signal)  # 模型处理信号
    output_signal = output_signal.detach().numpy()  # 转换为 NumPy 数组以便绘图

    # 绘图
    plt.figure(figsize=(15, 10))
    titles = ['Sensor 1', 'Sensor 2', 'Sensor 3']

    for i in range(3):
        plt.subplot(3, 1, i + 1)
        plt.plot(signal[i].numpy(), label='Original Signal', color='blue', alpha=0.6)
        plt.plot(output_signal[i], label='Processed Signal', color='red', linestyle='--')
        plt.title(f'Wavelet Gated Network Processing - {titles[i]}')
        plt.xlabel('Data Point Index')
        plt.ylabel('Signal Value')
        plt.legend()

    plt.tight_layout()
    plt.show()
 
   
    # Forward pass
    #output_signal = model(signal)
    print("Input Signal:", signal.shape)
    print("Output Signal:", output_signal.shape)

    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Model total parameter: %.4fMB\n' % (model_params/1024/1024))

   