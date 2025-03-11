import torch
import torch.nn.functional as F

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


class AdaptiveGatedUnit(nn.Module):
    def __init__(self, signal_length, level):
        super(AdaptiveGatedUnit, self).__init__()
        input_dim = signal_length // (2 ** level)
        self.gate = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid()
        )
        self.threshold = nn.Parameter(torch.tensor(0.1), requires_grad=True)  # 学习阈值

    def forward(self, coeffs):
        gated_coeffs = []
        for channel_coeffs in coeffs:
            channel_gated = []
            for c in channel_coeffs:
                gate_weight = self.gate(c)
                gated_coeff = c * gate_weight

                # 应用自适应阈值
                gated_coeff[torch.abs(gated_coeff) < self.threshold] = 0
                channel_gated.append(gated_coeff)
            gated_coeffs.append(channel_gated)
        return gated_coeffs



def moving_average(x, window_size=5):
    """Apply moving average to a 1D tensor."""
    weights = torch.ones(window_size) / window_size
    weights = weights.to(x.device)  # 将权重移动到与输入相同的设备
    smoothed = F.conv1d(x.unsqueeze(1), weights.view(1, 1, -1), padding=window_size // 2)
    return smoothed.squeeze(1)



# 修改 WaveletGatedNet 类，将移动平均应用到输出
class WaveletGatedNet(nn.Module):
    def __init__(self, signal_length, wavelet_name='db1', level=3, ma_window_size=5):
        super(WaveletGatedNet, self).__init__()
        self.wavelet_transform = WaveletPacketTransform(wavelet_name, level)
        self.gated_unit = AdaptiveGatedUnit(signal_length=signal_length, level=level)
        self.inverse_wavelet_transform = InverseWaveletPacketTransform(wavelet_name, level)
        self.ma_window_size = ma_window_size  # 移动平均窗口大小

    def forward(self, x):
        # Step 1: Wavelet Packet Decomposition
        coeffs = self.wavelet_transform(x)

        # Step 2: Apply Gated Unit
        gated_coeffs = self.gated_unit(coeffs)

        # Step 3: Inverse Wavelet Packet Transform
        reconstructed_signal = self.inverse_wavelet_transform(gated_coeffs)

        # Step 4: Apply Moving Average
        smoothed_signal = torch.stack([moving_average(reconstructed_signal[i].unsqueeze(0), self.ma_window_size)
                                       for i in range(reconstructed_signal.shape[0])])

        return smoothed_signal


# 示例代码
if __name__ == "__main__":
    # 读取 CSV 文件
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    file_path = r'E:\projects\UDTL-LoRA\data\leak_signals\0\7\(7).csv'  # 替换为你的文件路径
    data = pd.read_csv(file_path, usecols=[1, 3, 5])  # 读取第 1, 3, 5 列
    data = data.iloc[:1792]

    # 组合信号为一个三通道的输入
    y1 = data.iloc[:, 0].values
    y2 = data.iloc[:, 1].values
    y3 = data.iloc[:, 2].values
    signal = torch.tensor(np.stack([y1, y2, y3]), dtype=torch.float32)  # 形状为 (3, 1792)

    # 定义模型参数
    signal_length = 1792
    level = 8
    ma_window_size = 5
    model = WaveletGatedNet(signal_length=signal_length, wavelet_name='db1', level=level, ma_window_size=ma_window_size)

    # 前向传播
    output_signal = model(signal)  # 模型处理信号
    output_signal = output_signal.detach().numpy()  # 转换为 NumPy 数组以便绘图

    # 绘图
    plt.figure(figsize=(15, 10))
    titles = ['Sensor 1', 'Sensor 2', 'Sensor 3']

    for i in range(3):
        plt.subplot(3, 1, i + 1)
        plt.plot(signal[i].numpy(), label='Original Signal', color='blue', alpha=0.6)
        plt.plot(output_signal[i], label='Processed Signal with Smoothing', color='red', linestyle='--')
        plt.title(f'Wavelet Gated Network Processing - {titles[i]}')
        plt.xlabel('Data Point Index')
        plt.ylabel('Signal Value')
        plt.legend()

    plt.tight_layout()
    plt.show()
