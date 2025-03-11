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

# 加入bs和channels
class WaveletPacketTransform(nn.Module):
    def __init__(self, wavelet_name='db1', level=3):
        super(WaveletPacketTransform, self).__init__()
        self.wavelet = wavelet_name
        self.level = level

    def forward(self, x):
        batch_size, channels, length = x.shape
        coeffs = []
        for b in range(batch_size):
            channel_coeffs = []
            for c in range(channels):
                # 单独对每个通道的信号进行小波包分解
                wp = pywt.WaveletPacket(data=x[b, c].cpu().numpy(), wavelet=self.wavelet, mode='symmetric', maxlevel=self.level)
                nodes = wp.get_level(self.level, order='freq')
                channel_coeffs.append([torch.tensor(node.data).to(x.device) for node in nodes])
            coeffs.append(channel_coeffs)
        return coeffs


    
class InverseWaveletPacketTransform(nn.Module):
    def __init__(self, wavelet_name='db1', level=3):
        super(InverseWaveletPacketTransform, self).__init__()
        self.wavelet = wavelet_name
        self.level = level

    def forward(self, coeffs):
        batch_size = len(coeffs)
        channels = len(coeffs[0])
        reconstructed = []
        for b in range(batch_size):
            channel_reconstructed = []
            for c in range(channels):
                dummy_data = [0] * (2 ** self.level*7)   # dummy_data = [0] * (2 ** self.level*7)
                wp = pywt.WaveletPacket(data=dummy_data, wavelet=self.wavelet, mode='symmetric', maxlevel=self.level)
                nodes = wp.get_level(self.level, order='freq')

                if len(nodes) != len(coeffs[b][c]):
                    raise ValueError(f"系数数量 ({len(coeffs[b][c])}) 与节点数量 ({len(nodes)}) 不匹配，级别为 {self.level}")
                
                for i, node in enumerate(nodes):
                    node.data = coeffs[b][c][i].detach().cpu().numpy().flatten()
                reconstructed_signal = wp.reconstruct(update=True)
                channel_reconstructed.append(torch.tensor(reconstructed_signal).to(coeffs[0][0][0].device))
            reconstructed.append(torch.stack(channel_reconstructed, dim=0))
        return torch.stack(reconstructed, dim=0)
    

# join bs and channels
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
        for batch in coeffs:
            batch_gated = []
            for channel_coeffs in batch:
                channel_gated = []
                for c in channel_coeffs:
                    gate_weight = self.gate(c)
                    gated_coeff = c * gate_weight

                    # 应用自适应阈值
                    gated_coeff[torch.abs(gated_coeff) < self.threshold] = 0
                    channel_gated.append(gated_coeff)
                batch_gated.append(channel_gated)
            gated_coeffs.append(batch_gated)
        return gated_coeffs



class WaveletGatedNet(nn.Module):
    def __init__(self, signal_length, wavelet_name='db1', level=3):
        super(WaveletGatedNet, self).__init__()
        self.wavelet_transform = WaveletPacketTransform(wavelet_name, level)
        self.gated_unit = AdaptiveGatedUnit(signal_length=signal_length, level=level)
        self.inverse_wavelet_transform = InverseWaveletPacketTransform(wavelet_name,level)

    def forward(self, x):
        # Step 1: Wavelet Packet Decomposition
        temp = x
        coeffs = self.wavelet_transform(x)
        #print(f"Input shape: {x.shape}")
        #print(f"Coeffs shape after decomposition: [[{c.shape for c in channel}] for channel in coeffs]")

     
        # Step 2: Apply Gated Unit
        gated_coeffs = self.gated_unit(coeffs)
        #print(gated_coeffs.shape)

        # # Step 3: Inverse Wavelet Packet Transform
        reconstructed_signal = self.inverse_wavelet_transform(gated_coeffs) +temp
      

        return reconstructed_signal





# Example usage
if __name__ == "__main__":
    # Generate a random 1D signal
    import torch
    import torch.nn as nn
    import matplotlib.pyplot as plt

    signal = torch.randn([32, 3, 1792])
 
    signal_length = 1792
    level = 8
    # Initialize model
    model = WaveletGatedNet(signal_length=signal_length, wavelet_name='db1',level=level)

    # Forward pass
    output_signal = model(signal)
    print("Input Signal:", signal.shape)
    print("Output Signal:", output_signal.shape)

    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Model total parameter: %.4fMB\n' % (model_params/1024/1024))

    # model_state_dict = model.state_dict()

    # small_weights = {}
    # for name, param in model_state_dict.items():
    #     if param.requires_grad:  # 只关注需要更新的参数
    #         small_weights[name] = torch.abs(param)

    # # 设置阈值
    # threshold = 0.0001  # 示例阈值
    # small_weight_names = {name: param for name, param in small_weights.items() if torch.all(param < threshold)}

    # # 查看小权重
    # print("小于阈值的权重:")
    # for name, weights in small_weight_names.items():
    #     print(f'Layer: {name}, Small weights: {weights}')

    # # 可视化权重分布
    # flattened_weights = [torch.flatten(param) for param in small_weights.values()]
    # if flattened_weights:
    #     all_weights = torch.cat(flattened_weights)
    #     plt.hist(all_weights.cpu().numpy(), bins=50)
    #     plt.title('Weight Distribution')
    #     plt.xlabel('Weight Value')
    #     plt.ylabel('Frequency')
    #     plt.axvline(x=threshold, color='r', linestyle='--', label='Threshold')
    #     plt.legend()
    #     plt.show()
    # else:
    #     print("没有小于阈值的权重，无法进行连接。")