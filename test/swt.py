import numpy as np
import matplotlib.pyplot as plt

def frequency_shift_multi_channel(signals, sampling_rate, shift_amount):
    """
    对多个传感器信号进行统一的频率平移。
    
    参数:
    - signals: 形状为 (num_sensors, num_points) 的二维数组
    - sampling_rate: 采样频率（Hz）
    - shift_amount: 频率平移量（Hz），正数为上移，负数为下移
    
    返回:
    - shifted_signals: 经过频率扰动后的信号，形状与输入相同
    """
    num_sensors, num_points = signals.shape
    shifted_signals = np.zeros_like(signals)
    
    # 计算傅里叶变换的频率坐标
    freqs = np.fft.fftfreq(num_points, d=1/sampling_rate)
    shift_factor = np.exp(2j * np.pi * shift_amount * freqs)
    
    for i in range(num_sensors):
        # 对每个传感器信号进行傅里叶变换
        freq_domain = np.fft.fft(signals[i, :])
        
        # 进行频率平移
        shifted_freq_domain = freq_domain * shift_factor
        
        # 逆傅里叶变换回时域
        shifted_signals[i, :] = np.real(np.fft.ifft(shifted_freq_domain))
    
    return shifted_signals

# 示例用法
sampling_rate = 2000  # 采样频率（Hz）
num_points = 1792
time = np.arange(0, num_points) / sampling_rate

# 生成三个传感器的示例信号
original_signals = np.zeros((3, num_points))
original_signals[0, :] = np.sin(2 * np.pi * 50 * time)        # 传感器1
original_signals[1, :] = 0.5 * np.sin(2 * np.pi * 120 * time) # 传感器2
original_signals[2, :] = 0.3 * np.sin(2 * np.pi * 200 * time) # 传感器3

# 对信号进行频率上移 20 Hz
shifted_signals = frequency_shift_multi_channel(original_signals, sampling_rate, shift_amount=20)

# 绘图比较原始信号和频率扰动后的信号
plt.figure(figsize=(12, 8))
for i in range(3):
    plt.subplot(3, 2, 2*i + 1)
    plt.title(f"Original Signal - Sensor {i+1}")
    plt.plot(time, original_signals[i, :])
    
    plt.subplot(3, 2, 2*i + 2)
    plt.title(f"Frequency Shifted Signal - Sensor {i+1}")
    plt.plot(time, shifted_signals[i, :])

plt.tight_layout()
plt.show()
