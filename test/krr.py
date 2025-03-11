import numpy as np
from sklearn.kernel_ridge import KernelRidge
import matplotlib.pyplot as plt

# 生成示例数据：一个带噪声的信号
np.random.seed(0)
X = np.linspace(0, 10, 100)[:, None]  # 输入信号 (100 个点)
true_signal = np.sin(X).ravel()  # 原始信号（正弦函数）
noise = 0.3 * np.random.randn(100)  # 噪声
observed_signal = true_signal + noise  # 带噪信号

# 使用核岭回归进行信号重构
alpha = 1e-2  # 正则化参数，值越大，正则化效果越强
kr = KernelRidge(kernel='rbf', gamma=1.0, alpha=alpha)  # 高斯核 + 正则化项 alpha
kr.fit(X, observed_signal)  # 拟合带噪信号
reconstructed_signal = kr.predict(X)  # 重构信号

# 绘图比较
plt.figure(figsize=(10, 6))
plt.plot(X, true_signal, label="True Signal", linestyle='--', color='green')
plt.plot(X, observed_signal, label="Observed Signal (Noisy)", linestyle='-', color='red', alpha=0.6)
plt.plot(X, reconstructed_signal, label="Reconstructed Signal (RKHS with Regularization)", linestyle='-', color='blue')
plt.legend()
plt.xlabel("X")
plt.ylabel("Signal Value")
plt.title("Signal Denoising using Kernel Ridge Regression (RKHS)")
plt.show()
