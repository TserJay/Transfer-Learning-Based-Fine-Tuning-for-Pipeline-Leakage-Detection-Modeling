import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 假设 X 是信号矩阵，每列代表一个传感器的数据
def adaptive_rank(X, energy_threshold=0.9):
    """
    通过奇异值分解（SVD）和能量阈值选择自适应秩大小。
    
    参数:
    X: 输入信号矩阵 (例如时间-频率矩阵)
    energy_threshold: 保留的累积方差比例，默认0.9，意味着保留90%的能量。
    
    返回:
    U, S, V: SVD分解后的矩阵
    k: 选择的秩大小
    """
    # 奇异值分解（SVD）
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    
    # 计算奇异值的累积能量
    total_energy = np.sum(S**2)
    cumulative_energy = np.cumsum(S**2) / total_energy
    
    # 根据能量阈值选择秩大小
    k = np.argmax(cumulative_energy >= energy_threshold) + 1
    
    # 使用前k个奇异值进行矩阵重构
    U_k = U[:, :k]
    S_k = np.diag(S[:k])
    Vt_k = Vt[:k, :]
    
    # 重构信号
    X_reconstructed = np.dot(U_k, np.dot(S_k, Vt_k))
    
    return U_k, S[:k], Vt_k, k, X_reconstructed

# 示例数据，假设X是时间-频率矩阵
X = np.random.randn(100, 50)  # 100个时间步，50个频率成分或传感器数据

# 调用自适应秩选择函数
U_k, S_k, Vt_k, chosen_rank, X_reconstructed = adaptive_rank(X, energy_threshold=0.9)

# 打印选择的秩大小
print(f"Chosen rank: {chosen_rank}")

# 可视化奇异值的累积能量
plt.plot(np.cumsum(S_k**2) / np.sum(S_k**2))
plt.axvline(x=chosen_rank, color='r', linestyle='--', label=f"Chosen Rank: {chosen_rank}")
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Energy')
plt.legend()
plt.title('Cumulative Energy vs. Rank')
plt.show()
