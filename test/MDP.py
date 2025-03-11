import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from collections import defaultdict

def adaptive_rank(X, energy_threshold=0.9):
    """
    通过奇异值分解（SVD）和能量阈值选择自适应秩大小。
    """
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    total_energy = np.sum(S**2)
    cumulative_energy = np.cumsum(S**2) / total_energy
    k = np.argmax(cumulative_energy >= energy_threshold) + 1
    U_k = U[:, :k]
    S_k = np.diag(S[:k])
    Vt_k = Vt[:k, :]
    X_reconstructed = np.dot(U_k, np.dot(S_k, Vt_k))
    return U_k, S_k, Vt_k, k, X_reconstructed

def build_markov_chain(states):
    """
    构建马尔科夫链的状态转移矩阵
    """
    transition_matrix = defaultdict(lambda: defaultdict(int))
    for i in range(len(states) - 1):
        current_state = states[i]
        next_state = states[i + 1]
        transition_matrix[current_state][next_state] += 1

    # 转换为概率矩阵
    state_list = list(set(states))
    state_count = len(state_list)
    prob_matrix = np.zeros((state_count, state_count))

    for i, state_from in enumerate(state_list):
        total_transitions = sum(transition_matrix[state_from].values())
        if total_transitions > 0:
            for j, state_to in enumerate(state_list):
                prob_matrix[i, j] = transition_matrix[state_from][state_to] / total_transitions

    return state_list, prob_matrix

# 示例数据
np.random.seed(42)
X = np.random.randn(100, 50)  # 假设X是时间-频率矩阵

# 1. 使用低秩分解提取特征
U_k, S_k, Vt_k, chosen_rank, X_reconstructed = adaptive_rank(X, energy_threshold=0.9)

# 2. 对低秩特征进行聚类（如 KMeans）
kmeans = KMeans(n_clusters=5, random_state=42)
low_rank_features = U_k @ S_k  # 使用低秩特征
states = kmeans.fit_predict(low_rank_features)

# 3. 构建马尔科夫链
state_list, prob_matrix = build_markov_chain(states)

# 4. 打印状态转移矩阵
print("马尔科夫链的状态列表：", state_list)
print("状态转移概率矩阵：\n", prob_matrix)

# 5. 可视化状态转移矩阵
plt.imshow(prob_matrix, cmap='Blues', interpolation='none')
plt.colorbar()
plt.xticks(range(len(state_list)), state_list)
plt.yticks(range(len(state_list)), state_list)
plt.title('Markov Chain Transition Matrix')
plt.xlabel('Next State')
plt.ylabel('Current State')
plt.show()
