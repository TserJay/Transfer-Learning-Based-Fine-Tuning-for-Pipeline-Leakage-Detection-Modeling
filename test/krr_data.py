import pandas as pd
import numpy as np
from sklearn.kernel_ridge import KernelRidge
import matplotlib.pyplot as plt

file_path = r'E:\projects\UDTL-LoRA\data\leak_signals\3\7\(10).csv'  # 替换为你的文件路径
data = pd.read_csv(file_path, usecols=[1, 3, 5])  # 读取第 1, 3, 5 列
data = data.iloc[:1792]

# 提取列数据
X = np.linspace(0, 1, 1792)[:, None]  # 输入信号 (1792 个点)
y1 = data.iloc[:, 0].values  # 第 1 列作为第一个目标变量（传感器 1）
y2 = data.iloc[:, 1].values  # 第 2 列作为第二个目标变量（传感器 2）
y3 = data.iloc[:, 2].values  # 第 3 列作为第三个目标变量（传感器 3）

# 设置正则化参数
alpha = 0  # 正则化强度

# 创建并训练模型
krr1 = KernelRidge(kernel='rbf', alpha=alpha, gamma=1.0)
krr1.fit(X, y1)

krr2 = KernelRidge(kernel='rbf', alpha=alpha, gamma=1.0)
krr2.fit(X, y2)

krr3 = KernelRidge(kernel='rbf', alpha=alpha, gamma=1.0)
krr3.fit(X, y3)

# 进行预测
y1_pred = krr1.predict(X)
y2_pred = krr2.predict(X)
y3_pred = krr3.predict(X)

# 绘图比较
plt.figure(figsize=(20, 10))
t = range(len(X))  # 生成 0 到 1791 的索引

# 对于传感器 1 的真实值和预测值
plt.subplot(3, 1, 1)
plt.plot(t, y1, label='Observed (Sensor 1)', color='blue', alpha=0.6)
plt.plot(t, y1_pred, label='Predicted (Sensor 1)', color='red', linestyle='--')
plt.title('Kernel Ridge Regression for Sensor 1')
plt.xlabel('Data Point Index')
plt.ylabel('Value')
plt.legend()

# 对于传感器 2 的真实值和预测值
plt.subplot(3, 1, 2)
plt.plot(t, y2, label='Observed (Sensor 2)', color='blue', alpha=0.6)
plt.plot(t, y2_pred, label='Predicted (Sensor 2)', color='red', linestyle='--')
plt.title('Kernel Ridge Regression for Sensor 2')
plt.xlabel('Data Point Index')
plt.ylabel('Value')
plt.legend()

# 对于传感器 3 的真实值和预测值
plt.subplot(3, 1, 3)
plt.plot(t, y3, label='Observed (Sensor 3)', color='blue', alpha=0.6)
plt.plot(t, y3_pred, label='Predicted (Sensor 3)', color='red', linestyle='--')
plt.title('Kernel Ridge Regression for Sensor 3')
plt.xlabel('Data Point Index')
plt.ylabel('Value')
plt.legend()

plt.tight_layout()
plt.show()
