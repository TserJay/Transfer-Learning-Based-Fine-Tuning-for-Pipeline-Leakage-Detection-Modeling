import pandas as pd
import matplotlib.pyplot as plt
import os

# 读取CSV文件
df = pd.read_csv(r'E:\projects\UDTL-leak_v2\data\leak_signals\0\4\(50).csv')


# 选择特定列作为值
values_columns = [1, 3, 5]  # 选择第1、3、5列作为值

# 提取值数据
values = df.iloc[:, values_columns].values

# 绘制图表
plt.figure(figsize=(10, 6))
for i in range(len(values_columns)):
    plt.plot(range(len(values)), values[:, i], label=f'Value {i+1}')

plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Plot of Values by Index')
plt.legend()
plt.grid(True)

# plt.savefig(os.path.join(r'E:\projects\UDTL-leak_v2\data\leak_signals\0\2', '(1).jpg'))

plt.show()

