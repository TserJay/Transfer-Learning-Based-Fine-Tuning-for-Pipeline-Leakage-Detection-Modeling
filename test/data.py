import pandas as pd
import os

def process_csv_files_in_place(folder_path):
    # 遍历文件夹中的所有 CSV 文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            
            # 读取 CSV 文件
            df = pd.read_csv(file_path)
            
            # 检查文件是否有至少 3 列
            if df.shape[1] == 3:
                # 创建一个空 DataFrame，列数设置为 6
                new_df = pd.DataFrame(index=df.index, columns=range(6))
                
                # 将第 0, 1, 2 列的数据分别复制到新 DataFrame 的第 1, 3, 5 列
                new_df.iloc[:, 1] = df.iloc[:, 0]  # 将第 0 列的数据放到第 1 列
                new_df.iloc[:, 3] = df.iloc[:, 1]  # 将第 1 列的数据放到第 3 列
                new_df.iloc[:, 5] = df.iloc[:, 2]  # 将第 2 列的数据放到第 5 列
                
                # 保存修改后的文件（覆盖原始文件）
                new_df.to_csv(file_path, index=False)
                
                print(f'已处理并覆盖文件: {filename}')
            else:
                print(f'文件 {filename} 的列数不是 3 列，跳过处理')

    print("所有文件已处理完毕")

# 设置文件夹路径（包含所有 CSV 文件的文件夹）
folder_path = r'E:\projects\UDTL-LoRA\data\leak_TL_0.6Mpa\3\4'  # 请替换为你的实际文件夹路径

# 执行批量处理
process_csv_files_in_place(folder_path)