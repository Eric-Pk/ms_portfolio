import numpy as np
import os

timeseries_dir = '..'
output_dir = '..'

os.makedirs(output_dir, exist_ok=True)

file_list = [f for f in os.listdir(timeseries_dir) if f.endswith('.csv')]

for i, filename in enumerate(file_list):
    file_path = os.path.join(timeseries_dir, filename)
    print(f"Processing time series for file {i + 1}/{len(file_list)}: {filename}")

    # 加载时间序列数据
    time_series = np.loadtxt(file_path, delimiter=",")

    # 计算功能连接矩阵（Pearson相关性矩阵）
    connectivity_matrix = np.corrcoef(time_series.T)

    # 保存功能连接矩阵
    output_file = os.path.join(output_dir, f'connectivity_{filename}')
    np.savetxt(output_file, connectivity_matrix, delimiter=",")
    print(f"Connectivity matrix for {filename} saved to {output_file}\n")

print("Connectivity matrix computation completed for all files.")
