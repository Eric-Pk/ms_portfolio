import os
import networkx as nx
import pandas as pd
import numpy as np

# 1. 设置文件夹路径
folder_path = "path/to/your/csv_files"  #
output_folder = "path/to/output_files"  #
metrics_output_file = "brain_network_metrics_summary.csv"  # 保存指标的CSV文件
os.makedirs(output_folder, exist_ok=True)


# 2. 定义计算小世界指数的函数
def calculate_small_world_index(G, n_iter=100):
    # 生成随机图
    random_graph = nx.expected_degree_graph([d for _, d in G.degree()], selfloops=False)

    # 随机图属性
    random_clustering = nx.average_clustering(random_graph)
    random_path_length = nx.average_shortest_path_length(random_graph)

    # 实际图属性
    clustering = nx.average_clustering(G)
    path_length = nx.average_shortest_path_length(G)

    # 小世界指数
    small_world_index = (clustering / random_clustering) / (path_length / random_path_length)
    return small_world_index


# 3. 定义批量处理函数
def process_csv_file(file_path, output_folder, threshold=0.3):
    # 加载CSV为相关系数矩阵
    df = pd.read_csv(file_path, index_col=0)
    matrix = df.values

    # 检查矩阵尺寸
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Matrix in {file_path} is not square: {matrix.shape}")

    # 避免自环：将对角线元素排除（设置为0，仅用于构建邻接矩阵）
    np.fill_diagonal(matrix, 0)

    # 将相关系数矩阵转化为邻接矩阵
    adjacency_matrix = (matrix >= threshold).astype(int)

    G = nx.Graph()

    # 创建图
    nodes = df.index
    G.add_nodes_from(nodes)

    for i, node1 in enumerate(nodes):
        for j, node2 in enumerate(nodes):
            if adjacency_matrix[i, j] != 0:  # 避免自环已经通过对角线处理实现
                G.add_edge(node1, node2, weight=matrix[i, j])

    # 网络基本信息
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    density = nx.density(G)

    # 中心性指标
    degree_centrality = nx.degree_centrality(G)
    clustering_coefficient = nx.average_clustering(G)

    # 特征路径长度
    try:
        characteristic_path_length = nx.average_shortest_path_length(G)
    except nx.NetworkXError:
        characteristic_path_length = None  # 图可能不是连通的

    # 小世界指数
    try:
        small_world_index = calculate_small_world_index(G)
    except Exception as e:
        small_world_index = None

    # 导出网络为 GEXF 格式
    output_file = os.path.join(output_folder, os.path.splitext(os.path.basename(file_path))[0] + ".gexf")
    nx.write_gexf(G, output_file)

    # 返回分析结果
    return {
        "file_name": os.path.basename(file_path),
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "density": density,
        "clustering_coefficient": clustering_coefficient,
        "characteristic_path_length": characteristic_path_length,
        "small_world_index": small_world_index
    }


# 4. 批量处理文件夹中的所有 CSV 文件
all_metrics = []

for file_name in os.listdir(folder_path):
    if file_name.endswith(".csv"):  # 只处理CSV文件
        file_path = os.path.join(folder_path, file_name)
        try:
            metrics = process_csv_file(file_path, output_folder, threshold=0.3)
            all_metrics.append(metrics)
        except Exception as e:
            print(f"Error processing {file_name}: {e}")

# 5. 保存所有指标到CSV文件
metrics_df = pd.DataFrame(all_metrics)
metrics_df.to_csv(metrics_output_file, index=False)
print(f"Network metrics saved to {metrics_output_file}")

print("Batch processing and metrics analysis completed!")
