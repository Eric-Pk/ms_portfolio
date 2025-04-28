import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os

# Define directories for input and output
connectivity_dir = '..'
output_dir = '..'

os.makedirs(output_dir, exist_ok=True)

# Set a higher threshold to retain only the strongest connections
threshold = 0.3  # Adjust this value to make the network sparser

# List all connectivity matrix files
file_list = [f for f in os.listdir(connectivity_dir) if f.endswith('.csv')]

for i, filename in enumerate(file_list):
    file_path = os.path.join(connectivity_dir, filename)
    print(f"Processing binary network for file {i + 1}/{len(file_list)}: {filename}")

    matrix = np.loadtxt(file_path, delimiter=",")

    # Apply the threshold to create a binary matrix
    binary_matrix = np.where(matrix >= threshold, 1, 0)

    # Create undirected graph
    G = nx.from_numpy_array(binary_matrix)
    G.remove_edges_from(nx.selfloop_edges(G))

    #清除网络里的孤点
    isolated_nodes = list(nx.isolates(G))
    if isolated_nodes:
        print(f"Isolated nodes detected in {filename}: {len(isolated_nodes)}")
        G.remove_nodes_from(isolated_nodes)
    else:
        print(f"No isolated nodes in {filename}")

    # spring layout
    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(G, seed=42)

    nx.draw(G, pos, with_labels=False, node_size=50, node_color='blue', edge_color='black', width=0.5, alpha=0.7)

    # Save each binary network image with a unique name
    output_image = os.path.join(output_dir, f'binary_network_{filename}.png')
    plt.savefig(output_image)
    plt.close()
    print(f"Binary network image saved for {filename} at {output_image}\n")

print("Binary network visualization completed for all participants.")




