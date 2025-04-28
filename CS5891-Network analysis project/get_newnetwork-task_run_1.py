import os
import networkx as nx
import pandas as pd
import numpy as np
from networkx.algorithms.community import greedy_modularity_communities
from networkx.algorithms.community import modularity

input_dir = 'data_cor'
output_dir = 'newNetwork-task_run_1'
# Initialize an accumulator for the adjacency matrices
sum_adj_matrix = None
node_count = 0  # Track the number of nodes
count = 0
# Process each subject's folder
for subj in range(1, 34):
    subj_id = f'sub-{subj:02d}'
    in_dir = os.path.join(input_dir, subj_id)
    for filename in os.listdir(in_dir):
        if filename.endswith(".nii.gz.csv"):
            file_path = os.path.join(in_dir, filename)

            # Identify the state from the filename
            if "task-rest_run-1" in filename:
                count += 1
                # Load the Pearson correlation matrix from CSV
                correlation_matrix = np.loadtxt(file_path, delimiter=',')
                # Create a mask for positive correlations
                # Initialize the sum matrix if it's the first graph
                if sum_adj_matrix is None:
                    sum_adj_matrix = np.zeros_like(correlation_matrix)
                    node_count = correlation_matrix.shape[0]
                # Add the current matrix to the sum matrix
                sum_adj_matrix += correlation_matrix

average_adj_matrix = sum_adj_matrix / count
positive_mask = average_adj_matrix > 0
positive_matrix = average_adj_matrix * positive_mask  # Retain only positive values
# Create a graph from the average adjacency matrix
G_average = nx.from_numpy_array(positive_matrix)
# Remove self-loops
G_average.remove_edges_from(nx.selfloop_edges(G_average))
# Save the averaged network as a weighted edge list
nx.write_weighted_edgelist(G_average, os.path.join(output_dir, "average_network.edgelist"))
# Optionally, calculate metrics for the averaged network
average_metrics = {
    'weighted_degree_centrality': {
        node: sum(weight for _, _, weight in G_average.edges(node, data='weight'))
        for node in G_average.nodes()
    },
    'clustering_coefficient': nx.clustering(G_average, weight='weight')
}
np.savetxt(os.path.join(output_dir, "average_adjacency_matrix.csv"), positive_matrix, delimiter=',')

# # Save metrics to CSV
# metrics_df = pd.DataFrame({
#     'Node': list(average_metrics['weighted_degree_centrality'].keys()),
#     'WeightedDegreeCentrality': list(average_metrics['weighted_degree_centrality'].values()),
#     'ClusteringCoefficient': [average_metrics['clustering_coefficient'].get(node, np.nan) for node in G_average.nodes()]
# })
# metrics_df.to_csv(os.path.join(output_dir, "average_network_metrics.csv"), index=False)
#
# print("Averaged network and metrics have been saved.")
