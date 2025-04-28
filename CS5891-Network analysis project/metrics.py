import os
import networkx as nx
import pandas as pd
import numpy as np
from networkx.algorithms.community import greedy_modularity_communities
from networkx.algorithms.community import modularity

input_dir = 'data_cor'
output_dir = 'metrics'

os.makedirs(output_dir, exist_ok=True)

for subj in range(1, 34):
    subj_id = f'sub-{subj:02d}'
    in_dir = os.path.join(input_dir, subj_id)
    output_subj_dir = os.path.join(output_dir, subj_id)
    os.makedirs(output_subj_dir, exist_ok=True)
    for filename in os.listdir(in_dir):
        file_path = os.path.join(in_dir, filename)
        # Load the Pearson correlation matrix from CSV
        correlation_matrix = np.loadtxt(file_path, delimiter=',')

        # Create a mask for positive correlations
        positive_mask = correlation_matrix > 0
        positive_matrix = correlation_matrix * positive_mask  # Retain only positive values

        # Create a mask for negative correlations
        negative_mask = correlation_matrix < 0
        negative_matrix = np.abs(correlation_matrix * negative_mask)  # Retain absolute value of negative values

        # Construct the graphs directly from the adjacency matrices
        G_positive = nx.from_numpy_array(positive_matrix)
        G_negative = nx.from_numpy_array(negative_matrix)

        # Remove self loops
        G_positive.remove_edges_from(nx.selfloop_edges(G_positive))
        G_negative.remove_edges_from(nx.selfloop_edges(G_negative))

        # Save graphs to files (optional, can remove if not needed)
        nx.write_weighted_edgelist(G_positive, os.path.join(output_subj_dir, f"{filename}_positive.edgelist"))
        nx.write_weighted_edgelist(G_negative, os.path.join(output_subj_dir, f"{filename}_negative.edgelist"))

        # Calculate metrics
        metrics_positive = {}
        metrics_negative = {}

        # Weighted degree centrality
        metrics_positive['weighted_degree_centrality'] = {
            node: sum(weight for _, _, weight in G_positive.edges(node, data='weight'))
            for node in G_positive.nodes()}
        metrics_negative['weighted_degree_centrality'] = {
            node: sum(weight for _, _, weight in G_negative.edges(node, data='weight'))
            for node in G_negative.nodes()}

        # Clustering coefficient
        metrics_positive['clustering_coefficient'] = nx.clustering(G_positive, weight='weight')

        # Modularity
        communities = greedy_modularity_communities(G_positive, weight='weight')
        partition = [set(community) for community in communities]
        modularity_score = modularity(G_positive, partition, weight='weight')

        # Save node-based metrics
        metrics_df = pd.DataFrame({
            'Node': list(metrics_positive['weighted_degree_centrality'].keys()),
            'WeightedDegreeCentralityPos': list(metrics_positive['weighted_degree_centrality'].values()),
            'WeightedDegreeCentralityneg': list(metrics_negative['weighted_degree_centrality'].values()),
            'ClusteringCoefficient': [metrics_positive['clustering_coefficient'].get(node, np.nan) for node in G_positive.nodes()]
        })
        metrics_df.to_csv(os.path.join(output_subj_dir, f"{filename}_node_metrics.csv"), index=False)

        # Save modularity score
        with open(os.path.join(output_subj_dir, f"{filename}_modularity.txt"), "w") as f:
            f.write(f"Modularity: {modularity_score}\n")
