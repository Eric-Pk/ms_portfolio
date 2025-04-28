import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

# Load the relation matrix
relation_matrix = np.loadtxt('newNetwork-sleep/average_adjacency_matrix.csv', delimiter=',')

# Initialize the graph for positive correlations
positive_graph = nx.Graph()

# Traverse the matrix and add edges with positive weights
num_nodes = relation_matrix.shape[0]
for i in range(num_nodes):
    for j in range(num_nodes):
        if i != j:  # Ignore self-loops
            weight = relation_matrix[i, j]
            if weight > 0:  # Positive correlation
                positive_graph.add_edge(i, j, weight=weight)

# Define the layout for the graph
pos = nx.spring_layout(positive_graph, seed=42)

# Function: Draw weighted network with color gradients
def draw_weighted_network(graph, title, edge_cmap, edge_color_label):
    plt.figure(figsize=(10, 10))

    # Extract edges and their weights
    edges = graph.edges(data=True)
    weights = [d['weight'] for (u, v, d) in edges]

    # Create a color map for edge weights
    norm = Normalize(vmin=min(weights), vmax=max(weights))  # Normalize weights
    cmap = cm.get_cmap(edge_cmap)  # Get the colormap

    edge_colors = [cmap(norm(w)) for w in weights]  # Map weights to colors

    # Draw the network
    nx.draw(
        graph,
        pos,
        with_labels=False,
        node_size=50,
        node_color='gray',
        edge_color=edge_colors,
        width=[w * 5 for w in weights],  # Edge thickness proportional to weight
        alpha=0.8
    )

    # Add a color bar for the edge colors
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array(weights)
    cbar = plt.colorbar(sm, ax=plt.gca())
    cbar.set_label(edge_color_label)

    plt.title(title)
    plt.show()

# Draw the positive correlation network
draw_weighted_network(
    positive_graph,
    title="Positive Correlation Network",
    edge_cmap='Blues',  # Use a blue gradient for positive weights
    edge_color_label="Positive Weight"
)
