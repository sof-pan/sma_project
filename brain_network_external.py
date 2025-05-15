import networkx as nx
from preprocessing import GraphPreprocessor
import community as community_louvain
import igraph as ig
import leidenalg
import pandas as pd
import math

# Load the graph structure from edge list
brain_df = pd.read_csv('datasets/bn-fly-drosophila_medulla_1.edges', delim_whitespace=True, comment='#', header=None, names=['source', 'target', 'weight'])
brain_graph = nx.from_pandas_edgelist(brain_df, 'source', 'target', edge_attr='weight')

# Add weight to edges if it's not already there
for node1, node2 in brain_graph.edges():
    if 'weight' not in brain_graph[node1][node2] or math.isnan(brain_graph[node1][node2]['weight']):
        brain_graph[node1][node2]['weight'] = 1

processor = GraphPreprocessor(
    G=brain_graph,
    sample_fraction=0.9
)

brain_graph_processed = processor.process()

# LOUVAIN
partition_louvain = community_louvain.best_partition(brain_graph)

# Organize nodes by community
communities = {}
edge_weights = {}
for node, community in partition_louvain.items():
    if community not in communities:
        communities[community] = []
    communities[community].append(node)

# Collect edge weights for each community
for node1, node2, data in brain_graph.edges(data=True):
    weight = data.get('weight', 1)
    community1 = partition_louvain[node1]
    community2 = partition_louvain[node2]
    
    # If the communities of both nodes are the same, add the edge weight to that community
    if community1 == community2:
        if community1 not in edge_weights:
            edge_weights[community1] = 0
        edge_weights[community1] += weight

# Print communities and their corresponding edge weights
print("Final Community Assignments with Edge Weights:")
for community, nodes in communities.items():
    community_weight = edge_weights.get(community, 0)  # Get total weight for the community
    print(f"Community {community}: {nodes}, Total Edge Weight: {community_weight}")


# LEIDEN
G_nx = nx.convert_node_labels_to_integers(brain_graph_processed, label_attribute="original_id")
igraph_G = ig.Graph.TupleList(G_nx.edges(), directed=False)
igraph_G.vs['original_id'] = [data['original_id'] for _, data in G_nx.nodes(data=True)]

# Map: NetworkX node -> iGraph vertex index
nx_to_ig = {node_id: idx for idx, node_id in enumerate(igraph_G.vs['original_id'])}
partition_leiden = leidenalg.find_partition(igraph_G, leidenalg.ModularityVertexPartition)

# Collect communities
communities_leiden = {}
for node_idx, community_id in enumerate(partition_leiden.membership):
    nx_id = igraph_G.vs[node_idx]['original_id']
    if community_id not in communities_leiden:
        communities_leiden[community_id] = []
    communities_leiden[community_id].append(nx_id)

# Collect intra-community edge weights
edge_weights_leiden = {}
for u, v, data in brain_graph_processed.edges(data=True):
    if u not in nx_to_ig or v not in nx_to_ig:
        continue  # Skip isolated nodes
    weight = data.get('weight', 1)
    comm_u = partition_leiden.membership[nx_to_ig[u]]
    comm_v = partition_leiden.membership[nx_to_ig[v]]
    if comm_u == comm_v:
        edge_weights_leiden[comm_u] = edge_weights_leiden.get(comm_u, 0) + weight

# Print results
print("Final Leiden Community Assignments with Edge Weights:")
for community_id, nodes in communities_leiden.items():
    total_weight = edge_weights_leiden.get(community_id, 0)
    print(f"Community {community_id}: {nodes}, Total Edge Weight: {total_weight}")