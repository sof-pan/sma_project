import pandas as pd
import networkx as nx
import numpy as np
import math
from collections import defaultdict
from neo4j import GraphDatabase
import community as community_louvain
import igraph as ig
import leidenalg
import time
import os
import sys
from preprocessing import GraphPreprocessor
import evaluation

# Logging
timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
log_dir = os.path.join('logs', f"brain_network_external_{timestamp}")
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, 'output.txt')
sys.stdout = open(log_file_path, 'w')
sys.stderr = sys.stdout

start_time = time.time()

# Load brain network
brain_df = pd.read_csv('datasets/bn-fly-drosophila_medulla_1.edges', delim_whitespace=True, comment='#', header=None, names=['source', 'target', 'weight'])
brain_graph = nx.from_pandas_edgelist(brain_df, 'source', 'target', edge_attr='weight')

# Ensure all edges have weights
for u, v in brain_graph.edges():
    if 'weight' not in brain_graph[u][v] or math.isnan(brain_graph[u][v]['weight']):
        brain_graph[u][v]['weight'] = 1

# Preprocess
processor = GraphPreprocessor(
    G=brain_graph,
    sample_fraction=0.9
)
brain_graph_processed = processor.process()

# ---------------- Louvain ---------------- #
start_time_louv = time.time()
partition_louvain = community_louvain.best_partition(brain_graph_processed)
end_time_louv = time.time()
elapsed_louv = end_time_louv - start_time_louv
print("=======================================================")
print(f"Brain Network Louvain Time: {int(elapsed_louv // 3600)}h {int((elapsed_louv % 3600) // 60)}m {elapsed_louv % 60:.2f}s")
print("=======================================================")
community_nodes_louvain = defaultdict(list)
for node, comm_id in partition_louvain.items():
    community_nodes_louvain[comm_id].append(node)

# Annotate Louvain communities
for node, comm_id in partition_louvain.items():
    members_str = ", ".join(map(str, sorted(community_nodes_louvain[comm_id])))
    brain_graph_processed.nodes[node]['louvain_community'] = comm_id
    brain_graph_processed.nodes[node]['louvain_community_members'] = members_str

# ---------------- Leiden ---------------- #
start_time_leid = time.time()
G_nx = nx.convert_node_labels_to_integers(brain_graph_processed, label_attribute="original_id")
igraph_G = ig.Graph.TupleList(G_nx.edges(), directed=False)
original_ids = [data['original_id'] for _, data in G_nx.nodes(data=True)]
igraph_G.vs['name'] = original_ids  # Set node IDs directly to original IDs

# Map igraph indices to original node IDs

partition_leiden = leidenalg.find_partition(igraph_G, leidenalg.ModularityVertexPartition)
end_time_leid = time.time()
elapsed_leid = end_time_leid - start_time_leid
print("=======================================================")
print(f"Brain Network Leiden Time: {int(elapsed_leid // 3600)}h {int((elapsed_leid % 3600) // 60)}m {elapsed_leid % 60:.2f}s")
print("=======================================================")

# Build Leiden partition dict
community_nodes_leiden = defaultdict(list)
pred_partition_dict = {}
for comm_id, community in enumerate(partition_leiden):
    for node_id in community:
        original_id = igraph_G.vs[node_id]['name']
        community_nodes_leiden[comm_id].append(original_id)
        pred_partition_dict[original_id] = comm_id

# Annotate Leiden communities
for node, comm_id in pred_partition_dict.items():
    members_str = ", ".join(map(str, sorted(community_nodes_leiden[comm_id])))
    brain_graph_processed.nodes[node]['leiden_community'] = comm_id
    brain_graph_processed.nodes[node]['leiden_community_members'] = members_str

# ---------------- Evaluation ---------------- #
evaluation.evaluate_communities_without_ground_truth(brain_graph_processed, partition_louvain, "Louvain")
# evaluation.evaluate_communities_without_ground_truth(brain_graph_processed, pred_partition_dict, "Leiden")

evaluation.evaluate_cpm(brain_graph_processed, partition_louvain, gamma=0.2, method="Louvain")
evaluation.evaluate_cpm(brain_graph_processed, pred_partition_dict, gamma=0.2, method="Leiden")

# ---------------- Neo4j Export ---------------- #
# uri = "bolt://localhost:7687"
# user = "neo4j"
# password = "test1234"
# driver = GraphDatabase.driver(uri, auth=(user, password))

# with driver.session() as session:
#     session.run("MATCH (n) DETACH DELETE n")

#     # Export Louvain nodes
#     for node, data in brain_graph_processed.nodes(data=True):
#         session.run(
#             """
#             CREATE (n:NeuronLouvain {
#                 id: $id,
#                 degree_centrality: $dc,
#                 betweenness_centrality: $bc,
#                 community: $community,
#                 community_members: $community_members
#             })
#             """,
#             {
#                 "id": node,
#                 "dc": data.get("degree_centrality"),
#                 "bc": data.get("betweenness_centrality"),
#                 "community": data.get("louvain_community"),
#                 "community_members": data.get("louvain_community_members")
#             }
#         )

#     # Export Leiden nodes
#     for node, data in brain_graph_processed.nodes(data=True):
#         session.run(
#             """
#             CREATE (n:NeuronLeiden {
#                 id: $id,
#                 degree_centrality: $dc,
#                 betweenness_centrality: $bc,
#                 community: $community,
#                 community_members: $community_members
#             })
#             """,
#             {
#                 "id": node,
#                 "dc": data.get("degree_centrality"),
#                 "bc": data.get("betweenness_centrality"),
#                 "community": data.get("leiden_community"),
#                 "community_members": data.get("leiden_community_members")
#             }
#         )

#     # Export edges for both
#     for u, v, data in brain_graph_processed.edges(data=True):
#         session.run(
#             """
#             MATCH (aL:NeuronLouvain {id: $source}), (bL:NeuronLouvain {id: $target}),
#                   (aLe:NeuronLeiden {id: $source}), (bLe:NeuronLeiden {id: $target})
#             CREATE (aL)-[:CONNECTS {weight: $weight}]->(bL),
#                    (aLe)-[:CONNECTS {weight: $weight}]->(bLe)
#             """,
#             {
#                 "source": u,
#                 "target": v,
#                 "weight": data.get("weight", 1)
#             }
#         )

# driver.close()

# ---------------- Done ---------------- #
end_time = time.time()
elapsed = end_time - start_time
print("=======================================================")
print(f"Brain Network Total Time: {int(elapsed // 3600)}h {int((elapsed % 3600) // 60)}m {elapsed % 60:.2f}s")
print("=======================================================")
