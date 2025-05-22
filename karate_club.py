import networkx as nx
import community as community_louvain
from louvain import Louvain as Lvn
from leiden import Leiden as Ldn
from collections import defaultdict
import evaluation
from preprocessing import GraphPreprocessor
import igraph as ig
import leidenalg
import time

# Load sample graph with ground truth (karate club)
G: nx.Graph = nx.karate_club_graph()

# Ground truth communities from node attribute 'club'
# Map club names to integer labels
true_labels = []
club_to_int = {'Mr. Hi': 0, 'Officer': 1}
for node in G.nodes():
    true_labels.append(club_to_int[G.nodes[node]['club']])

processed_graph = GraphPreprocessor(G, z_threshold=5).process()

# --- Louvain from NetworkX
start_time = time.time()
partition_louvain = community_louvain.best_partition(processed_graph)
end_time = time.time()
elapsed = end_time - start_time
print("=======================================================")
print(f"Karate Club Network Louvain Networkx Time: {int(elapsed // 3600)}h {int((elapsed % 3600) // 60)}m {elapsed % 60:.4f}s")
print("=======================================================")
community_nodes_louvain = defaultdict(list)
louvain = Lvn(processed_graph)
# --- Custom Louvain
_, _, _, louvain_final_partition = louvain.run(print_results=False)

# --- Leiden from community-igraph
G_ig = ig.Graph.TupleList(processed_graph.edges(), directed=False)
G_ig.vs["name"] = list(processed_graph.nodes())
start_time = time.time()
leiden_igraph_partition = leidenalg.find_partition(G_ig, leidenalg.ModularityVertexPartition)
partition_leiden = {
    int(G_ig.vs[i]["name"]): cid
    for i, cid in enumerate(leiden_igraph_partition.membership)
}
end_time = time.time()
elapsed = end_time - start_time
print("=======================================================")
print(f"Karate Club Network Leiden Community Time: {int(elapsed // 3600)}h {int((elapsed % 3600) // 60)}m {elapsed % 60:.4f}s")
print("=======================================================")
# --- Custom Leiden
leiden = Ldn(processed_graph)
_, _, _, leiden_final_partition = leiden.run(print_results=False)

node_labels = {node: club_to_int[G.nodes[node]['club']] for node in G.nodes()}
evaluation.evaluate_communities_with_ground_truth(louvain_final_partition, node_labels, "Louvain")
evaluation.evaluate_cpm(G, louvain_final_partition, gamma=0.5, method="Louvain")
evaluation.evaluate_communities_with_ground_truth(leiden_final_partition, node_labels, "Leiden")
evaluation.evaluate_cpm(G, leiden_final_partition, gamma=0.5, method="Leiden")

evaluation.evaluate_communities_with_ground_truth(partition_louvain, node_labels, "Louvain NetworkX")
evaluation.evaluate_cpm(G, partition_louvain, gamma=0.5, method="Louvain NetworkX")

evaluation.evaluate_communities_with_ground_truth(partition_leiden, node_labels, "Leiden Community")
evaluation.evaluate_cpm(G, partition_leiden, gamma=0.5, method="Leiden Community")
print("=======================================================")

# print communities
communities = defaultdict(list)
for node, comm_id in partition_louvain.items():
    communities[comm_id].append(node)
print("Detected communities for Louvain NetworkX:", dict(communities))
print("=======================================================")

communities_leid = defaultdict(list)
for node, comm_id in partition_leiden.items():
    communities_leid[comm_id].append(node)
print("Detected communities for Leiden Community:", dict(communities_leid))
print("=======================================================")

print("Detected communities for Louvain:")
louvain.print_final_community_assignments_and_edge_weights()
print("=======================================================")

print("Detected communities for Leiden:")
louvain.print_final_community_assignments_and_edge_weights()
print("=======================================================")
