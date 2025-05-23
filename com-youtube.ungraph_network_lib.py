import networkx as nx
import community as community_louvain
from sklearn.metrics import normalized_mutual_info_score
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import time
import os
import json
from matplotlib.patches import Patch
from collections import defaultdict, Counter
from sklearn.metrics import precision_score, recall_score, f1_score
import random

size_reduction_dataset = 2500 # 1134890 = max number of vertex
random_state = 42

#compute cohesiveness_and_separateness for louvain and leiden
def compute_cohesiveness_and_separateness(G, partition):
    cohesiveness_scores = []
    separateness_scores = []

    for community in partition:
        subgraph = G.subgraph(community)
        internal_edges = subgraph.number_of_edges()
        cut_edges = sum(1 for node in community for neighbor in G.neighbors(node)
                        if neighbor not in community) / 2  # / by 2 edges calculated 2x

        cohesiveness = internal_edges / len(community) if len(community) > 0 else 0
        separateness = internal_edges / (internal_edges + cut_edges) if (internal_edges + cut_edges) > 0 else 0

        cohesiveness_scores.append(cohesiveness)
        separateness_scores.append(separateness)

    return np.mean(cohesiveness_scores), np.mean(separateness_scores)

#compute cohesiveness_and_separateness for louvain and leiden (same fonction but for  igraph IA generated)
def compute_cohesiveness_and_separateness_igraph(G, partition):
    cohesiveness_scores = []
    separateness_scores = []

    for community in partition:
        community_set = set(community)
        internal_edges = 0
        cut_edges = 0

        for node in community:
            neighbors = G.neighbors(node)
            for neighbor in neighbors:
                if neighbor in community_set:
                    internal_edges += 1
                else:
                    cut_edges += 1

        internal_edges = internal_edges // 2  # chaque arête interne est comptée deux fois
        cohesiveness = internal_edges / len(community) if len(community) > 0 else 0
        total_edges = internal_edges
        separateness = internal_edges / total_edges if total_edges > 0 else 0

        cohesiveness_scores.append(cohesiveness)
        separateness_scores.append(separateness)

    return np.mean(cohesiveness_scores), np.mean(separateness_scores)
# IA proposal to resolve transform dic to sets
def dict_to_partition_communities(partition_dict):
    community_dict = defaultdict(set)
    for node, community_id in partition_dict.items():
        community_dict[community_id].add(node)
    return list(community_dict.values())

# Load the YouTube graph
G = nx.read_edgelist('datasets/com-youtube.ungraph.txt', nodetype=int)

#use pseudo random vertex
random.seed(random_state)
selected_nodes = random.sample(list(G.nodes), size_reduction_dataset)

#create a sub graph with the selecte vertex
G_sub = G.subgraph(selected_nodes).copy()

# Load ground truth communities
ground_truth = {}
with open('datasets/com-youtube.all.cmty.txt', 'r') as f:
    for line in f:
        parts = line.strip().split()
        community_id = int(parts[0])
        for node in parts[1:]:
            ground_truth[int(node)] = community_id

# Optional: limit graph to N nodes for visualization
#N = 100  # You can change this value to process more nodes (e.g., N = len(G.nodes) for full graph)
#selected_nodes = list(G.nodes)[:N]
#G_sub = G.subgraph(selected_nodes).copy()

# ---------------- Louvain algorithm -------------------------------------
start_time = time.time()
partition = community_louvain.best_partition(G_sub)
exec_time = time.time() - start_time
modularity = community_louvain.modularity(partition, G_sub)

# Prepare labels for NMI
true_labels = []
predicted_labels = []
for node in G_sub.nodes:
    if int(node) in ground_truth:
        true_labels.append(ground_truth[int(node)])
    else:
        true_labels.append(None)
    predicted_labels.append(partition[node])

valid_true_labels = [label for label in true_labels if label is not None]
valid_predicted_labels = [pred for label, pred in zip(true_labels, predicted_labels) if label is not None]
nmi = normalized_mutual_info_score(valid_true_labels, valid_predicted_labels) if valid_true_labels else float('nan')

print(f"Execution time: {exec_time:.4f} seconds")
print(f"Modularity: {modularity:.4f}")
print(f"NMI: {nmi:.4f}")

partition_as_sets = dict_to_partition_communities(partition)
cohesiveness_louvain, separateness_louvain = compute_cohesiveness_and_separateness(G, partition_as_sets)

# Optional: match detected Louvain communities with ground truth communities
# This helps interpret what each Louvain community might represent

# Invert ground_truth: ground truth ID -> set of nodes
gt_coms = defaultdict(set)
for node, cid in ground_truth.items():
    gt_coms[cid].add(str(node))  # Cast to str to match G_sub nodes

# Group nodes by Louvain community
detected_coms = defaultdict(set)
for node, cid in partition.items():
    detected_coms[cid].add(node)

# Map Louvain communities to ground truth communities by highest overlap
louvain_to_gt = {}
print("\n--- Community Matching (Optional) ---")
for lid, louvain_nodes in detected_coms.items():
    best_match = None
    max_overlap = 0
    for gt_id, gt_nodes in gt_coms.items():
        overlap = len(louvain_nodes & gt_nodes)
        if overlap > max_overlap:
            max_overlap = overlap
            best_match = gt_id
    if best_match is not None:
        percent = max_overlap / len(louvain_nodes) * 100
        louvain_to_gt[lid] = best_match
        print(f"Louvain community {lid} ({len(louvain_nodes)} nodes) best matches GT community {best_match} "
              f"({max_overlap} overlapping nodes, {percent:.1f}%)")

# Optional: provide symbolic names to some known ground truth communities
# This is optional and for interpretability only. You can extend this mapping manually.
gt_names = {
    13: "Gaming",
    2: "Music",
    42: "Comedy",
    8: "Education",
    27: "Sports"
    # Add more if needed
}

# Visualization
''' #desactivted for performace 
print("\nGenerating graph visualization...")

# Assign colors
num_coms = len(set(partition.values()))
palette = sns.color_palette("hls", num_coms)
color_map = [palette[partition[node]] for node in G_sub.nodes]

# Generate layout
pos = nx.spring_layout(G_sub, seed=42)
fig, ax = plt.subplots(figsize=(10, 10))

# Draw edges
for u, v in G_sub.edges():
    x = [pos[u][0], pos[v][0]]
    y = [pos[u][1], pos[v][1]]
    ax.plot(x, y, color="gray", linewidth=0.5, zorder=1)

# Draw nodes
for i, node in enumerate(G_sub.nodes()):
    x, y = pos[node]
    ax.scatter(x, y, color=color_map[i], s=200, zorder=2)
    ax.text(x, y, str(node), fontsize=8, ha='center', va='center', color='white', zorder=3)

# Legend with GT interpretation (if available)
legend_elements = []
for i in set(partition.values()):
    label = f"Community {i}"
    if i in louvain_to_gt:
        gt_id = louvain_to_gt[i]
        label += f" → GT {gt_id}"
        if gt_id in gt_names:
            label += f" ({gt_names[gt_id]})"
    legend_elements.append(Patch(facecolor=palette[i], edgecolor='black', label=label))

ax.legend(handles=legend_elements, loc='lower right', title='Communities (Louvaine)', fontsize=8)

ax.axis('off')
plt.tight_layout()
plt.savefig("results/img/youtube_graph_louvaine.png", dpi=300)
plt.show()
'''
# Prepare community sizes (number of nodes per detected community)
community_sizes = list(Counter(partition.values()).values())

# Filter ground truth and predicted labels to only nodes that have ground truth
true_labels = []
predicted_labels = []
for node in G_sub.nodes:
    if int(node) in ground_truth:
        true_labels.append(ground_truth[int(node)])
        predicted_labels.append(partition[node])

# Compute scores if we have valid ground truth labels
if true_labels:
    precision = precision_score(true_labels, predicted_labels, average='macro')
    recall = recall_score(true_labels, predicted_labels, average='macro')
    f1 = f1_score(true_labels, predicted_labels, average='macro')
else:
    precision = recall = f1 = float('nan')

# Prepare the result dictionary
results = {
    "execution_time": exec_time,
    "cohesiveness": cohesiveness_louvain,
    "separateness": separateness_louvain,
    "modularity": modularity,
    "nmi": nmi,
    "community_sizes": community_sizes,
    "partition": partition,
    "scores": {
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1
    }
}

# Create results directory if it doesn't exist
os.makedirs("results", exist_ok=True)

# Save the result to a unified JSON file
with open("results/youtube_network_results_louvain_lib.json", "w") as f:

    json.dump(results, f, indent=4)

# --------------------------- Leiden algorithm section ---------------------------

import igraph as ig
import leidenalg

# Convert NetworkX graph to igraph
print("\nRunning Leiden algorithm...")

G_igraph = ig.Graph(directed=False)
G_igraph.add_vertices(list(G_sub.nodes()))

node_list = list(G_sub.nodes())
#Mapping node to index
node_to_index = {node: idx for idx, node in enumerate(node_list)}
#link 
edges = [(node_to_index[u], node_to_index[v]) for u, v in G_sub.edges()]
G_igraph.add_edges(edges)

partition_leiden = leidenalg.find_partition(G_igraph, leidenalg.ModularityVertexPartition)

# Apply Leiden algorithm
start_time_leiden = time.time()
cohesiveness_leiden, separateness_leiden = compute_cohesiveness_and_separateness_igraph(G_igraph, partition_leiden)
exec_time_leiden = time.time() - start_time_leiden
modularity_leiden = partition_leiden.modularity

# Build partition mapping: node -> community ID
partition_leiden_dict = {}
for comm_id, community in enumerate(partition_leiden):
    for node in community:
        partition_leiden_dict[G_igraph.vs[node]["name"]] = comm_id

# Prepare labels for NMI
true_labels_leiden = []
predicted_labels_leiden = []
for node in G_sub.nodes:
    if int(node) in ground_truth:
        true_labels_leiden.append(ground_truth[int(node)])
    else:
        true_labels_leiden.append(None)
    predicted_labels_leiden.append(partition_leiden_dict[node])

valid_true_labels_leiden = [label for label in true_labels_leiden if label is not None]
valid_predicted_labels_leiden = [pred for label, pred in zip(true_labels_leiden, predicted_labels_leiden) if label is not None]
nmi_leiden = normalized_mutual_info_score(valid_true_labels_leiden, valid_predicted_labels_leiden) if valid_true_labels_leiden else float('nan')

print(f"Execution time (Leiden): {exec_time_leiden:.4f} seconds")
print(f"Modularity (Leiden): {modularity_leiden:.4f}")
print(f"NMI (Leiden): {nmi_leiden:.4f}")

partition_leiden = leidenalg.find_partition(G_igraph, leidenalg.ModularityVertexPartition)

# Match Leiden communities to ground truth
gt_coms_leiden = defaultdict(set)
for node, cid in ground_truth.items():
    gt_coms_leiden[cid].add(str(node))

detected_coms_leiden = defaultdict(set)
for node, cid in partition_leiden_dict.items():
    detected_coms_leiden[cid].add(node)

leiden_to_gt = {}
print("\n--- Community Matching (Leiden) ---")
for lid, leiden_nodes in detected_coms_leiden.items():
    best_match = None
    max_overlap = 0
    for gt_id, gt_nodes in gt_coms_leiden.items():
        overlap = len(leiden_nodes & gt_nodes)
        if overlap > max_overlap:
            max_overlap = overlap
            best_match = gt_id
    if best_match is not None:
        percent = max_overlap / len(leiden_nodes) * 100
        leiden_to_gt[lid] = best_match
        print(f"Leiden community {lid} ({len(leiden_nodes)} nodes) best matches GT community {best_match} "
              f"({max_overlap} overlapping nodes, {percent:.1f}%)")

# Community sizes for Leiden
community_sizes_leiden = list(Counter(partition_leiden_dict.values()).values())

# Precision, Recall, F1 for Leiden
true_labels_leiden = []
predicted_labels_leiden = []
for node in G_sub.nodes:
    if int(node) in ground_truth:
        true_labels_leiden.append(ground_truth[int(node)])
        predicted_labels_leiden.append(partition_leiden_dict[node])

if true_labels_leiden:
    precision_leiden = precision_score(true_labels_leiden, predicted_labels_leiden, average='macro')
    recall_leiden = recall_score(true_labels_leiden, predicted_labels_leiden, average='macro')
    f1_leiden = f1_score(true_labels_leiden, predicted_labels_leiden, average='macro')
else:
    precision_leiden = recall_leiden = f1_leiden = float('nan')

# Prepare result dictionary for Leiden
results_leiden = {
    "execution_time": exec_time_leiden,
    "cohesiveness": cohesiveness_leiden,
    "separateness": separateness_leiden,
    "modularity": modularity_leiden,
    "nmi": nmi_leiden,
    "community_sizes": community_sizes_leiden,
    "partition": partition_leiden_dict,
    "scores": {
        "Precision": precision_leiden,
        "Recall": recall_leiden,
        "F1-score": f1_leiden
    }}

# Save Leiden result to JSON
with open("results/youtube_network_results_leiden__lib.json", "w") as f:

    json.dump(results_leiden, f, indent=4)

# ---- Visualization for Leiden ----
''' #desactiveted for performace
print("\nGenerating graph visualization for Leiden...")

# Assign colors
num_coms_leiden = len(set(partition_leiden_dict.values()))
palette_leiden = sns.color_palette("Set2", num_coms_leiden)
color_map_leiden = [palette_leiden[partition_leiden_dict[node]] for node in G_sub.nodes]

# Generate layout (reuse pos from earlier to align both visualizations)
fig, ax = plt.subplots(figsize=(10, 10))

# Draw edges
for u, v in G_sub.edges():
    x = [pos[u][0], pos[v][0]]
    y = [pos[u][1], pos[v][1]]
    ax.plot(x, y, color="gray", linewidth=0.5, zorder=1)

# Draw nodes
for i, node in enumerate(G_sub.nodes()):
    x, y = pos[node]
    ax.scatter(x, y, color=color_map_leiden[i], s=200, zorder=2)
    ax.text(x, y, str(node), fontsize=8, ha='center', va='center', color='white', zorder=3)

# Legend with GT interpretation (if available)
unique_coms = sorted(set(partition_leiden_dict.values()))
com_to_color_idx = {com: idx for idx, com in enumerate(unique_coms)}

legend_elements_leiden = []
for i in unique_coms:
    label = f"Community {i}"
    if i in leiden_to_gt:
        gt_id = leiden_to_gt[i]
        label += f" → GT {gt_id}"
        if gt_id in gt_names:
            label += f" ({gt_names[gt_id]})"
    color_idx = com_to_color_idx[i]
    legend_elements_leiden.append(Patch(facecolor=palette_leiden[color_idx], edgecolor='black', label=label))


ax.legend(handles=legend_elements_leiden, loc='lower right', title='Communities (Leiden)', fontsize=8)

ax.axis('off')
plt.tight_layout()
plt.savefig("results/img/youtube_graph_leiden.png", dpi=300)
plt.show()
'''