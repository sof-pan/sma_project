from sklearn.metrics import normalized_mutual_info_score, pairwise_distances
import networkx as nx
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# AI Generated Methods for Evaluating Community Detection

# We expect high NMI score for good community detection
# If NMI is low, the predicted communities are not similar to the ground truth communities
# NMI = 1 → Perfect match between predicted and ground-truth communities.
# NMI = 0 → Completely independent / unrelated community assignments.
# NMI near 0.5 → Some overlap, but not a strong match.
def evaluate_communities_with_ground_truth(pred_partition: dict, ground_truth: dict, method):
    # create a sorted list of all nodes present in the ground truth dictionary
    # keep it sorted as both dicts should have the same node order
    nodes = sorted(ground_truth.keys())

    # create empty numpy arrays to hold the true and predicted labels
    # they are created with the same length as the number of nodes
    nodes_len = len(nodes)
    true_labels = np.empty(nodes_len, dtype=int)
    pred_labels = np.empty(nodes_len, dtype=int)

    # iterate over the nodes
    for i, node in enumerate(nodes):
        # assign the true community label for the current node from ground truth
        true_labels[i] = ground_truth[node]
        # assign the predicted community label if available, else assign -1
        pred_labels[i] = pred_partition.get(node, -1)

    # compute NMI score between true and predicted labels
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    print(f"Normalized Mutual Information (NMI) for {method}: {nmi:.4f}")
    return nmi

# We expect low cohesiveness and high separateness for good community detection
# If cohesiveness is high, nodes within communities are not close to each other feature-wise.
# If separateness is low, communities overlap heavily in feature space.
def evaluate_communities_without_ground_truth(G: nx.Graph, partition, method):
    # Ensure nodes have 'feature'; if not, add degree as feature
    for node in G.nodes():
        if node not in partition:
            # skip nodes not in partition or assign default community
            continue
        if 'feature' not in G.nodes[node]:
            G.nodes[node]['feature'] = np.array([G.degree(node)])
    # convert node features to a matrix
    X = []
    for n in G.nodes():
        X.append(G.nodes[n]['feature'])
    X = np.array(X)

    # Normalize features to [0,1] range
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    
    # get labels for each node from partition
    labels = []
    for n in G.nodes():
        labels.append(partition[n])
    
    # compute unique community IDs
    communities = set(labels)
    
    # compute centroids of each community
    centroids = []
    for c in communities:
        members_idx = []
        for idx, label in enumerate(labels):
            if label == c:
                members_idx.append(idx)
        
        members = X[members_idx, :]
        centroid = members.mean(axis=0)
        centroids.append(centroid)
    centroids = np.array(centroids)
    
    # global centroid
    global_centroid = X.mean(axis=0).reshape(1, -1)
    
    # compute cohesiveness: avg distance of nodes to their community centroid
    cohesiveness_sum = 0.0
    for idx_c, c in enumerate(communities):
        members_idx = []
        for idx, label in enumerate(labels):
            if label == c:
                members_idx.append(idx)
        
        members = X[members_idx, :]
        centroid = centroids[idx_c].reshape(1, -1)
        distances = pairwise_distances(members, centroid)
        cohesiveness_sum += distances.mean()
    cohesiveness = cohesiveness_sum / len(communities)
    
    # compute separateness: avg distance of centroids to global centroid
    separateness_distances = pairwise_distances(centroids, global_centroid)
    separateness = separateness_distances.mean()

    print(f"Cohesiveness for {method}: {cohesiveness:.4f}")
    print(f"Separateness for {method}: {separateness:.4f}")
    
    return cohesiveness, separateness

# For smaller graphs with fewer communities, use a lower gamma value
def evaluate_cpm(G: nx.Graph, partition: dict, gamma: float = 0.5, method: str = "Unknown"):
    # Keep track of total CPM score
    Q = 0.0

    # Group nodes by community
    community_dict = {}
    for node, comm in partition.items():
        community_dict.setdefault(comm, []).append(node)

    for nodes in community_dict.values():
        subgraph = G.subgraph(nodes)
        internal_weight = sum(data.get('weight', 1) for _, _, data in subgraph.edges(data=True))
        n = len(nodes)
        Q += internal_weight - gamma * n * (n - 1) / 2  # num possible internal edges in undirected graph

    print(f"CPM Score (gamma={gamma}) for {method}: {Q:.4f}")
    return Q