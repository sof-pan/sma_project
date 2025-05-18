import networkx as nx
import gzip
from preprocessing import GraphPreprocessor
import community as community_louvain
from neo4j import GraphDatabase
from collections import defaultdict
import igraph as ig
import leidenalg
import math
import time
import evaluation
import sys
import os

timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
log_dir = os.path.join('logs', f"email_network_external_{timestamp}")
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, 'output.txt')
sys.stdout = open(log_file_path, 'w')
sys.stderr = sys.stdout

start_time = time.time()

# Load the graph structure from edge list
email_eu_graph_initial: nx.Graph = nx.read_edgelist('datasets/email-Eu-core.txt.gz', comments='#', nodetype=int)
email_eu_graph: nx. Graph = email_eu_graph_initial.to_undirected()

# Load department labels
node_labels = {}
with gzip.open('datasets/email-Eu-core-department-labels.txt.gz', 'rt', encoding='utf-8') as f:
    for line in f:
        node, label = map(int, line.strip().split())
        node_labels[node] = label

# Add department as node attribute
nx.set_node_attributes(email_eu_graph, node_labels, 'department')

# Add weight to edges if it's not already there
for node1, node2 in email_eu_graph.edges():
    if 'weight' not in email_eu_graph[node1][node2] or math.isnan(email_eu_graph[node1][node2]['weight']):
        email_eu_graph[node1][node2]['weight'] = 1

processor = GraphPreprocessor(
    G=email_eu_graph,
    node_attributes=['department'],
    edge_attributes=['weight'],
    selected_node_attributes=['department', 'degree_centrality', 'betweenness_centrality'],
    selected_edge_attributes=['weight'],
    sample_fraction=0.9
)

email_eu_graph_processed = processor.process()

# LOUVAIN
start_time_louv = time.time()
partition_louvain = community_louvain.best_partition(email_eu_graph_processed)
end_time_louv = time.time()
elapsed_louv = end_time_louv - start_time_louv
print("=======================================================")
print(f"Brain Network Louvain Time: {int(elapsed_louv // 3600)}h {int((elapsed_louv % 3600) // 60)}m {elapsed_louv % 60:.2f}s")
print("=======================================================")
community_nodes_louvain = defaultdict(list)
for node, comm_id in partition_louvain.items():
    community_nodes_louvain[comm_id].append(node)

# Add Louvain attributes to nodes
for node, comm_id in partition_louvain.items():
    members_str = ", ".join(map(str, sorted(community_nodes_louvain[comm_id])))
    email_eu_graph_processed.nodes[node]['louvain_community'] = comm_id
    email_eu_graph_processed.nodes[node]['louvain_community_members'] = members_str


# LEIDEN
start_time_leid = time.time()
G_nx = nx.convert_node_labels_to_integers(email_eu_graph_processed, label_attribute="original_id")
igraph_G = ig.Graph.TupleList(G_nx.edges(), directed=False)
igraph_G.vs['original_id'] = [data['original_id'] for _, data in G_nx.nodes(data=True)]

# Map: NetworkX node -> iGraph vertex index
nx_to_ig = {node_id: idx for idx, node_id in enumerate(igraph_G.vs['original_id'])}
partition_leiden = leidenalg.find_partition(igraph_G, leidenalg.ModularityVertexPartition)
end_time_leid = time.time()
elapsed_leid = end_time_leid - start_time_leid
print("=======================================================")
print(f"Brain Network Leiden Time: {int(elapsed_leid // 3600)}h {int((elapsed_leid % 3600) // 60)}m {elapsed_leid % 60:.2f}s")
print("=======================================================")

ig_to_nx = {idx: node_id for node_id, idx in nx_to_ig.items()}
community_nodes = defaultdict(list)
for comm_id, community in enumerate(partition_leiden):
    for ig_node_idx in community:
        original_nx_node = ig_to_nx[ig_node_idx]
        community_nodes[comm_id].append(original_nx_node)

pred_partition_dict = {}
for comm_id, nodes in community_nodes.items():
    for node in nodes:
        pred_partition_dict[node] = comm_id

# Add community and community_members string to graph nodes
for node, comm_id in pred_partition_dict.items():
    members_str = ", ".join(map(str, sorted(community_nodes[comm_id])))
    email_eu_graph_processed.nodes[node]['leiden_community'] = comm_id
    email_eu_graph_processed.nodes[node]['leiden_community_members'] = members_str

evaluation.evaluate_communities_with_ground_truth(partition_louvain, node_labels, "Louvain")
evaluation.evaluate_communities_with_ground_truth(pred_partition_dict, node_labels, "Leiden")

# Configure Neo4j connection
uri = "bolt://localhost:7687"
user = "neo4j"
password = "test1234"

driver = GraphDatabase.driver(uri, auth=(user, password))

with driver.session() as session:
    session.run("MATCH (n) DETACH DELETE n")

    # Export Louvain nodes
    for node, data in email_eu_graph_processed.nodes(data=True):
        session.run(
            """
            CREATE (n:PersonLouvain {
                id: $id,
                department: $department,
                degree_centrality: $dc,
                betweenness_centrality: $bc,
                community: $community,
                community_members: $community_members
            })
            """,
            {
                "id": node,
                "department": data.get("department"),
                "dc": data.get("degree_centrality"),
                "bc": data.get("betweenness_centrality"),
                "community": data.get("louvain_community"),
                "community_members": data.get("louvain_community_members")
            }
        )

    # Export Leiden nodes
    for node, data in email_eu_graph_processed.nodes(data=True):
        session.run(
            """
            CREATE (n:PersonLeiden {
                id: $id,
                department: $department,
                degree_centrality: $dc,
                betweenness_centrality: $bc,
                community: $community,
                community_members: $community_members
            })
            """,
            {
                "id": node,
                "department": data.get("department"),
                "dc": data.get("degree_centrality"),
                "bc": data.get("betweenness_centrality"),
                "community": data.get("leiden_community"),
                "community_members": data.get("leiden_community_members")
            }
        )

    # Export edges (shared for both)
    for u, v, data in email_eu_graph_processed.edges(data=True):
        session.run(
            """
            MATCH (aL:PersonLouvain {id: $source}), (bL:PersonLouvain {id: $target}),
                    (aLe:PersonLeiden {id: $source}), (bLe:PersonLeiden {id: $target})
            CREATE (aL)-[:EMAIL {weight: $weight}]->(bL),
                    (aLe)-[:EMAIL {weight: $weight}]->(bLe)
            """,
            {
                "source": u,
                "target": v,
                "weight": data.get("weight", 1)
            }
        )

driver.close()

end_time = time.time()
elapsed_time = end_time - start_time
hours = int(elapsed_time // 3600)
minutes = int((elapsed_time % 3600) // 60)
seconds = elapsed_time % 60
print("=======================================================")
print(f"Total time for Email EU Core Network: {hours} hours, {minutes} minutes, {seconds:.4f} seconds")
print("=======================================================")