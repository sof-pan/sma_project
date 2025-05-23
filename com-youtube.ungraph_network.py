import networkx as nx
from neo4j_export import Neo4jGraph, Neo4jBrainGraphExporter, Neo4jGraphExporter
from preprocessing import GraphPreprocessor
from louvain import Louvain as Lvn
from leiden import Leiden as Ldn
import time
import math
import pandas as pd
import evaluation
import argparse
import sys
import os
import json
from collections import Counter
import random

random.seed(42)
###(calculation with  50 000 take 11h27
#for a 
#Intel Core i5-4210M (2 physical cores, 4 threads, 2.6–3.2 GHz)
# 64-bit L3 Cache 3 MiB, 16 GO Ram
size_reduction_dataset = 1500 # take 47 min  /work only on one threads

timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
log_dir = os.path.join('logs', f"youtube_network_{timestamp}")
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, 'output.txt')
sys.stdout = open(log_file_path, 'w')
sys.stderr = sys.stdout

parser = argparse.ArgumentParser(description="Run community detection on dataset.")
parser.add_argument('--export_graphs', action='store_true', help='If set, export the graphs to Neo4j, otherwise do not export.')
args = parser.parse_args()

print("Start time:", time.strftime("%Y-%m-%d %H:%M:%S"))
start_time = time.time()

# Load the graph structure from edge list [sep =detecte automatically separator]
dataSet_df = pd.read_csv('datasets/com-youtube.ungraph.txt', sep='\s+', comment='#', header=None, names=['source', 'target', 'weight'])
dataSet_df = dataSet_df.sort_values(by=['source', 'target']).reset_index(drop=True)

# Take only a part of the graph for test proposal 
#dataSet_df = dataSet_df.sample(size_reduction_dataset, random_state=42)
dataSet_df = dataSet_df.head(size_reduction_dataset) #IA proposal to have the same random on differant machine

the_graph: nx.Graph = nx.from_pandas_edgelist(dataSet_df, 'source', 'target', edge_attr='weight')

# Add weight to edges if it's not already there
for node1, node2 in the_graph.edges():
    if 'weight' not in the_graph[node1][node2] or math.isnan(the_graph[node1][node2]['weight']):
        the_graph[node1][node2]['weight'] = 1

if args.export_graphs:
    neo4j = Neo4jGraph()
    neo4j.clear()
    neo4j.close()

    original_graph_exporter = Neo4jBrainGraphExporter()
    original_graph_exporter.export_graph(the_graph)
    original_graph_exporter.close()

processor = GraphPreprocessor(
    G=the_graph,
    sample_fraction=0.9
)

the_graph_processed = processor.process()

louvain = Lvn(the_graph_processed)
start_louvain = time.time()
louvain_original_G, louvain_G, louvain_partition, louvain_final_partition = louvain.run(print_results=False)
louvain_time = time.time() - start_louvain

leiden = Ldn(the_graph_processed)
start_leiden = time.time()
leiden_original_G, leiden_G, leiden_partition, leiden_final_partition = leiden.run(print_results=False)
leiden_time = time.time() - start_leiden

evaluation.evaluate_communities_without_ground_truth(louvain_G, louvain_final_partition, "Louvain")
evaluation.evaluate_communities_without_ground_truth(leiden_G, leiden_partition, "Leiden")

evaluation.evaluate_cpm(the_graph_processed, louvain_final_partition, method="Louvain")
evaluation.evaluate_cpm(the_graph_processed, leiden_final_partition, method="Leiden")


if args.export_graphs:
    louvain_exporter = Neo4jGraphExporter(label="LouvainNode")
    louvain_exporter.export_graph(louvain_G, louvain_original_G, community_dict=louvain_final_partition, original_nodes=louvain.original_nodes)
    louvain_exporter.close()

    leiden_exporter = Neo4jGraphExporter(label="LeidenNode")
    leiden_exporter.export_graph(leiden_G, leiden_original_G, community_dict=leiden_partition, original_nodes=leiden.original_nodes, for_louvain=False)
    leiden_exporter.close()

end_time = time.time()
elapsed_time = end_time - start_time

hours = int(elapsed_time // 3600)
minutes = int((elapsed_time % 3600) // 60)
seconds = elapsed_time % 60
louvain_elapsed_time = elapsed_time
print("=======================================================")
print(f"Total time for Brain Network: {hours} hours, {minutes} minutes, {seconds:.4f} seconds")
print("=======================================================")




# genarate a json file for louvain and one for leiden
import json
import os
from collections import Counter
from networkx.algorithms.community import modularity
from collections import defaultdict

# calculate the modularity (not from scrach)

def partition_to_communities(partition_dict):
    communities = defaultdict(set)
    for node, comm_id in partition_dict.items():
        communities[comm_id].add(node)
    return list(communities.values())

#  convert partition Louvain
louvain_communities = partition_to_communities(louvain_final_partition)
louvain_modularity = modularity(the_graph_processed, louvain_communities)

# convert  partition Leiden
leiden_communities = partition_to_communities(leiden_final_partition)
leiden_modularity = modularity(the_graph_processed, leiden_communities)

louvain_cohesiveness, louvain_separateness = evaluation.evaluate_communities_without_ground_truth(
    the_graph_processed, louvain_final_partition, "Louvain"
)
leiden_cohesiveness, leiden_separateness = evaluation.evaluate_communities_without_ground_truth(
    the_graph_processed, leiden_final_partition, "Leiden"
)

# compute the CPM
louvain_cpm = evaluation.evaluate_cpm(the_graph_processed, louvain_final_partition, method="Louvain")
leiden_cpm = evaluation.evaluate_cpm(the_graph_processed, leiden_final_partition, method="Leiden")

# dictionnary with results
louvain_results = {
    "execution_time": louvain_time,
    "cohesiveness": louvain_cohesiveness,
    "separateness": louvain_separateness,
    "modularity": louvain_modularity,
    "cpm": louvain_cpm,
    "community_sizes": sorted(Counter(louvain_final_partition.values()).values(), reverse=True)
}

leiden_results = {
    "execution_time": leiden_time,
    "cohesiveness": leiden_cohesiveness,
    "separateness": leiden_separateness,
    "modularity": leiden_modularity,
    "cpm": leiden_cpm,
    "community_sizes": sorted(Counter(leiden_final_partition.values()).values(), reverse=True)
}

# save in a JSON format
with open("results/2youtube_network_results_louvain_scratch.json", "w") as f:
    json.dump(louvain_results, f, indent=4)

with open("results/2youtube_network_results_leiden_scratch.json", "w") as f:
    json.dump(leiden_results, f, indent=4)

sys.stdout.close()
sys.stdout = sys.__stdout__