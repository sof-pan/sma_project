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

timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
log_dir = os.path.join('logs', f"brain_network_{timestamp}")
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, 'output.txt')
sys.stdout = open(log_file_path, 'w')
sys.stderr = sys.stdout

parser = argparse.ArgumentParser(description="Run community detection on Email EU Core dataset.")
parser.add_argument('--export_graphs', action='store_true', help='If set, export the graphs to Neo4j, otherwise do not export.')
args = parser.parse_args()

print("Start time:", time.strftime("%Y-%m-%d %H:%M:%S"))
start_time = time.time()

# Load the graph structure from edge list
brain_df = pd.read_csv('datasets/bn-fly-drosophila_medulla_1.edges', sep=" ", comment='#', header=None, names=['source', 'target', 'weight'])
brain_graph: nx.Graph = nx.from_pandas_edgelist(brain_df, 'source', 'target', edge_attr='weight')

# Add weight to edges if it's not already there
for node1, node2 in brain_graph.edges():
    if 'weight' not in brain_graph[node1][node2] or math.isnan(brain_graph[node1][node2]['weight']):
        brain_graph[node1][node2]['weight'] = 1

if args.export_graphs:
    neo4j = Neo4jGraph()
    neo4j.clear()
    neo4j.close()

    original_graph_exporter = Neo4jBrainGraphExporter()
    original_graph_exporter.export_graph(brain_graph)
    original_graph_exporter.close()

processor = GraphPreprocessor(
    G=brain_graph,
    sample_fraction=0.9
)

brain_graph_processed = processor.process()

louvain = Lvn(brain_graph_processed)
louvain_original_G, louvain_G, louvain_partition, louvain_final_partition = louvain.run(print_results=False)

leiden = Ldn(brain_graph_processed)
leiden_original_G, leiden_G, leiden_partition, leiden_final_partition = leiden.run(print_results=False)

evaluation.evaluate_communities_without_ground_truth(louvain_G, louvain_final_partition, "Louvain")
evaluation.evaluate_communities_without_ground_truth(leiden_G, leiden_partition, "Leiden")

evaluation.evaluate_cpm(brain_graph_processed, louvain_final_partition, "Louvain")
evaluation.evaluate_cpm(brain_graph_processed, leiden_final_partition, "Leiden")

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
print("=======================================================")
print(f"Total time for Brain Network: {hours} hours, {minutes} minutes, {seconds:.4f} seconds")
print("=======================================================")

sys.stdout.close()
sys.stdout = sys.__stdout__