import networkx as nx
import pandas as pd
from louvain_try_1 import Louvain as Lvn1
from louvain_try_2 import Louvain as Lvn2
from louvain import Louvain as Lvn
from leiden import Leiden as Ldn
from neo4j_export import Neo4jGraph, Neo4jGraphExporter
from preprocessing import GraphPreprocessor
import evaluation
import time
import sys
import os

# timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
# log_dir = os.path.join('logs', timestamp)
# os.makedirs(log_dir, exist_ok=True)
# log_file_path = os.path.join(log_dir, 'output.txt')
# sys.stdout = open(log_file_path, 'w')
# sys.stderr = sys.stdout

def load_dataset_and_run():
    # Example usage
    # G = nx.karate_club_graph()
    G = nx.les_miserables_graph()
    # G = nx.florentine_families_graph()
    # G = nx.erdos_renyi_graph(n=50, p=0.1)
    # G = nx.sudoku_graph(n=3)

    # example that should work with leiden and fail with louvain
    # df = pd.read_csv('test_datasets/problematic_graph_2.txt', header=None, sep=" ", names=['source', 'target'])
    # G = nx.from_pandas_edgelist(df, "source", "target")

    # example = pd.read_csv("test_datasets/example.txt", sep=" ", names=["start_node", "end_node"])
    # df = pd.read_csv('test_datasets/example.txt', header=None, sep=" ", names=['source', 'target'])
    # G = nx.from_pandas_edgelist(df, "source", "target")

    processed_graph = GraphPreprocessor(G).process()

    # louvain1 = Lvn1(processed_graph)
    # louvain2 = Lvn2(processed_graph)
    louvain = Lvn(processed_graph)
    # louvain1.run("Louvain_ver_1")
    # louvain2.run("Louvain_ver_2")
    _, louvain_G, louvain_partition, louvain_final_partition = louvain.run()

    leiden = Ldn(processed_graph)
    _, leiden_G, leiden_partition, leiden_final_partition = leiden.run()

    evaluation.evaluate_communities_without_ground_truth(louvain_G, louvain_final_partition, "Louvain")
    evaluation.evaluate_communities_without_ground_truth(leiden_G, leiden_partition, "Leiden")

    # For smaller graphs with fewer communities, use a lower gamma value
    evaluation.evaluate_cpm(G, louvain_final_partition, gamma=0.2, method="Louvain")
    evaluation.evaluate_cpm(G, leiden_final_partition, gamma=0.2, method="Leiden")

if __name__ == "__main__":
    load_dataset_and_run()