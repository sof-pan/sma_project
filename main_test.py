import networkx as nx
import pandas as pd
from louvain_try_1 import Louvain as Lvn1
from louvain_try_2 import Louvain as Lvn2
from louvain import Louvain as Lvn
from leiden import Leiden as Ldn
from neo4j_export import Neo4jGraph, Neo4jGraphExporter
from preprocessing import GraphPreprocessor
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
    # G = nx.les_miserables_graph()
    # G = nx.florentine_families_graph()
    # G = nx.erdos_renyi_graph(n=50, p=0.1)
    # G = nx.sudoku_graph(n=3)

    # example that should work with leiden and fail with louvain
    # df = pd.read_csv('test_datasets/problematic_graph_2.txt', header=None, sep=" ", names=['source', 'target'])
    # G = nx.from_pandas_edgelist(df, "source", "target")

    # example = pd.read_csv("test_datasets/example.txt", sep=" ", names=["start_node", "end_node"])
    df = pd.read_csv('test_datasets/example.txt', header=None, sep=" ", names=['source', 'target'])
    G = nx.from_pandas_edgelist(df, "source", "target")

    processed_graph = GraphPreprocessor(G).process()

    # louvain1 = Lvn1(processed_graph)
    # louvain2 = Lvn2(processed_graph)
    louvain = Lvn(processed_graph)
    # louvain1.run("Louvain_ver_1")
    # louvain2.run("Louvain_ver_2")
    louvain_original_G, louvain_G, louvain_partition, _ = louvain.run()

    leiden = Ldn(processed_graph)
    leiden_original_G, leiden_G, leiden_partition, _ = leiden.run()

    # neo4j = Neo4jGraph()
    # neo4j.clear()
    # neo4j.close()
    # louvain_exporter = Neo4jGraphExporter(label="LouvainNode")
    # louvain_exporter.export_graph(louvain_G, louvain_original_G, community_dict=louvain_partition, original_nodes=louvain.original_nodes)
    # louvain_exporter.close()

    # leiden_exporter = Neo4jGraphExporter(label="LeidenNode")
    # leiden_exporter.export_graph(leiden_G, leiden_original_G, community_dict=leiden_partition, original_nodes=leiden.original_nodes)
    # leiden_exporter.close()

    # sys.stdout.close()
    # sys.stdout = sys.__stdout__


if __name__ == "__main__":
    load_dataset_and_run()