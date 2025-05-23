import networkx as nx
import community as community_louvain
from sklearn.metrics import normalized_mutual_info_score
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import json

#################################################################
#
##          V2.01 For data we use an .json file  
#           #####################################################
#################################################################


# load data from json (génerated by com.youtube.ungraph_network_lib.py or com.youtube.ungraph_network_lib_scratch.py)
def load_results(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# read louvain_library_results json 
louvain_library_data = load_results("results/1500_youtube_network_results_louvain_lib.json")
louvain_scratch_data = load_results("results/1500_youtube_network_results_louvain_scratch.json")
# read leiden_library_results json 
leiden_library_data = load_results("results/1500_youtube_network_results_leiden_lib.json")
leiden_scratch_data = load_results("results/1500_youtube_network_results_leiden_scratch.json")




#####################################################
#  
#           Simulated data to remplace with real later 
#           ###########################################
#
# Simulated value for time and modularity (graph 1)

#louvain_library_data
exec_timeLouvaineLib = louvain_library_data["execution_time"]
modularityLouvaineLib = louvain_library_data["modularity"]
nmiLouvaineLib = louvain_library_data["nmi"]
community_sizesLouvaineLib = louvain_library_data["community_sizes"]
partitionLouvaineLib = louvain_library_data["partition"]
scoresLouvaineLib = louvain_library_data["scores"]

precisionLouvaineLib = scoresLouvaineLib["Precision"]
recallLouvaineLib = scoresLouvaineLib["Recall"]
f1LouvaineLib = scoresLouvaineLib["F1-score"]

#louvain_scratch_data
exec_timeLouvaineScratch = louvain_scratch_data["execution_time"]
modularityLouvaineScratch = louvain_scratch_data["modularity"]
community_sizesLouvaineScratch = louvain_scratch_data["community_sizes"]
#partitionLouvaineScratch = louvain_scratch_data["partition"]

#leiden_library_data
exec_timeLeidenLib = leiden_library_data["execution_time"]
modularityLeidenLib = leiden_library_data["modularity"]
nmiLeidenLib = leiden_library_data["nmi"]
community_sizesLeidenLib = leiden_library_data["community_sizes"]
partitionLeidenLib = leiden_library_data["partition"]
scoresLeidenLib = leiden_library_data["scores"]

precisionLeidenLib = scoresLeidenLib["Precision"]
recallLeidenLib = scoresLeidenLib["Recall"]
f1LeidenLib = scoresLeidenLib["F1-score"]

#leiden_scratch_data
exec_timeLeidenScratch = leiden_scratch_data["execution_time"]
modularityLeidenScratch = leiden_scratch_data["modularity"]
community_sizesLeidenScratch = leiden_scratch_data["community_sizes"]
#partitionLeidenScratch = leiden_scratch_data["partition"]


# Simulated community assignments (graph 3)
def partition_dict_to_list(partition_dict):
    return [partition_dict[node] for node in sorted(partition_dict)]

modularity_time = [
    {"Algorithm": "Louvain", "Version": "Library", "Execution Time (s)": exec_timeLeidenLib, "Modularity": modularityLeidenLib},
    {"Algorithm": "Louvain", "Version": "From Scratch", "Execution Time (s)": exec_timeLouvaineLib, "Modularity": modularityLouvaineScratch},
    {"Algorithm": "Leiden", "Version": "Library", "Execution Time (s)": exec_timeLeidenLib, "Modularity": modularityLeidenLib},
    {"Algorithm": "Leiden", "Version": "From Scratch", "Execution Time (s)": exec_timeLeidenScratch, "Modularity": modularityLeidenScratch},
]

#print (modularity_time)

# --- Simulated community size data (graph 2)
community_sizes = {
    ("Louvain", "Library"): louvain_library_data["community_sizes"],
    ("Louvain", "From Scratch"): louvain_scratch_data["community_sizes"],
    ("Leiden", "Library"): leiden_library_data["community_sizes"],
    ("Leiden", "From Scratch"): leiden_scratch_data["community_sizes"]
}

# transform louvaine library to a listn
louvain_partition_list = partition_dict_to_list(partitionLouvaineLib)
# transform leiden library to a listn
leiden_partition_list = partition_dict_to_list(partitionLeidenLib)
 

partitions = {
    "Louvain Library":        louvain_partition_list,
   # "Louvain From Scratch":   louvain_partition_list, #to remplace but we dont have a partition for scratch version now 
    "Leiden Library":         leiden_partition_list,
   # "Leiden From Scratch":    louvain_partition_list #to remplace #to remplace but we dont have a partition for scratch version now
}
#
# 
# Data for the F-score, Precision, and Recall (graph4)
PrecisionFscoreRec = [
    {"Algorithm": "Louvain", "Version": "Library", "Metric": "Precision", "Score": precisionLouvaineLib},
    {"Algorithm": "Louvain", "Version": "Library", "Metric": "Recall", "Score": recallLouvaineLib},
    {"Algorithm": "Louvain", "Version": "Library", "Metric": "F1-score", "Score": f1LouvaineLib},
    
    {"Algorithm": "Louvain", "Version": "From Scratch", "Metric": "Precision", "Score": 0.833},
    {"Algorithm": "Louvain", "Version": "From Scratch", "Metric": "Recall", "Score": 0.889},
    {"Algorithm": "Louvain", "Version": "From Scratch", "Metric": "F1-score", "Score": 0.822},
    
    {"Algorithm": "Leiden", "Version": "Library", "Metric": "Precision", "Score": precisionLeidenLib},
    {"Algorithm": "Leiden", "Version": "Library", "Metric": "Recall", "Score": recallLeidenLib},
    {"Algorithm": "Leiden", "Version": "Library", "Metric": "F1-score", "Score": f1LeidenLib},
    
    {"Algorithm": "Leiden", "Version": "From Scratch", "Metric": "Precision", "Score": 0.778},
    {"Algorithm": "Leiden", "Version": "From Scratch", "Metric": "Recall", "Score": 0.778},
    {"Algorithm": "Leiden", "Version": "From Scratch", "Metric": "F1-score", "Score": 0.667},
]


custom_palette = {
    "Louvain": "#af7ac5",
    "Leiden": "#cd6155"
}

def plot_execution_time_and_modularity(df):
    fig, ax1 = plt.subplots(figsize=(10, 10))
    sns.barplot(x="Version", y="Execution Time (s)", hue="Algorithm", data=df, ax=ax1, palette=custom_palette)
    ax1.set_ylabel("Execution Time (s)", color='#2980b9')
    ax1.tick_params(axis='y', labelcolor='#2980b9')

    for i in range(len(df)):
        bar = ax1.patches[i]
        y_position = bar.get_height()
        center_x = bar.get_x() + bar.get_width() / 2
        ax1.text(center_x, y_position + 2, f"{df['Execution Time (s)'][i]:.4f}s", ha='center', color='#2980b9', fontsize=10)
        ax1.text(center_x, y_position + 4, f"{df['Modularity'][i]:.4f}", ha='center', color='#ff7f0e', fontsize=10)

    ax2 = ax1.twinx()
    sns.pointplot(x="Version", y="Modularity", hue="Algorithm", data=df, ax=ax2, dodge=0.5,
                  linestyle='None', markers="D", palette='dark:black', legend=False)
    ax2.set_ylabel("Modularity", color='#ff7f0e')
    ax2.tick_params(axis='y', labelcolor='#ff7f0e')

    plt.title("Execution Time (s) and Modularity by Version and Algorithm")
    ax1.legend(title="Algorithm", loc='upper left')
    plt.tight_layout()
    plt.savefig("results/img/execution Time (s) and Modularity.png", dpi=300)
    plt.show()


def plot_community_size_distribution(community_sizes):
    records = []
    for (algorithm, version), sizes in community_sizes.items():
        for size in sizes:
            records.append({"Algorithm": algorithm, "Version": version, "Community Size": size})
    size_df = pd.DataFrame(records)

    plt.figure(figsize=(10, 6))
    ax = sns.stripplot(x="Version", y="Community Size", hue="Algorithm", data=size_df, palette=custom_palette)
    for artist in ax.findobj():
        if isinstance(artist, plt.Line2D):
            artist.set_linewidth(2)

    plt.title("Distribution of Community Sizes by Version and Algorithm")
    plt.ylabel("Community Size")
    plt.xlabel("Version")
    plt.legend(title="Algorithm", loc="upper right")
    plt.tight_layout()
    #plt.text(0, 20, f'Louvain Library: μ={recallLouvaineLib:.1f}', bbox=dict(boxstyle="round"))
    plt.savefig("results/img/distribution of Community Sizes.png", dpi=300)
    plt.show()


def plot_partition_similarity(partitions):
    labels = list(partitions.keys())
    n = len(labels)
    similarity_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            similarity_matrix[i][j] = normalized_mutual_info_score(partitions[labels[i]], partitions[labels[j]])

    similarity_df = pd.DataFrame(similarity_matrix, index=labels, columns=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(similarity_df, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={'label': 'NMI Similarity'})
    plt.title("Community Partition Similarity (NMI)")
    plt.tight_layout()
    plt.savefig("results/img/community Partition Similarity.png", dpi=300)
    plt.show()


def plot_fscore_comparison(metrics_data):
    df = pd.DataFrame(metrics_data)
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=df, x="Metric", y="Score", hue="Algorithm", palette=custom_palette, ci=None, dodge=True)

    for bar in ax.patches:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.015, f"{height:.3f}", ha='center', va='bottom', fontsize=9)

    ax.set_title("Precision, Recall, and F1-score per Algorithm and Version", fontsize=14)
    ax.set_ylabel("Score")
    ax.set_xlabel("Metric")
    ax.set_ylim(0, 1.1)
    ax.legend(title="Algorithm", loc='upper right')
    plt.tight_layout()
    plt.savefig("results/img/Precision_Recall_F1-score.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    df_metrics = pd.DataFrame(modularity_time)

    plot_execution_time_and_modularity(df_metrics)
    plot_community_size_distribution(community_sizes)
    plot_partition_similarity(partitions)
    plot_fscore_comparison(PrecisionFscoreRec)