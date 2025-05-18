import networkx as nx
import numpy as np
import random

class GraphPreprocessor:
    def __init__(self, G: nx.Graph, noise_threshold=0.1, z_threshold=3.0, node_attributes: list=None, edge_attributes: list=None, unique_attributes: list=None, selected_node_attributes: list=None, selected_edge_attributes: list=None, sample_fraction=1.0):
        self.noise_threshold = noise_threshold
        self.z_threshold = z_threshold
        self.node_attributes = node_attributes
        self.edge_attributes = edge_attributes
        self.unique_attributes = unique_attributes
        self.selected_node_attributes = selected_node_attributes
        self.selected_edge_attributes = selected_edge_attributes
        self.sample_fraction = sample_fraction
        self.G = G

    # For noise detection and removal we will remove edges with low weight and isolated nodes
    def remove_noise(self):
        # removes edges with noise and isolated nodes from the graph
        # first edges with weight below threshold will be removed
        edges_to_remove = []

        # iterate over all edges and remove those with weight below the threshold
        for node_1, node_2, data in self.G.edges(data=True):
            weight = data.get('weight', 1)
            if weight < self.noise_threshold:
                edges_to_remove.append((node_1, node_2))
        self.G.remove_edges_from(edges_to_remove)
        print(f"Removed {len(edges_to_remove)} edges with weight below {self.noise_threshold:}.")

        # remove isolated nodes
        # from networkx: An isolate is a node with no neighbors (that is, with degree zero). For directed graphs, this means no in-neighbors and no out-neighbors
        isolated_nodes = list(nx.isolates(self.G))
        self.G.remove_nodes_from(isolated_nodes)
        print(f"Removed {len(isolated_nodes)} isolated nodes.")

    # For outlier detection and removal we will remove nodes with degree far from the mean
    def remove_outliers(self):
        # remove nodes that are considered outliers based on node degree
        # nodes with degree far from the mean (beyond the defined threshold standard deviations) are removed
        # create a list to store the degree of each node
        degrees = []
        # iterate over all nodes and add their degree to the list
        for node in self.G.nodes:
            degrees.append(self.G.degree(node))

        mean_degree = np.mean(degrees)
        std_degree = np.std(degrees)

        # create a list to store detected outlier nodes
        outliers = []
        # iterate over all nodes and calculate the z-score for each node
        # if the absolute z-score exceeds the threshold, this node is as an outlier
        for node in self.G.nodes:
            if std_degree != 0:
                z_score = (self.G.degree(node) - mean_degree) / std_degree
            else:
                z_score = 0
            if abs(z_score) > self.z_threshold:
                outliers.append(node)

        self.G.remove_nodes_from(outliers)
        print(f"Removed {len(outliers)} outlier nodes with degree z-score above {self.z_threshold}.")

    # For missing values detection and removal we will remove edges and nodes with missing attributes
    def remove_missing_values(self):
        if self.node_attributes is None:
            print("No attributes specified for node missing values removal.")
        else:
            # iterate through the attributes and check if they are present in the graph
            # if not, remove the nodes with missing attributes
            for attribute in self.node_attributes:
                # list to store nodes with missing attributes
                missing_nodes = []
                # iterate through the nodes and their attributes in the graph
                for node, data in self.G.nodes(data=True):
                    # if attribute is missing add node to the missing_nodes list
                    if attribute not in data:
                        missing_nodes.append(node)

            # if there are any nodes with missing attributes, remove them
            if len(missing_nodes) > 0:
                self.G.remove_nodes_from(missing_nodes)
                print(f"Removed {len(missing_nodes)} nodes with missing attributes.")

        if self.edge_attributes is None:
            print("No attributes specified for edge missing values removal.")
        else:
            # iterate through the attributes and check if they are present in the graph
            # if not, remove the edges with missing attributes
            for attribute in self.edge_attributes:
                missing_edges = []
                # iterate through the edges and their attributes in the graph
                for node_1, node_2, data in self.G.edges(data=True):
                    # if attribute is missing add edge to the missing_edges list
                    if attribute not in data:
                        missing_edges.append((node_1, node_2))

            # if there are any edges with missing attributes, remove them
            if len(missing_edges) > 0:
                self.G.remove_edges_from(missing_edges)
                print(f"Removed {len(missing_edges)} edges with missing attributes.")

    # For duplicate edges detection and removal we will remove edges that appear more than once in the graph
    # For duplicate nodes detection and removal we will find nodes that have the same attributes and remove them
    def remove_duplicates(self):
        # dictionary to store how many times each edge appears in the graph
        # this will help us to identify duplicate edges
        existing_edges = {}
        # list to store duplicate edges
        # this will help us to remove duplicate edges
        duplicate_edges = []

        for node_1, node_2 in self.G.edges():
            # keep same ordering (important for undirected graphs)
            edge_key = tuple(sorted([node_1, node_2]))
            # if the edge already exists in the dictionary, it is a duplicate
            if edge_key in existing_edges:
                duplicate_edges.append((node_1, node_2))
            # if the edge does not exist in the dictionary, add it
            else:
                existing_edges[edge_key] = 1

        # if there are any duplicate edges, remove them
        if len(duplicate_edges) > 0:
            self.G.remove_edges_from(duplicate_edges)
            print(f"Removed {len(duplicate_edges)} duplicate edges.")

        if self.unique_attributes is None:
            print("No unique attributes specified for duplicate nodes removal.")
        else:
            # dictionary to store how many times each attribute appears in the graph
            # this will help us to identify duplicate nodes
            existing_nodes_and_attributes = {}
            # list to store duplicate nodes
            duplicate_nodes = []
            # iterate through the nodes and get their attributes
            for node, data in self.G.nodes(data=True):
                attribute_values = []
                # iterate through the attributes and get their values
                # if the attribute is not present in the node, add None to the list
                for attr in self.unique_attributes:
                    attribute_values.append(data.get(attr, None))
                attribute_values = tuple(attribute_values)
                if attribute_values in existing_nodes_and_attributes:
                    duplicate_nodes.append(node)
                else:
                    existing_nodes_and_attributes[attribute_values] = node

            # if there are any duplicate nodes, remove them
            if len(duplicate_nodes) > 0:
                self.G.remove_nodes_from(duplicate_nodes)
                print(f"Removed {len(duplicate_nodes)} duplicate nodes.")

    # For feature selection we will keep only the selected attributes for nodes and edges
    # and remove any other attributes that are not in the selected lists
    def feature_selection(self):
        if self.selected_node_attributes is None:
            print("No selected node attributes specified for feature selection.")
        else:
            # iterate through all nodes and their attributes
            for node, data in self.G.nodes(data=True):
                # delete attributes from the node that are not in the selected attributes list
                for key in list(data.keys()):
                    if key not in self.selected_node_attributes:
                        del data[key]

        if self.selected_edge_attributes is None:
            print("No selected edge attributes specified for feature selection.")
        else:
            # iterate through all edges and their attributes
            for node_1, node_2, data in self.G.edges(data=True):
                # delete attributes from the edge that are not in the selected attributes list
                for key in list(data.keys()):
                    if key not in self.selected_edge_attributes:
                        del data[key]
        print("Feature selection completed: kept selected node and edge attributes only.")

    # For feature extraction we will compute degree centrality and betweenness centrality
    # and assign them to the nodes as attributes
    def feature_extraction(self):
        # compute degree centrality for each node and assign it as an attribute
        # degree centrality is the fraction of nodes that a node is connected to
        degree_centrality = nx.degree_centrality(self.G)
        # iterate through the nodes and add the degree centrality as an attribute
        for node, centrality in degree_centrality.items():
            self.G.nodes[node]['degree_centrality'] = centrality
        
        # compute betweenness centrality for each node and assign it as an attribute
        # betweenness centrality is the fraction of shortest paths that pass through a node
        betweenness_centrality = nx.betweenness_centrality(self.G)
        # iterate through the nodes and add the betweenness centrality as an attribute
        for node, centrality in betweenness_centrality.items():
            self.G.nodes[node]['betweenness_centrality'] = centrality

        print("Feature extraction completed: added degree and betweenness centrality.")

    # For random sampling we will keep only a random subset of nodes and their edges
    # and remove any other nodes and edges that are not in the selected subset
    def random_sampling(self):
        if (self.sample_fraction == 1.0):
            print("Sample fraction is 1.0, no sampling will be performed.")
        else:
            # get list of nodes in the graph
            nodes = list(self.G.nodes)
            # determine number of nodes to keep
            total_nodes = len(nodes)
            sample_size = int(total_nodes * self.sample_fraction)

            # randomly keep some nodes
            sampled_nodes = random.sample(nodes, sample_size)

            # remove nodes that are not in the sampled nodes
            nodes_to_remove = []
            for node in nodes:
                if node not in sampled_nodes:
                    nodes_to_remove.append(node)

            self.G.remove_nodes_from(nodes_to_remove)
            print(f"Sampled {sample_size} nodes out of {total_nodes}. Removed {len(nodes_to_remove)} nodes.")

    def quality_checks(self):
        self.remove_noise()
        self.remove_outliers()
        self.remove_missing_values()
        self.remove_duplicates()

    def preprocessing(self):
        # no aggregation phase since we will perform it in the Louvain and Leiden algorithms later
        self.feature_selection()
        self.feature_extraction()
        self.random_sampling()

    def process(self):
        self.quality_checks()
        self.preprocessing()
        return self.G
