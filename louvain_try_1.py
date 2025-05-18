import networkx as nx
import time

class Louvain:
    def __init__(self, G: nx.Graph):
        # initialize the graph
        self.G = G
        self.original_G = G
        # create a dictionary with key/values nodes indicating the edge between them
        # initially each node is a community so the dictionary entries "point" to themselves
        self.partition = {}
        # use this dictionary to keep the list of nodes belonging to communities after iterations to print it later
        self.original_nodes = {}
        for node in G.nodes():
            self.partition.update({node: node})
            self.original_nodes.update({node: [node]})

    def modularity(self):
        # total number of edges
        m = self.G.number_of_edges()
        # initialize modularity
        Q = 0.0
        
        # iterate over the values of the partition dictionary
        # keep them in a set to remove duplicates
        for community in set(self.partition.values()):
            # get all nodes that belong to the current community
            nodes = [n for n in self.partition if self.partition[n] == community]
            # create a subgraph with the nodes of the current community
            subgraph: nx.Graph = self.G.subgraph(nodes)
            # the shared degree is the number of edges of the current community
            shared_degree = subgraph.number_of_edges()
            # the community degree is the sum of the degrees of this community
            community_degree = 0
            for node in nodes:
                community_degree = community_degree + self.G.degree(node)
            
            # modularity contribution from this community
            Q += (shared_degree / m) - (community_degree / (2 * m)) ** 2
        
        return Q

    def best_partition(self):
        # Phase 1: 
        # Iterates over nodes and calculate modularity by creating neighboring communities
        # Continues until the partitions do not evolve anymore
        evolve_further = True
        while evolve_further:
            # set evolve_further to False to exit the loop, unless a better modularity score is found
            evolve_further = False
            previous_modularity = self.modularity()

            # iterate over the graph nodes
            for node in self.G.nodes():
                # initially the "best" community is the single-node-community itself
                best_community = self.partition[node]
                # initially the "best" modularity is from the single-node-community
                best_modularity = previous_modularity

                # check modularity for node with its neighbors
                # re-assign best_modularity if there is a better modularity for the neighbor we iterate over
                for neighbor in self.G.neighbors(node):
                    # try moving to neighbor's community
                    self.partition[node] = self.partition[neighbor]
                    new_modularity = self.modularity()

                    # there is a new better modularity for the given node
                    if new_modularity > best_modularity:
                        # the best community for this node now is with this neighbor node with the better modularity
                        best_community = self.partition[neighbor]
                        best_modularity = new_modularity
                        # since we found a better modularity we need to re-set the evolve_further variable to True
                        # so we do another iteration
                        evolve_further = True
                
                # after all iterations with neighbors are done, the best community (with the highest modularity) for this node is chosen
                self.partition[node] = best_community
            
            # stop if we can't improve modularity more
            if self.modularity() <= previous_modularity:
                evolve_further = False

    def aggregate_graph(self):
        # Phase 2: Community Aggregation - Groups nodes into hyperrnodes
        # create a new graph for aggregated communities
        new_G = nx.Graph()
        # we create a new dictionary (like we had the partition before) to add the hypernode communities
        community_dict = {}

        # iterate over the partition dictionary and add a new key for each node and then a set of nodes
        # we will end up with something like, showing a node as a key and the nodes of the same community as a set:
        # 1 = {0, 1, 2, 4, 5}
        # 6 = {3, 6, 7}
        for node, community in self.partition.items():
            if community not in community_dict:
                community_dict[community] = set()
            community_dict[community].add(node)

        # create the same partition dictionary containing each hypernode as each community, as done in the __init__ method for the initial nodes
        hypernode_partition = {}
        for community in community_dict.keys():
            hypernode_partition.update({community: community})

        # populate the new, aggregated Graph with the new communities
        for community in community_dict.keys():
            new_G.add_node(community)

        # add edges in the new, aggregated Graph
        # iterate over the original graph and get all the connected nodes and the edge weights between them
        for u, v, weight in self.G.edges(data='weight', default=1):
            # retrieve the community that node u belongs to
            comm_u = self.partition[u]
            # retrieve the the community that node v belongs to
            comm_v = self.partition[v]
            # if there is already an edge between the nodes, then just increase the weight of the edge
            if new_G.has_edge(comm_u, comm_v):
                # get the current weight of the edge between the nodes
                current_weight = new_G.get_edge_data(comm_u, comm_v).get('weight', 0)
                # update  the weight
                new_G[comm_u][comm_v]['weight'] = current_weight + weight
            else:
                # if there is no edge already between the nodes, just create one
                new_G.add_edge(comm_u, comm_v, weight=weight)

        # update the original nodes dictionary that is used to print the communities later
        self.original_nodes = {
            community: [node for n in nodes for node in self.original_nodes[n]]
            for community, nodes in community_dict.items()
        }

        return self.original_G, new_G, hypernode_partition


    def run(self, method='Louvain'):
        print(f"Starting {method} algorithm...")
        start_time = time.time()
        # initialize values
        prev_modularity = -1
        current_modularity = self.modularity()
        iteration = 1

        # create loop to iterate over the hypernodes and try to improve modularity each time
        while True:
            # perform phase 1
            self.best_partition()
            current_modularity = self.modularity()
            print(f"Iteration {iteration}, Modularity: {current_modularity}")
            # print the communities of the current iteration and increment
            self.print_current_communities(iteration)
            iteration = iteration + 1

            # if there is an improvement greater than 0.00001 in modularity, only then continue
            # we set this threshold in order not to take into account as improvements very minor changes, perhaps even rounding errors
            if abs(current_modularity - prev_modularity) <= 0.00001:
                break

            # update the values and aggregate to continue with the next iteration
            prev_modularity = current_modularity
            new_G, new_partition = self.aggregate_graph()
            self.G = new_G
            self.partition = new_partition

        # print the communities and the weights after all iterations are done
        self.print_final_community_assignments_and_edge_weights()
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Time taken for {method}: {elapsed_time:.4f} seconds")
        print("=======================================================")
        return self.G, self.partition

    # AI GENERATED - just used to print the communities nicely, no louvain logic here
    def print_current_communities(self, iteration):
        """Prints current community structure using original node IDs."""
        community_dict = {}
        for node, community in self.partition.items():
            for original_node in self.original_nodes[node]:
                if community not in community_dict:
                    community_dict[community] = []  # Initialize the list if the community doesn't exist yet
                community_dict[community].append(original_node)

        formatted_communities = [
            f'Community {idx + 1}: {"-".join(map(str, sorted(nodes)))}'
            for idx, (_, nodes) in enumerate(sorted(community_dict.items()))
        ]

        print(f"Iteration {iteration} Community Assignments:")
        print(formatted_communities)


    # AI GENERATED - just used to print the hypernode communities nicely, no louvain logic here
    def print_final_community_assignments_and_edge_weights(self):
        """Formats and prints detected communities with edge weights in the final graph."""
        community_map = {}
        for node, community in self.partition.items():
            if community not in community_map:
                community_map[community] = []
            community_map[community].extend(self.original_nodes[node])

        print("\nFinal Community Assignments:")
        community_index = 1
        for community, nodes in sorted(community_map.items()):
            print(f"Community {community_index}: {sorted(nodes)}")
            community_index += 1

        print("\nFinal Edge Weights:")
        for u, v, data in self.G.edges(data=True):
            print(f"Edge ({u}, {v}): Weight {data.get('weight', 1)}")
