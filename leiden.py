import networkx as nx
from louvain import Louvain

class Leiden(Louvain):
    def __init__(self, G: nx.Graph):
        super().__init__(G)

    def best_partition(self):
        super().best_partition()
        self.refine_partition()

    def refine_partition(self):
        # Refine communities to ensure intra-community connection (Leiden refinement phase)
        # in this dictionaty we will store the new refined partition (node -> new community ID)
        new_partition = {}
        community_counter = 0

        # iterate over the unique communities in the current partition dictionary
        # and for each community, we will find all connected components
        # within the subgraph induced by the nodes of that community
        # and assign a new community ID to each connected component
        # this will ensure that all communities are now connected components
        partition_communities = set(self.partition.values())
        for community in partition_communities:
            # get all nodes that belong to the current community
            nodes_in_community = [n for n in self.partition if self.partition[n] == community]
            # create a subgraph with the nodes of the current community
            subgraph: nx.Graph = self.G.subgraph(nodes_in_community)
            # find all connected components within this subgraph
            # if the community is disconnected we will find each connected part
            # and assign it a new community ID
            for component in nx.connected_components(subgraph):
                # for every node in this connected component, assign it a new community ID
                for node in component:
                    new_partition[node] = community_counter
                community_counter += 1

        # after processing all communities and their connected components
        # update the partition to the refined one where all communities are now connected components
        self.partition = new_partition

    def aggregate_graph(self):
        # Leiden's aggregation phase reusing Louvain's aggregate_graph and refining communities
        # save current original_nodes before Louvain aggregation modifies it
        # this keeps track of which original graph nodes belong to which hypernodes
        original_nodes_copy = self.original_nodes.copy()
        _, hypernode_partition = super().aggregate_graph()

        # Leiden refinement — split disconnected communities into connected components
        # in this dictionaty we will store the new refined partition (node -> new community ID)
        refined_partition = {}
        community_counter = 0

        # iterate over the unique communities in the current partition dictionary
        partition_communities = set(self.partition.values())
        for community in partition_communities:
            # create a subgraph with the nodes of the current community
            nodes_in_community = [n for n in self.partition if self.partition[n] == community]
            # create a subgraph with the nodes of the current community
            subgraph: nx.Graph = self.G.subgraph(nodes_in_community)
            # find all connected components within this subgraph
            for component in nx.connected_components(subgraph):
                # for every node in this connected component, assign it a new community ID
                for node in component:
                    refined_partition[node] = community_counter
                community_counter += 1

        new_G, hypernode_partition = self.build_aggregated_graph(refined_partition, original_nodes_copy, False)
        self.partition = refined_partition

        return new_G, hypernode_partition

    def run(self, print_results=True):
        return super().run('Leiden', print_results=print_results)
