from neo4j import GraphDatabase
import networkx as nx

class Neo4jGraph:
    def __init__(self):
        self.driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "test1234"))

    def clear(self):
        with self.driver.session() as session:
            session.run(f"MATCH (n) DETACH DELETE n")

    def close(self):
        self.driver.close()

class Neo4jEmailGraphExporter(Neo4jGraph):
    def __init__(self):
        super().__init__()
        self.label = "EmailEuCore"

    def export_graph(self, G: nx.Graph):
        print(f"Exporting graph with label '{self.label}' to Neo4j...")
        with self.driver.session() as session:

            for node, data in G.nodes(data=True):
                props = {
                    "id": node,
                    "name": f"Node {node}",
                    "department": data.get('department')
                }
                session.run(f"""
                    MERGE (n:{self.label} {{id: $id}})
                    SET n.name = $name, n.department = $department
                """, props)

            for u, v, edge_data in G.edges(data=True):
                weight = edge_data.get('weight', 1)

                session.run(f"""
                    MATCH (a:{self.label} {{id: $u}}), (b:{self.label} {{id: $v}})
                    MERGE (a)-[:EMAILED {{weight: $weight}}]->(b)
                """, {"u": u, "v": v, "weight": weight})

            print(f"Graph with label '{self.label}' exported to Neo4j successfully!")

class Neo4jBrainGraphExporter(Neo4jGraph):
    def __init__(self):
        super().__init__()
        self.label = "BrainNetwork"

    def export_graph(self, G: nx.Graph):
        print(f"Exporting graph with label '{self.label}' to Neo4j...")
        with self.driver.session() as session:

            for node, _ in G.nodes(data=True):
                props = {
                    "id": node,
                    "name": f"Node {node}"
                }
                session.run(f"""
                    MERGE (n:{self.label} {{id: $id}})
                    SET n.name = $name
                """, props)

            for u, v, edge_data in G.edges(data=True):
                weight = edge_data.get('weight', 1)

                session.run(f"""
                    MATCH (a:{self.label} {{id: $u}}), (b:{self.label} {{id: $v}})
                    MERGE (a)-[:CONNECTED {{weight: $weight}}]->(b)
                """, {"u": u, "v": v, "weight": weight})

            print(f"Graph with label '{self.label}' exported to Neo4j successfully!")

class Neo4jGraphExporter(Neo4jGraph):
    def __init__(self, label="Node"):
        super().__init__()
        self.label = label

    def find_key_by_value(self, nodes: dict, target_value):
        for key, values in nodes.items():
            if target_value in values:
                return key
        return -1

    def export_graph(self, G: nx.Graph, original_G: nx.Graph, community_dict: dict=None, original_nodes: dict=None, for_louvain=True):
        print(f"Exporting graph with label '{self.label}' to Neo4j...")
        with self.driver.session() as session:

            id_orig = 1
            for node in original_G.nodes():
                props = {"id": f"{id_orig}_{self.label}", "name": f"Node {node}_{self.label}", "dict_id": f"{node}_{self.label}"}
                if for_louvain:
                    node_community = community_dict.get(node, -1)
                else:
                    node_community = self.find_key_by_value(self.transform_nodes_dictionary(original_nodes), node)
                label = ""
                if node_community != -1:
                    label = f"{self.label}_original"
                    community_label = f":Community_{self.label}_{node_community}"
                    session.run(f"CREATE (n:{label}{community_label} $props)", props=props)
                    id_orig += 1

            for node_1, node_2, data in original_G.edges(data=True):
                weight = data.get('weight', 1)
                dict_id_1 = f"{node_1}_{self.label}"
                dict_id_2 = f"{node_2}_{self.label}"
                session.run(f"""
                    MATCH (a:{label} {{dict_id: $u}}), (b:{label} {{dict_id: $v}})
                    MERGE (a)-[:CONNECTED {{weight: $weight}}]->(b)
                """, {"u": dict_id_1, "v": dict_id_2, "weight": weight})

            id = 1
            for node in G.nodes():
                props = {"id": id, "name": f"Community {id}", "dict_id": node}
                id += 1

                if community_dict and node in community_dict:
                    community = community_dict[node]
                    community_name = "-".join(sorted(map(str, original_nodes.get(community, []))))
                    props["community"] = community_name
                session.run(f"CREATE (n:{self.label} $props)", props=props)

            for node_1, node_2, data in G.edges(data=True):
                weight = data.get('weight', 1)
                session.run(f"""
                    MATCH (a:{self.label} {{dict_id: $u}}), (b:{self.label} {{dict_id: $v}})
                    MERGE (a)-[:CONNECTED {{weight: $weight}}]->(b)
                """, {"u": node_1, "v": node_2, "weight": weight})

            print(f"Graph with label '{self.label}' exported to Neo4j successfully!")

    def transform_nodes_dictionary(self, original_nodes: dict):
        nodes = {}
        i = 0
        for node, community in original_nodes.items():
            nodes[i] = community
            i += 1
        return nodes