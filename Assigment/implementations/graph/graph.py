import matplotlib, time, os, random, heapq

matplotlib.use("TkAgg") 

import matplotlib.pyplot as plt
import networkx as nx
from typing import List, Tuple

class Graph:
    """
    Initializes the graph adjacency matrix with a nodes X nodes size matrix
    """
    def __init__(
            self, 
            nodes: int
        ):
        self.nodes = nodes
        self.adj_matrix = [
            [0 for _ in range(nodes)] 
            for _ in range(nodes)
        ]

    def add_edge(
            self, 
            u: int, 
            v: int, 
            w: int
        ):
        self.adj_matrix[u][v] = w

    def remove_edge(
            self, 
            u: int, 
            v: int
        ):
        self.adj_matrix[u][v] = 0

    def get_neighbors(
            self, 
            u: int
        ) -> List[Tuple[int, int]]:
        return [(v, self.adj_matrix[u][v]) 
                for v in range(self.nodes) 
                    if self.adj_matrix[u][v] > 0]

    def has_edge(
            self, 
            u: int, 
            v: int
        ) -> bool:
        return self.adj_matrix[u][v] > 0

    def print_matrix(self):
        for row in self.adj_matrix:
            print(row)

    """
    Responsible for drawing the graph network using the networkx library.
    """
    def draw(self):
        G = nx.DiGraph()

        for n in range(self.nodes):
            G.add_node(n)


        for i in range(self.nodes):
            for j in range(self.nodes):
                if self.adj_matrix[i][j] > 0:
                    G.add_edge(i, j, weight=self.adj_matrix[i][j])

        pos = nx.spring_layout(G, seed=42)

        nx.draw(
            G, 
            pos, 
            with_labels=(self.nodes <= 20), 
            node_color='lightblue', 
            node_size=500 * 1 / self.nodes
        )

        edge_labels = nx.get_edge_attributes(
            G, 
            'weight'
        )
        
        nx.draw_networkx_edge_labels(
            G, 
            pos, 
            edge_labels=edge_labels
        )

        plt.show(block=True)

        save_path = f"./results/graphs/graph_{self.nodes}_nodes.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Graph visualization saved to {save_path}")

    def dijkstra(
            self,
            source: int
    ):
        start = time.   perf_counter()
        dist: dict[int, float] = {
            node: float('inf') 
            for node in range(self.nodes)
        }
        visited: dict[int, bool] = {
            node: False 
            for node in range(self.nodes)
        }
        prev: dict[int, int | None ] = {
            node: None 
            for node in range(self.nodes)
        }
        dist[source] = 0

        pq: list[tuple[float, int]] = [(0, source)]

        while pq:
            current_dist, current_node = heapq.heappop(pq)
            
            if visited[current_node]:
                continue

            visited[current_node] = True

            for neighbor, weight in self.get_neighbors(current_node):

                if weight <= 0:
                    continue

                new_distance = current_dist + weight

                if new_distance < dist[neighbor]:
                    dist[neighbor] = new_distance
                    prev[neighbor] = current_node
                    heapq.heappush(
                        pq, 
                        (new_distance, neighbor)
                    )
        end = time.perf_counter()
        execution_time = end - start
        
        return dist, prev, execution_time

    @staticmethod
    def graph_generator(
            file_name: str,
            file_path: str,
            density: float,
            n: int,
            minimum_weight: int = 1,
            maximum_weight: int = 20,
            directed: bool = True
    ) -> 'Graph':
        graph = Graph(n)

        if not (0 <= density <= 1):
            raise ValueError("Density must be between 0 and 1.")
        
        max_edges = (
            n * (n - 1) if directed 
            else n * (n - 1) // 2
        )

        e = int(density * max_edges)

        edges: set[tuple[int, int]] = set()

        while len(edges) < e:
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)

            if u == v:
                continue

            if directed:
                edge = (u, v)
            else:
                if u < v:
                    edge = (u, v)
                else:
                    edge = (v, u)

            if edge in edges:
                continue

            edges.add(edge)

            weight = random.randint(
                minimum_weight, 
                maximum_weight
            ) 

            graph.add_edge(u, v, weight)

        s = random.randint(0, n - 1)

        os.makedirs(file_path, exist_ok=True)

        with open(
            f"{file_path}/{file_name}",
            "w",
            encoding="utf-8"
            ) as f:

            f.write(f"{n} {e}\n")
            for u, v in edges:
                weight = random.randint(minimum_weight, maximum_weight)
                f.write(f"{u} {v} {weight}\n")

            f.write(f"{s}\n")

        print(f"Graph generated and saved to {file_path}/{file_name}, with {n} nodes, {e} edges, and density {density:.2f}.")

        return graph

    
    @classmethod
    def from_txt(
            cls,
            filename: str
    ):
        with open(
            filename,
            "r",
            encoding="utf-8"
        ) as f:

            n, e = map(
                int,
                f.readline().split()
            )

            graph = cls(n)

            for _ in range(e):
                u, v, w = map(
                    int,
                    f.readline().split()
                )

                graph.add_edge(u, v, w)

            s = int(f.readline())

        return graph, s