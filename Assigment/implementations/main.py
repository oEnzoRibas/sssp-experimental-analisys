import matplotlib, time, os, random, heapq, math, csv

matplotlib.use("TkAgg")

from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx
from typing import List, Tuple

def main():
    runner = BenchmarkRunner()
    runner.run_and_export()

def tests():
    g, s = Graph.graph_generator(e=1, n=4)
    g.add_edge(2, 0, 0)
    g.draw()
    g.print_adj_list()

class Graph:

    def __init__(self, nodes: int):
        self.nodes = nodes
        self.adj_list: list[dict[int, int]] = [{} for _ in range(nodes)]

    def add_edge(self, u: int, v: int, w: int):
        self.adj_list[u][v] = w

    def remove_edge(self, u: int, v: int):
        self.adj_list[u].pop(v, None)

    def get_neighbors(self, u: int) -> list[tuple[int, int]]:
        return list(self.adj_list[u].items())

    def has_edge(self, u: int, v: int) -> bool:
        return v in self.adj_list[u]

    def print_adj_list(self):
        for row in self.adj_list:
            print(row)

    def draw(self):
        G = nx.DiGraph()
        G.add_nodes_from(range(self.nodes))
        for u in range(self.nodes):
            for v, weight in self.adj_list[u].items():
                G.add_edge(u, v, weight=weight)
        pos = nx.shell_layout(G)
        node_values = [G.degree(node) for node in G.nodes()]
        nx.draw(
            G, pos,
            with_labels=(self.nodes <= 20),
            node_color=node_values,
            cmap="viridis",
            node_size=5000 * 1 / self.nodes
        )
        if self.nodes <= 20:
            edge_labels = nx.get_edge_attributes(G, "weight")
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        plt.show(block=True)
        save_path = f"./results/graphs/graph_{self.nodes}_nodes.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Graph visualization saved to {save_path}")

    def dijkstra(self, source: int):
        start = time.perf_counter()
        dist: dict[int, float] = {node: float("inf") for node in range(self.nodes)}
        visited: dict[int, bool] = {node: False for node in range(self.nodes)}
        prev: dict[int, int | None] = {node: None for node in range(self.nodes)}
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
                    heapq.heappush(pq, (new_distance, neighbor))
        end = time.perf_counter()
        return dist, prev, end - start

    def find_pivots(
        self,
        B: float,
        S: list[int],
        d_hat: dict[int, float],
        k: int
    ) -> tuple[set[int], set[int]]:

        W: set[int] = set(S)
        layers: list[list[int]] = [list(S)]

        # FASE 1: BFS de Relaxamento (Exatamente k passos)
        for _ in range(k):
            next_layer: list[int] = []

            for u in layers[-1]:
                base_dist = d_hat.get(u, float("inf"))

                for v, weight in self.get_neighbors(u):
                    if weight <= 0:
                        continue

                    new_dist = base_dist + weight

                    if new_dist < d_hat.get(v, float("inf")) and new_dist <= B:
                        d_hat[v] = new_dist
                        
                        if v not in W:
                            W.add(v)
                            next_layer.append(v)

            layers.append(next_layer)

            # Gatilho de segurança do Lema 3.2
            if len(W) > k * len(S):
                return set(S), W

            # Interrompe se não houver mais vizinhos alcançáveis
            if not next_layer:
                break

        # FASE 2: Construir a Floresta sobre W com distâncias consolidadas
        pred_in_f: dict[int, int] = {}
        for u in W:
            for v, weight in self.get_neighbors(u):
                if weight <= 0:
                    continue
                # Aresta válida apenas se ambos pertencem a W
                if v in W and v not in pred_in_f:
                    # Verifica se faz parte do caminho mínimo (tolerância para float)
                    if abs(d_hat.get(u, float("inf")) + weight - d_hat.get(v, float("inf"))) < 1e-9:
                        pred_in_f[v] = u

        # FASE 3: Identificar Raízes e Propagar tamanhos via DFS
        roots: set[int] = {v for v in W if v not in pred_in_f}
        children: dict[int, list[int]] = {v: [] for v in W}

        for v, u in pred_in_f.items():
            children[u].append(v)

        tree_size: dict[int, int] = {r: 0 for r in roots}
        root_of: dict[int, int] = {r: r for r in roots}
        stack: list[int] = list(roots)

        while stack:
            u = stack.pop()
            r = root_of[u]
            tree_size[r] += 1

            for v in children[u]:
                root_of[v] = r
                stack.append(v)

        # FASE 4: Filtrar Pivôs (Apenas raízes originadas em S com tamanho >= k)
        S_set = set(S)
        P: set[int] = {
            r
            for r in roots
            if r in S_set and tree_size.get(r, 0) >= k
        }

        return P, W

    def partial_dijkstra(
        self,
        source: int,
        extraction_limit: int
    ) -> tuple[dict[int, float], float, list[int]]:
        
        dist: dict[int, float] = {
            node: float("inf") 
            for node in range(self.nodes)
        }
        
        visited: dict[int, bool] = {
            node: False 
            for node in range(self.nodes)
        }
        
        dist[source] = 0
        pq: list[tuple[float, int]] = [(0, source)]
        processed_count = 0
        last_dist = 0.0
        
        while pq and processed_count < extraction_limit:
            current_dist, current_node = heapq.heappop(pq)
            
            if visited[current_node]:
                continue
            
            visited[current_node] = True
            processed_count += 1
            last_dist = current_dist
            
            for neighbor, weight in self.get_neighbors(current_node):
                if weight <= 0:
                    continue
            
                new_distance = current_dist + weight
            
                if new_distance < dist[neighbor]:
                    dist[neighbor] = new_distance
                    heapq.heappush(
                        pq, 
                        (new_distance, neighbor)
                    )

        S = [
            node
            for node in range(self.nodes)
            if (not visited[node] and dist[node] < float("inf"))
        ]
        
        B = last_dist

        return dist, B, S

    def optimized_dijkstra_with_pivots(
        self,
        source: int,
        extraction_limit: int,
        k: int
    ) -> tuple[dict[int, float], dict[str, int], float, float]:
        import time
        start_total = time.perf_counter()

        d_hat, B, S = self.partial_dijkstra(
            source,
            extraction_limit
        )

        d_hat_state = d_hat.copy()
        
        start_pivots = time.perf_counter()
        
        # CORREÇÃO SSSP: B deve ser float("inf") na aplicação como pré-processamento
        P, W = self.find_pivots(
            float("inf"),
            S,
            d_hat_state,
            k
        )
        pivots_only_time = time.perf_counter() - start_pivots

        pivot_list = list(P)
        pivot_distances: list[dict[int, float]] = []

        for p in pivot_list:
            p_dist, _, _ = self.dijkstra(p)
            pivot_distances.append(p_dist)

        combined_dist: dict[int, float] = {
            node: d_hat.get(node, float("inf"))
            for node in range(self.nodes)
        }

        for index, p in enumerate(pivot_list):
            dist_to_pivot = d_hat.get(p, float("inf"))
            p_dist_map = pivot_distances[index]

            for node in range(self.nodes):
                path_via_p = dist_to_pivot + p_dist_map.get(node, float("inf"))

                if path_via_p < combined_dist[node]:
                    combined_dist[node] = path_via_p

        total_execution_time = time.perf_counter() - start_total

        u_tilde = sum(
            1
            for dist in d_hat.values()
            if dist <= B
        )

        metrics = {
            "size_S": len(S),
            "size_W": len(W),
            "size_P": len(P),
            "size_U_tilde": u_tilde
        }

        return combined_dist, metrics, total_execution_time, pivots_only_time

    def bellman_ford(self, source: int):
        start = time.perf_counter()
        dist: dict[int, float] = {node: float("inf") for node in range(self.nodes)}
        prev: dict[int, int | None] = {node: None for node in range(self.nodes)}
        dist[source] = 0
        for _ in range(self.nodes - 1):
            changed = False
            for u in range(self.nodes):
                for v, weight in self.get_neighbors(u):
                    if dist[u] != float("inf") and dist[u] + weight < dist[v]:
                        dist[v] = dist[u] + weight
                        prev[v] = u
                        changed = True
            if not changed:
                break
        for u in range(self.nodes):
            for v, weight in self.get_neighbors(u):
                if dist[u] != float("inf") and dist[u] + weight < dist[v]:
                    raise ValueError("The graph has a negative cycle")
        end = time.perf_counter()
        return dist, prev, end - start

    @staticmethod
    def graph_generator(
        e: int,
        n: int,
        minimum_weight: int = 1,
        maximum_weight: int = 20,
        directed: bool = True,
        file_name: str | None = None,
        file_path: str | None = None
    ) -> tuple["Graph", int]:
        max_possible_edges = n * (n - 1) if directed else (n * (n - 1)) // 2
        if e > max_possible_edges:
            raise ValueError(
                f"Impossible to generate {e} unique edges for {n} nodes. "
                f"Maximum capacity: {max_possible_edges}"
            )
        graph = Graph(n)
        edges: list[tuple[int, int, int]] = []
        seen_edges: set[tuple[int, int]] = set()
        while len(edges) < e:
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            check_u, check_v = (u, v) if directed or u < v else (v, u)
            if (check_u, check_v) in seen_edges:
                continue
            seen_edges.add((check_u, check_v))
            weight = random.randint(minimum_weight, maximum_weight)
            edges.append((check_u, check_v, weight))
            graph.add_edge(check_u, check_v, weight)
            if not directed:
                graph.add_edge(check_v, check_u, weight)
        s = random.randint(0, n - 1)
        if file_name and file_path:
            os.makedirs(file_path, exist_ok=True)
            full_path = os.path.join(file_path, file_name)
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(f"{n} {e}\n")
                for u, v, weight in edges:
                    f.write(f"{u} {v} {weight}\n")
                f.write(f"{s}\n")
            print(f"Graph generated and saved to {full_path}, with {n} nodes, {e} edges")
        return graph, s

    @staticmethod
    def grid_generator(
        n: int,
        minimum_weight: int = 1,
        maximum_weight: int = 20,
        file_name: str | None = None,
        file_path: str | None = None
    ) -> tuple["Graph", int]:
        side = int(math.sqrt(n))
        actual_n = side * side
        nx_graph = nx.grid_2d_graph(side, side)
        mapping = {node: i for i, node in enumerate(nx_graph.nodes())}
        nx_graph = nx.relabel_nodes(nx_graph, mapping)
        
        graph = Graph(actual_n)
        edges: list[tuple[int, int, int]] = []
        for u, v in nx_graph.edges():
            weight = random.randint(minimum_weight, maximum_weight)
            graph.add_edge(u, v, weight)
            graph.add_edge(v, u, weight)
            edges.append((u, v, weight))
            edges.append((v, u, weight))
            
        s = 0 
        
        if file_name and file_path:
            os.makedirs(file_path, exist_ok=True)
            full_path = os.path.join(file_path, file_name)
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(f"{actual_n} {len(edges)}\n")
                for u, v, weight in edges:
                    f.write(f"{u} {v} {weight}\n")
                f.write(f"{s}\n")
            print(f"Grid Graph generated and saved to {full_path}, with {actual_n} nodes")
            
        return graph, s
    
    @classmethod
    def from_txt(cls, filename: str):
        with open(filename, "r", encoding="utf-8") as f:
            n, e = map(int, f.readline().split())
            graph = cls(n)
            for _ in range(e):
                u, v, w = map(int, f.readline().split())
                graph.add_edge(u, v, w)
            s = int(f.readline())
        return graph, s

class BenchmarkRunner:

    def __init__(self, output_dir: str = "results/"):
        self.output_dir = Path(output_dir)
        self.graphs_dir = self.output_dir / "generated_graphs"
        self.graphs_dir.mkdir(parents=True, exist_ok=True)

    def _get_or_create_graph(self, e: int, n: int, filename: str) -> tuple["Graph", int]:
        file_path = self.graphs_dir / filename
        if file_path.exists() and file_path.stat().st_size > 0:
            print(f"Cache Hit: Loading {filename} from disk...")
            return Graph.from_txt(str(file_path))
        print(f"Cache Miss: Generating new graph {filename}...")
        return Graph.graph_generator(
            e=e, n=n, directed=True,
            file_name=filename, file_path=str(self.graphs_dir)
        )
    
    def _get_or_create_grid_graph(self, n: int, filename: str) -> tuple["Graph", int]:
        file_path = self.graphs_dir / filename
        if file_path.exists() and file_path.stat().st_size > 0:
            print(f"Cache Hit: Loading GRID {filename} from disk...")
            return Graph.from_txt(str(file_path))
        print(f"Cache Miss: Generating new GRID graph {filename}...")
        return Graph.grid_generator(
            n=n,
            file_name=filename, file_path=str(self.graphs_dir)
        )

    def _format_result(
        self,
        exp_name: str,
        g_type: str,
        n: int,
        e: int,
        k: int,
        metrics: dict,
        base_time: float,
        opt_time: float,
        bf_time: float | None = None
    ) -> dict:
        data = {
            "Experiment": exp_name,
            "Type": g_type,
            "N": n,
            "E": e,
            "K": k,
            "|S|": metrics["size_S"],
            "|W|": metrics["size_W"],
            "|P|": metrics["size_P"],
            "|U_tilde|": metrics["size_U_tilde"],
            "Dijkstra Time (s)": f"{base_time:.6f}",
            "Pivots Time (s)": f"{opt_time:.6f}",
            "Bellman-Ford Time (s)": f"{bf_time:.6f}" if bf_time is not None else "N/A"
        }
        return data

    def _run_experiment_1_scalability(self) -> list[dict]:
        results = []
        n_values = [100, 500, 1000]
        e_multipliers = [2, 3, 4]
        e_divisors = [2, 4]

        print("\n--- Starting Experiment 1: Scalability (Base Algorithms Only) ---")

        for n in n_values:
            for mult in e_multipliers:
                e_sparse = n * mult
                sparse_filename = f"sparse_n_{n}_e_{mult}n.txt"
                graph_s, source_s = self._get_or_create_graph(e=e_sparse, n=n, filename=sparse_filename)
                _, _, base_time_s = graph_s.dijkstra(source_s)
                start_bf_s = time.perf_counter()
                graph_s.bellman_ford(source_s)
                bf_time_s = time.perf_counter() - start_bf_s
                results.append({
                    "Experiment": "1_Scalability",
                    "Type": "sparse",
                    "N": n,
                    "E": e_sparse,
                    "Dijkstra Time (s)": f"{base_time_s:.6f}",
                    "Bellman-Ford Time (s)": f"{bf_time_s:.6f}"
                })

            for div in e_divisors:
                e_dense = (n * (n - 1)) // div
                dense_filename = f"dense_n_{n}_e_div_{div}.txt"
                graph_d, source_d = self._get_or_create_graph(e=e_dense, n=n, filename=dense_filename)
                _, _, base_time_d = graph_d.dijkstra(source_d)
                _, _, bf_time_d = graph_d.bellman_ford(source_d)
                results.append({
                    "Experiment": "1_Scalability",
                    "Type": "dense",
                    "N": n,
                    "E": e_dense,
                    "Dijkstra Time (s)": f"{base_time_d:.6f}",
                    "Bellman-Ford Time (s)": f"{bf_time_d:.6f}"
                })

        return results

    def _run_experiment_2_sensitivity(self) -> list[dict]:
        import math
        results = []
        k_values = [2, 4, 6, 8, 10, 12, 14, 16, 32 ,64, 128]
        n_values = [100, 500, 1000, 5000, 10000]
        mult = 2
        
        
        print("\n--- Starting Experiment 2: k Sensitivity ---")
        
        for n in n_values:
            e = n * mult
            filename = f"sparse_n_{n}_e_{mult}n.txt"
            graph, source = self._get_or_create_graph(e=e, n=n, filename=filename)
            extraction_limit = int(math.sqrt(n))
            
            _, _, base_time = graph.dijkstra(source)

            for k in k_values:
                _, metrics, total_time, pivots_time = graph.optimized_dijkstra_with_pivots(
                    source=source, extraction_limit=extraction_limit, k=k
                )
                
                size_w = metrics["size_W"]
                size_p = metrics["size_P"]
                
                w_over_k = size_w / k if k > 0 else 1
                ratio_p_w_k = size_p / w_over_k if w_over_k > 0 else 0
                k_times_w = k * size_w
                percent_covered = (metrics["size_U_tilde"] / n) * 100
                
                results.append({
                    "Experiment": "2_Sensitivity",
                    "Type": "sparse",
                    "N": n,
                    "E": e,
                    "K": k,
                    "|S|": metrics["size_S"],
                    "|W|": size_w,
                    "|P|": size_p,
                    "|U_tilde|": metrics["size_U_tilde"],
                    "Ratio |P|/(|W|/k)": f"{ratio_p_w_k:.4f}",
                    "k * |W|": k_times_w,
                    "% Coverage": f"{percent_covered:.2f}%",
                    "Dijkstra Base Time (s)": f"{base_time:.6f}",
                    "FindPivots Only Time (s)": f"{pivots_time:.6f}",
                    "Total Optimized Time (s)": f"{total_time:.6f}"
                })

        return results

    def _run_experiment_3_grid(self) -> list[dict]:
        import math
        results = []
        k_values = [2, 4, 6, 8, 10, 12, 14, 16, 32, 64, 128]
        n_values = [1000, 5000, 10000]
        
        print("\n--- Starting Experiment 3: Grid Topology (The Real Pruning) ---")
        
        for n in n_values:
            side = int(math.sqrt(n))
            actual_n = side * side  # Assegura que é um quadrado perfeito
            filename = f"grid_n_{actual_n}.txt"
            
            graph, source = self._get_or_create_grid_graph(n=actual_n, filename=filename)
            extraction_limit = int(math.sqrt(actual_n))
            
            _, _, base_time = graph.dijkstra(source)

            for k in k_values:
                _, metrics, total_time, pivots_time = graph.optimized_dijkstra_with_pivots(
                    source=source, extraction_limit=extraction_limit, k=k
                )
                
                size_w = metrics["size_W"]
                size_p = metrics["size_P"]
                
                w_over_k = size_w / k if k > 0 else 1
                ratio_p_w_k = size_p / w_over_k if w_over_k > 0 else 0
                k_times_w = k * size_w
                percent_covered = (metrics["size_U_tilde"] / actual_n) * 100
                
                results.append({
                    "Experiment": "3_Grid_Topology",
                    "Type": "grid",
                    "N": actual_n,
                    "E": len(graph.adj_list) * 2, # Aproximação baseada em malha bi-direcional
                    "K": k,
                    "|S|": metrics["size_S"],
                    "|W|": size_w,
                    "|P|": size_p,
                    "|U_tilde|": metrics["size_U_tilde"],
                    "Ratio |P|/(|W|/k)": f"{ratio_p_w_k:.4f}",
                    "k * |W|": k_times_w,
                    "% Coverage": f"{percent_covered:.2f}%",
                    "Dijkstra Base Time (s)": f"{base_time:.6f}",
                    "FindPivots Only Time (s)": f"{pivots_time:.6f}",
                    "Total Optimized Time (s)": f"{total_time:.6f}"
                })

        return results
    
    def run_and_export(self):
        exp1_results = self._run_experiment_1_scalability()
        self._save_to_csv(exp1_results, "benchmark_exp1_scalability.csv")

        exp2_results = self._run_experiment_2_sensitivity()
        self._save_to_csv(exp2_results, "benchmark_exp2_sensitivity.csv")

        exp3_results = self._run_experiment_3_grid()
        self._save_to_csv(exp3_results, "benchmark_exp3_grid.csv")

    def _save_to_csv(self, data: list[dict], filename: str):
        if not data:
            return
        csv_path = self.output_dir / filename
        keys = data[0].keys()
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(data)
        print(f"\nSuccess! Results exported to: {csv_path}")

if __name__ == "__main__":
    main()