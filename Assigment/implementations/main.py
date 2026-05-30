from graph.graph import Graph
def main():
    # g = Graph(4)

    # g.add_edge(0, 1, 10)
    # g.add_edge(0, 2, 5)
    # g.add_edge(0, 3, 2)
    # g.add_edge(1, 0, 2)
    # g.add_edge(1, 2, 1)
    # g.add_edge(1, 3, 6)
    # g.add_edge(2, 0, 2)
    # g.add_edge(2, 1, 3)
    # g.add_edge(2, 3, 9)
    # g.add_edge(3, 0, 2)
    # g.add_edge(3, 1, 3)
    # g.add_edge(3, 2, 4)

    h = Graph.graph_generator(
        "sparse.txt",
        "./results/gen_graphs",
        density=0.05,
        n=100
    )

    dist, prev, exec_time = h.dijkstra(0)
    # print("Distances:", dist)
    # print("Previous nodes:", prev)
    print(f"Dijkstra's algorithm took {exec_time:.6f} seconds")
    h.draw()

if __name__ == "__main__":
    main()