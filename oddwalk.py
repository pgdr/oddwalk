import sys
import math
import heapq
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt


def plot_graph(G, path, pos):
    plt.figure(figsize=(10, 8))

    node_color = ["red" if node in path else "lightblue" for node in G.nodes()]
    edge_color = [
        (
            "red"
            if (
                u in path
                and v in path
                and ((u, v) in zip(path, path[1:]) or (v, u) in zip(path, path[1:]))
            )
            else "lightgray"
        )
        for u, v in G.edges()
    ]

    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color=node_color,
        node_size=500,
        font_size=10,
        edge_color=edge_color,
    )

    # edge_labels = {(u, v): f'{d["weight"]:.2f}' for u, v, d in G.edges(data=True)}
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    plt.title(sys.argv[1])
    plt.show()


def cheapest_walk(G, s, k):
    opt = defaultdict(lambda: 10**8)
    PQ = []

    # l, k, p, v
    heapq.heappush(PQ, (0, 0, 0, s))

    while PQ:
        ell, k_prime, p, v = heapq.heappop(PQ)

        if opt[(k_prime, p, v)] < ell:
            continue  # discard sub-optimal vector

        opt[(k_prime, p, v)] = ell

        for u in G[v]:
            w = G.edges[u, v]["weight"]
            heapq.heappush(PQ, (round(ell + w, 2), k_prime, 1 - p, u))
            if k_prime < k:
                heapq.heappush(PQ, (ell, k_prime + 1, 1 - p, u))

    return opt


def read_graph(fname):
    edges = []
    with open(fname, "r") as fin:
        n, m = map(int, fin.readline().split())
        pos = {}
        for _ in range(n):
            v, x, y = fin.readline().split()
            pos[int(v)] = float(x), float(y)
        minx = 10**10
        miny = 10**10
        for v, (x, y) in pos.items():
            minx = min(x, minx)
            miny = min(y, miny)
        for v in pos:
            x, y = pos[v]
            pos[v] = round((x - minx) / 1000, 2), round((y - miny) / 1000, 2)
        for _ in range(m):
            u, v, w = fin.readline().split()
            edges.append((int(u), int(v), float(w)))
    G = nx.Graph()
    for u, v, w in edges:
        G.add_edge(u, v, weight=w)
    return G, pos


def backtrack(G, s, t, k, p, opt):
    np = 1 - p
    yield t
    if s == t:
        return
    l = opt[k, p, t]
    for v in G[t]:
        w = G.edges[v, t]["weight"]
        if opt[k - 1, np, v] <= (l + 0.1):
            yield from backtrack(G, s, v, k - 1, np, opt)
        elif opt[k, np, v] <= (l - w + 0.1):
            yield from backtrack(G, s, v, k, np, opt)


if __name__ == "__main__":
    if len(sys.argv) != 5:
        exit("Usage: oddwalk file.in s t k")
    fname = sys.argv[1]
    s = int(sys.argv[2])
    t = int(sys.argv[3])
    K = int(sys.argv[4])
    G, pos = read_graph(fname)
    d = cheapest_walk(G, s, K)
    the_path = list(backtrack(G, s, t, K, 1, d))
    plot_graph(G, the_path, pos)
