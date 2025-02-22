import sys
import math
import heapq
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt


def path_edge(path, e):
    for i in range(len(path) - 1):
        v1, v2 = (path[i], path[i + 1])
        if e in ((v1, v2), (v2, v1)):
            return True
    return False


def plot_graph(G, path, pos, green):
    plt.figure(figsize=(10, 8))

    node_color = ["red" if node in path else "lightblue" for node in G.nodes()]
    edge_color = []
    for u, v in G.edges():
        if (u, v) in green or (v, u) in green:
            edge_color.append("green")
        elif path_edge(path, (u, v)):
            edge_color.append("red")
        else:
            edge_color.append("lightgray")

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

    plt.title("Odd walk")
    plt.show()


def cheapest_parity_walk(G, s, k):
    """Computes the cheapest parity walk with k free edges."""
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


def interpolate(p1, p2):
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)


def subdivide(G, pos, dont=[]):
    G = G.copy()
    subdivision = []
    for e in dont:
        if e not in G.edges():
            if len(G.edges()) <= 10:
                print(sorted(G.edges()))
            else:
                print(sorted(G.edges())[:10], "...")
            raise ValueError("Not an edge in dont: ", e)
    for u, v in G.edges():
        if (u, v) in dont or (v, u) in dont:
            continue
        ux, uy = pos[u]
        vx, vy = pos[v]
        s = interpolate((ux, uy), (vx, vy))
        subdivision.append((s, u, v))
    n = len(G.nodes())
    for s, u, v in subdivision:
        w = G.edges[u, v]["weight"] / 2
        G.remove_edge(u, v)
        G.add_edge(n, u, weight=int(1 + w))
        G.add_edge(n, v, weight=int(1 + w))
        pos[n] = s
        n += 1
    return G


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
        if w <= 0:
            raise ValueError("Weights must be positive")
        G.add_edge(u, v, weight=int(1 + w))
    return G, pos


def backtrack(G, s, t, k, p, opt):
    np = 1 - p
    yield t
    if s == t and p == 0:
        return
    l = opt[k, p, t]
    for v in G[t]:
        w = G.edges[v, t]["weight"]
        if opt[k - 1, np, v] <= (l + 0.001):
            yield from backtrack(G, s, v, k - 1, np, opt)
            break
        elif opt[k, np, v] <= (l - w + 0.001):
            yield from backtrack(G, s, v, k, np, opt)
            break


if __name__ == "__main__":
    if len(sys.argv) != 6:
        exit("Usage: oddwalk file.in s t e")
    fname = sys.argv[1]
    s = int(sys.argv[2])
    t = int(sys.argv[3])
    b1 = int(sys.argv[4])
    b2 = int(sys.argv[5])

    K = 0

    G, pos = read_graph(fname)
    dont = [(b1, b2)]
    print(dont)
    G = subdivide(G, pos, dont=dont)
    d = cheapest_parity_walk(G, s, K)
    p = 1  # parity: odd=1, even=0
    path = list(reversed(list(backtrack(G, s, t, K, p, d))))
    print(path)
    assert (len(path) - 1) % 2 == p
    plot_graph(G, path, pos, green=dont)
