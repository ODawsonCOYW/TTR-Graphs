import numpy as np
import sympy as sp
import heapq
from numpy import linalg as LA
import matplotlib.pyplot as plt 
import networkx as nx
from numpy.linalg import matrix_power
from collections import deque
from Graph_Data import Graph, adj_list_EU, adj_list_Am, adj_list_FJ, adj_list_FJE, adj_list_London, routes_Germany
from Graph_Data import adj_list_NY, adj_list_NY_W, adj_list_USA, adj_NL, adj_list_GER, adj_list_OW, adj_list_HoA
from Graph_Data import Europe_Route_Freq, adj_list_PEN, routes_EU, routes_USA, routes_NY, routes_PEN, routes_London
from Graph_Data import routes_India, adj_list_India, adj_list_USA_W, adj_list_EU_W, adj_list_PEN_W, adj_list_India_W
from Graph_Data import adj_list_London_W, adj_list_GER_W
from math import inf
from itertools import combinations

np.set_printoptions(threshold=np.inf)

def form_adj_matrix(n, AL):

    adj_matrix = np.zeros((n, n), dtype=int)

    # Fill the adjacency matrix (undirected graph)
    for i, neighbors in AL.items():
        for j in neighbors:
            adj_matrix[i-1, j-1] = 1
            adj_matrix[j-1, i-1] = 1  # symmetric
            
    return adj_matrix

def form_weighted_adj_matrix(n, WAL):
    
    adj_matrix = np.zeros((n, n), dtype=int)
    
    for u in adj_list_NY_W:
        for v, w in adj_list_NY_W[u]:
            adj_matrix[u-1][v-1] = w
    
    return adj_matrix

# print(form_weighted_adj_matrix(15, adj_list_NY_W))

def form_spectrum(A):
        
    spectrum, eigenvectors = LA.eig(A)
    return spectrum

def laplacian_from_adj(A):
    # Degree matrix is just a diagonal of row sums
    degrees = np.sum(A, axis=1)
    D = np.diag(degrees)
    return D - A

def num_spanning_trees_exact(n, AL):
    A = form_adj_matrix(n, AL)
    L = laplacian_from_adj(A)
    n = L.shape[0]
    if n <= 1:
        return 1
    # delete first row and column
    Lm = L[1:, 1:]
    Lm_sym = sp.Matrix(Lm)
    return int(Lm_sym.det())


def plot_deg_dist(A_list, name="Graph"):
    degrees = [len(A_list[i]) for i in sorted(A_list)]
    norm_degrees = [d / (len(degrees) - 1) for d in degrees]

    plt.hist(degrees)
    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    plt.title(f"Histogram of Node Degrees ({name})")
    plt.show()

    plt.hist(norm_degrees)
    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    plt.title(f"Histogram of Normalised Node Degrees ({name})")
    plt.show()

def avg_deg_vs_size():
    # Combine in a dictionary
    graphs = {
        "London": adj_list_London,
        "NY": adj_list_NY,
        "FJ": adj_list_FJ,
        "FJE": adj_list_FJE,
        "Am": adj_list_Am,
        "USA": adj_list_USA,
        "EU" : adj_list_EU,
        "Germany": adj_list_GER,
        "Ned": adj_NL,
        "Penn": adj_list_PEN,
        "HoA": adj_list_HoA,
        "India": adj_list_India,
        "Old West": adj_list_OW
    }
    
    # --- Compute stats ---
    num_vertices = []
    avg_degrees = []
    labels = []
    
    for name, adj in graphs.items():
        n = len(adj)
        total_deg = sum(len(neigh) for neigh in adj.values())
        avg_deg = total_deg / n
        
        num_vertices.append(n)
        avg_degrees.append(avg_deg)
        labels.append(name)
    
    # --- Plot ---
    plt.scatter(num_vertices, avg_degrees)
    plt.xlabel("Number of vertices")
    plt.ylabel("Average degree")
    plt.title("Average Degree vs Number of Vertices")
    
    # Label each point
    for i, label in enumerate(labels):
        plt.text(num_vertices[i] + 0.15, avg_degrees[i], label)
    
    plt.show()

def Spanning_trees_vs_size():
    # --- Put your adjacency lists into a list ---
    graphs = [
        ("London", adj_list_London),
        ("NY", adj_list_NY),
        ("FJ", adj_list_FJ),
        ("FJE", adj_list_FJE),
        ("Am", adj_list_Am),
        ("USA", adj_list_USA),
        ("Germany", adj_list_GER),
        ("Ned", adj_NL),
        ("Penn", adj_list_PEN),
        ("HoA", adj_list_HoA),
        ("India", adj_list_India),
        ("Old West", adj_list_OW)
    ]
    
    sizes = []
    spanning_trees = []
    labels = []
    
    # --- Compute spanning trees for each graph ---
    for name, AL in graphs:
        n = len(AL)
        tau = num_spanning_trees_exact(n, AL)
    
        sizes.append(n)
        spanning_trees.append(tau)
        labels.append(name)
    
        print(f"{name}: n={n}, spanning trees = {tau}")
        
        print(spanning_trees)
    
    # --- Scatter plot ---
    plt.scatter(sizes, spanning_trees)
    
    # Label each point with graph name
    for i, label in enumerate(labels):
        plt.annotate(label, (sizes[i], spanning_trees[i]), textcoords="offset points",
                     xytext=(5,5), ha='left')
        
    plt.yscale("log")
    plt.xlabel("Number of vertices (n)")
    plt.ylabel("Number of spanning trees (log 10 scaled)")
    plt.title("Spanning Trees vs Graph Size")
    plt.grid(True)
    plt.show()

#Run a for loop over indices i, let these indices be the power of the adjacency matrix
#Then progressively sum them to an empty nxn matrix. This will form a matrix witht the
#number of walks of maximum length max(i)

def number_of_walks(AL, n, i, j, max_length):
    
    # AL : Adj List
    # n: Number of vertices
    # i, j: i -> j
    # max_length: Maximum number of edges in a path to be considered in the count  
    # Max length allowed is 15 before negatives
    
    start = np.zeros((n, n), dtype=int)
    A = form_adj_matrix(n, AL)
    
    for k in range(1,max_length+1):
        B = matrix_power(A, k)
        start = np.add(start, B)

    return start[i][j]

# The above function counts all walks so use the one below which counts simple paths

def number_of_simple_paths(AL, n, i, j, max_length):
    count = 0
    
    def dfs(current, target, visited, length):
        nonlocal count
        if length > max_length:
            return
        if current == target and length > 0:
            count += 1
        for neighbor in AL[current]:
            if neighbor not in visited:
                dfs(neighbor, target, visited | {neighbor}, length + 1)
    
    dfs(i, j, {i}, 0)
    return count

def number_of_simple_paths_weighted(AL, i, j, max_weight):
    count = 0

    def dfs(current, target, visited, total_weight):
        nonlocal count

        if total_weight > max_weight:
            return

        if current == target and total_weight > 0:
            count += 1

        for neighbor, weight in AL[current]:
            if neighbor not in visited:
                dfs(
                    neighbor,
                    target,
                    visited | {neighbor},
                    total_weight + weight
                )

    dfs(i, j, {i}, 0)
    return count

def connectivity(AL, s, t):
    
    # Returns the number of edge disjoint paths from s to t, using mengers thm to find max flow given unit capacity on edges.
    
    G = nx.Graph()
    
    for u, nbrs in AL.items():
        for v in nbrs:
            if u < v:
                G.add_edge(u, v, capacity=1)
    
    flow_value, _ = nx.maximum_flow(G, s, t)
    
    return f"The number of edge-disjoint paths between {s} and {t} is {flow_value}"

def bfs_shortest_path(AL, s, t):
    dist = {v: float("inf") for v in AL}
    prev = {v: None for v in AL}

    dist[s] = 0
    q = deque([s])

    while q:
        u = q.popleft()

        # early exit: first time we see t is shortest
        if u == t:
            break

        for v in AL[u]:
            if dist[v] == float("inf"):   # not visited
                dist[v] = dist[u] + 1
                prev[v] = u
                q.append(v)

    # if t was never reached
    if dist[t] == float("inf"):
        return None, float("inf")

    # reconstruct path
    path = []
    cur = t
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()

    return path, dist[t]


def dijkstra_shortest_path(graph, source, path_end):
    # shortest distance to each node
    dist = {v: float('inf') for v in graph}
    dist[source] = 0

    # priority queue of (distance, node)
    pq = [(0, source)]

    while pq:
        current_dist, u = heapq.heappop(pq)

        # skip outdated entries
        if current_dist > dist[u]:
            continue

        for v, weight in graph[u]:
            new_dist = current_dist + weight

            if new_dist < dist[v]:
                dist[v] = new_dist
                heapq.heappush(pq, (new_dist, v))

    return dist[path_end]

def count_paths(adj, s, t, target_weight):
    count = 0

    def dfs(u, current_weight, visited):
        nonlocal count

        # prune
        if current_weight > target_weight:
            return

        if u == t:
            if current_weight == target_weight:
                count += 1
            return

        for v, w in adj[u]:
            if v not in visited:
                visited.add(v)
                dfs(v, current_weight + w, visited)
                visited.remove(v)

    dfs(s, 0, {s})
    return count

def find_paths(adj, s, t, target_weight):
    paths = []

    def dfs(u, current_weight, path, visited):
        # prune
        if current_weight > target_weight:
            return

        if u == t:
            if current_weight == target_weight:
                paths.append(path.copy())
            return

        for v, w in adj[u]:
            if v not in visited:
                visited.add(v)
                path.append(v)

                dfs(v, current_weight + w, path, visited)

                path.pop()
                visited.remove(v)

    dfs(s, 0, [s], {s})
    return paths

def number_of_shortest_paths(adj, s, t):
    
    weight = dijkstra_shortest_path(adj, s, t)
    count = count_paths(adj, s, t, weight)
    paths = find_paths(adj, s, t, weight)
    
    return count, paths

def edges_in_paths(paths):
    edges = set()
    for path in paths:
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            edges.add(tuple(sorted((u, v))))
    return edges

def remove_edge(adj, u, v):
    """
    Safely remove undirected edge (u,v).
    Returns the weight if removed, else None.
    """
    w = None

    for i, (x, wt) in enumerate(adj.get(u, [])):
        if x == v:
            w = wt
            adj[u].pop(i)
            break

    for i, (x, wt) in enumerate(adj.get(v, [])):
        if x == u:
            adj[v].pop(i)
            break

    return w


def restore_edge(adj, u, v, w):
    if w is not None:
        adj[u].append((v, w))
        adj[v].append((u, w))

def replacement_shortest_paths(adj, paths, s, t):
    results = {}

    original_dist = dijkstra_shortest_path(adj, s, t)
    edges = edges_in_paths(paths)

    for u, v in edges:
        # remove edge
        w = remove_edge(adj, u, v)

        # recompute shortest path
        new_dist = dijkstra_shortest_path(adj, s, t)
        results[(u, v)] = new_dist

        # restore edge
        restore_edge(adj, u, v, w)

    return original_dist, results

def lambda_robust_shortest_paths(adj, s, t, lam):
    original_dist = dijkstra_shortest_path(adj, s, t)
    paths = find_paths(adj, s, t, original_dist)
    edges = list(edges_in_paths(paths))

    results = {}

    for edge_set in combinations(edges, lam):
        removed = []

        for u, v in edge_set:
            w = remove_edge(adj, u, v)   
            removed.append((u, v, w))

        new_dist = dijkstra_shortest_path(adj, s, t)
        results[edge_set] = new_dist

        for u, v, w in removed:
            restore_edge(adj, u, v, w)   

    return original_dist, results

def adaptive_lambda_all_outcomes(adj, s, t, lam):
    """
    Returns:
        outcomes : dict[tuple[edge], distance]
    """

    outcomes = {}

    def recurse(k, removed_sequence):
        dist = dijkstra_shortest_path(adj, s, t)

        # disconnected
        if dist == inf:
            outcomes[tuple(removed_sequence)] = inf
            return

        # no more removals allowed
        if k == 0:
            outcomes[tuple(removed_sequence)] = dist
            return

        paths = find_paths(adj, s, t, dist)

        edges = set()
        for path in paths:
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                edges.add((u, v))

        for (u, v) in edges:
            w = remove_edge(adj, u, v)
            if w is None:
                continue

            recurse(k - 1, removed_sequence + [(u, v)])

            restore_edge(adj, u, v, w)

    recurse(lam, [])
    return outcomes

# result = adaptive_lambda_all_outcomes(adj_list_London_W, 1, 10, 3)
# max_val = max(result.values())

# max_sequences = [k for k, v in result.items() if v == max_val]

# print(f"The most trains needed to complete the destination ticket allowing for 3 removals is {max_val}")
# print("The edge removal sets that forced this are the following:")
# print(max_sequences)

avg_deg_vs_size()