import numpy as np
import sympy as sp
from numpy import linalg as LA
import matplotlib.pyplot as plt 
from numpy.linalg import matrix_power
from Graph_Data import Graph, adj_list_EU, adj_list_Am, adj_list_FJ, adj_list_FJE, adj_list_London
from Graph_Data import adj_list_NY, adj_list_NY_W, adj_list_USA, adj_NL

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
        "USA": adj_list_USA
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

# plot_deg_dist(adj_list_NY, name="New York")
# plot_deg_dist(adj_list_London, name="London")
# plot_deg_dist(adj_list_FJ, name="First Journey")
# plot_deg_dist(adj_list_FJE, name="First Journey Europe")
# plot_deg_dist(adj_list_USA, name="USA")

def Spanning_trees_vs_size():
    # --- Put your adjacency lists into a list ---
    graphs = [
        ("London", adj_list_London),
        ("NY", adj_list_NY),
        ("FJ", adj_list_FJ),
        ("FJE", adj_list_FJE),
        ("Am", adj_list_Am),
        ("USA", adj_list_USA),
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



