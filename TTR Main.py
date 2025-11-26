import numpy as np
import sympy as sp
from numpy import linalg as LA
import matplotlib.pyplot as plt 

np.set_printoptions(threshold=np.inf)

adj_list_London = {
    1:  [2, 6, 7, 11],
    2:  [1, 3, 6],
    3:  [2, 4, 6],
    4:  [3, 5, 6, 9],
    5:  [4, 9, 10],
    6:  [1, 2, 3, 4, 7, 8],
    7:  [1, 6, 8, 11, 12, 13],
    8:  [6, 7, 9, 13],
    9:  [4, 5, 8, 10, 16],
    10: [5, 9, 16, 17],
    11: [1, 7, 12],
    12: [7, 11, 13, 14],
    13: [7, 8, 12, 14, 15],
    14: [12, 13, 15, 17],
    15: [13, 14, 16, 17],
    16: [9, 10, 15, 17],
    17: [10, 14, 15, 16],
}

# New York: Max 15 Units

adj_list_NY = {
    1: [2, 3, 4],
    2: [1, 4, 5],
    3: [1, 4, 6, 7],
    4: [1, 2, 3, 5, 6],
    5: [2, 4, 6, 8],
    6: [3, 4, 5, 7, 8],
    7: [3, 6, 8, 9, 11],
    8: [5, 6, 7, 9, 10],
    9: [7, 8, 10, 11, 12, 13],
    10: [8, 9, 12],
    11: [7, 9, 14],
    12: [9, 10, 13, 15],
    13: [15, 14, 12, 9],
    14: [11, 13, 15],
    15: [14, 13, 12]
}

adj_list_NY_W = {
    1: [(2,2), (3,2), (4,2)],
    2: [(1,2), (4,2), (5,2)],
    3: [(1,2), (4,1), (6,2), (7,2)],
    4: [(1,2), (2,2), (3,1), (5,2), (6,1)],
    5: [(2,2), (4,2), (6,2), (8,3)],
    6: [(3,2), (4,1), (5,2), (7,2), (8,1)],
    7: [(3,2), (6,2), (8,2), (9,3), (11,4)],
    8: [(5,3), (6,1), (7,2), (9,2), (10,2)],
    9: [(7,3), (8,2), (10,2), (11,2), (12,2), (13,2)],
    10: [(8,2), (9,2), (12,1)],
    11: [(7,4), (9,2), (14,2)],
    12: [(9,1), (10,2), (13,1), (15,3)],
    13: [(15,3), (14,1), (12,1), (9,2)],
    14: [(11,2), (13,1), (15,3)],
    15: [(14,3), (13,3), (12,3)]
}

adj_list_FJ = {
    1: [2, 3, 9, 10],
    2: [1, 3, 4],
    3: [1, 2, 4, 10, 11],
    4: [2, 3, 5],
    5: [4, 6, 8, 11, 12],
    6: [5, 7, 8],
    7: [6, 8, 14],
    8: [5, 6, 7, 12, 13, 14],
    9: [1, 15, 10],
    10: [1, 3, 9, 11, 15, 16],
    11: [3, 5, 12, 16, 10],
    12: [5, 8, 11, 17, 13],
    13: [18, 19, 14, 12, 8, 17],
    14: [7, 8, 13, 19],
    15: [9, 10, 16],
    16: [10, 11, 15, 17],
    17: [16, 18, 13, 12],
    18: [17, 13, 19],
    19: [18, 13, 14]
}

adj_list_FJE = {
    1:  [2, 11],
    2:  [1, 3, 11, 12],
    3:  [2, 4, 8, 12],
    4:  [3, 5, 8],
    5:  [4, 6],
    6:  [5, 9, 7],
    7:  [6, 9, 10, 16],
    8:  [3, 4, 9, 12, 14],
    9:  [6, 7, 8, 10, 14],
    10: [9, 7, 14, 15, 16],
    11: [1, 2, 12, 17],
    12: [2, 3, 8, 11, 13, 17, 18],
    13: [12, 14, 19],
    14: [8, 9, 10, 13, 15, 19, 20],
    15: [10, 14, 16, 20, 21, 22],
    16: [7, 10, 15, 22],
    17: [11, 12, 18],
    18: [12, 17, 19],
    19: [13, 14, 18, 20, 21],
    20: [14, 15, 19, 21],
    21: [15, 19, 20, 22],
    22: [15, 16, 21]
}

adj_list_Am = {
    1:  [2, 4],
    2:  [1, 3, 4, 5],
    3:  [2, 6, 7, 8],
    4:  [1, 2, 5, 9, 10],
    5:  [2, 4, 6, 10],
    6:  [3, 5, 7, 11],
    7:  [3, 6, 8, 12],
    8:  [3, 7, 13],
    9:  [4, 14, 15],
    10: [4, 5, 11, 15],
    11: [6, 10, 12, 16],
    12: [7, 11, 13, 17],
    13: [8, 12, 17],
    14: [9, 15, 16],
    15: [9, 10, 14, 16],
    16: [11, 14, 15, 17],
    17: [12, 13, 16]
}

adj_list_USA = {
    1: [2,3],
    2: [1,3,7,8],
    3: [1,2,4],
    4: [3,5,9],
    5: [4,6,9],
    6: [5,10,11,18],
    7: [2,8,12,21],
    8: [2,3,7,9,12,13,14],
    9: [4,5,8,10,14],
    10: [6,9],
    11: [6,14,16,18],
    12: [7,8,13,21,22,26],
    13: [8,12,14,15,22],
    14: [8,9,11,13,15,16,17],
    15: [13,14,17,23],
    16: [11,14,17,18], 
    17: [14,15,16,18,19,24],
    18: [6,11,16,17,19,20],
    19: [17,18,20,24],
    20: [18,19,25],
    21: [7,12,26,30],
    22: [12,13,23,26,27],
    23: [15,22,24,27,28],
    24: [17,19,23,25,28],
    25: [20,24,29,36],
    26: [12,21,22,27,30],
    27: [22,23,26,28,32,33,34],
    28: [23,24,27,29,34],
    29: [25,28,34,35,36],
    30: [21,26,31,32],
    31: [30,32],
    32: [27,30,31,33],
    33: [27,32,34],
    34: [27,28,29,33,35],
    35: [29,34,36],
    36: [25,29,35]
    }

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

print(form_weighted_adj_matrix(15, adj_list_NY_W))

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
    
    # Max length allowed is 15 before negatives
    
    start = np.zeros((n, n), dtype=int)
    
    for i in range(1,max_length):
        A = matrix_power(form_adj_matrix(n, AL), i)
        start = np.add(start, A)

    return start[i][j]
