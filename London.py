import numpy as np
import sympy as sp
from numpy import linalg as LA
import matplotlib.pyplot as plt 

n = 17

# Initialize an n×n matrix of zeros
adj_matrix = np.zeros((n, n), dtype=int)

adj_list = {
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

# Fill the adjacency matrix (undirected graph)
for i, neighbors in adj_list.items():
    for j in neighbors:
        adj_matrix[i-1, j-1] = 1
        adj_matrix[j-1, i-1] = 1  # symmetric

# Print the adjacency matrix
# print(adj_matrix)

eigenvalues, eigenvectors = LA.eig(adj_matrix)

#print(eigenvalues)
# print(eigenvectors)

degrees = [len(adj_list[i]) for i in sorted(adj_list)]

norm_degrees = [d / (len(degrees) - 1) for d in degrees]

plt.hist(degrees)
plt.xlabel("Degree")
plt.ylabel("Frequency")
plt.title("Histogram of Node Degrees (London)")
plt.show()

plt.hist(norm_degrees)
plt.xlabel("Degree")
plt.ylabel("Frequency")
plt.title("Histogram of Normalised Node Degrees (London)")
plt.show()

def laplacian_from_adj(A):
    # Degree matrix is just a diagonal of row sums
    degrees = np.sum(A, axis=1)
    D = np.diag(degrees)
    return D - A

laplacian = laplacian_from_adj(adj_matrix)

def num_spanning_trees_exact(adj):
    L = laplacian_from_adj(adj_matrix)
    n = L.shape[0]
    if n <= 1:
        return 1
    # delete first row and column
    Lm = L[1:, 1:]
    Lm_sym = sp.Matrix(Lm)
    return int(Lm_sym.det())

print(num_spanning_trees_exact(adj_matrix))







