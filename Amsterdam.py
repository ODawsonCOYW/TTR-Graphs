import numpy as npimport numpy as np
import sympy as sp
from numpy import linalg as LA
import matplotlib.pyplot as plt 

n = 17

# Initialize an n×n matrix of zeros
adj_matrix = np.zeros((n, n), dtype=int)

# Adjacency list (1-indexed)
adj_list = {
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

# Fill the adjacency matrix (undirected graph)
for i, neighbors in adj_list.items():
    for j in neighbors:
        adj_matrix[i-1, j-1] = 1
        adj_matrix[j-1, i-1] = 1  # symmetric

# Print the adjacency matrix
#print(adj_matrix)

eigenvalues, eigenvectors = LA.eig(adj_matrix)

# print(eigenvalues)

degrees = [len(adj_list[i]) for i in sorted(adj_list)]

norm_degrees = [d / (len(degrees) - 1) for d in degrees]

plt.hist(degrees)
plt.xlabel("Degree")
plt.ylabel("Frequency")
plt.title("Histogram of Node Degrees (Amsterdam)")
plt.show()

plt.hist(norm_degrees)
plt.xlabel("Degree")
plt.ylabel("Frequency")
plt.title("Histogram of Normalised Node Degrees (Amsterdam)")
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

### SAGE 

# # number of vertices
# n = 17

# # initialize an n×n zero matrix
# adj_matrix = matrix(ZZ, n, n)

# # adjacency list (1-indexed)
# adj_list = {
#     1:  [2, 4],
#     2:  [1, 3, 4, 5],
#     3:  [2, 6, 7, 8],
#     4:  [1, 2, 5, 9, 10],
#     5:  [2, 4, 6, 10],
#     6:  [3, 5, 7, 11],
#     7:  [3, 6, 8, 12],
#     8:  [3, 7, 13],
#     9:  [4, 14, 15],
#     10: [4, 5, 11, 15],
#     11: [6, 10, 12, 16],
#     12: [7, 11, 13, 17],
#     13: [8, 12, 17],
#     14: [9, 15, 16],
#     15: [9, 10, 14, 16],
#     16: [11, 14, 15, 17],
#     17: [12, 13, 16]
# }

# # fill symmetric adjacency matrix
# for i, neighbors in adj_list.items():
#     for j in neighbors:
#         adj_matrix[i-1, j-1] = 1
#         adj_matrix[j-1, i-1] = 1

# adj_matrix


