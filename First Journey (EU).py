import numpy as np
from numpy import linalg as LA

# Number of vertices
n = 22

# Initialize an n×n matrix of zeros
adj_matrix = np.zeros((n, n), dtype=int)

# Adjacency list (1-indexed)
adj_list = {
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

# Fill the adjacency matrix (undirected graph)
for i, neighbors in adj_list.items():
    for j in neighbors:
        adj_matrix[i-1, j-1] = 1
        adj_matrix[j-1, i-1] = 1  # symmetric

# Print the adjacency matrix
print(adj_matrix)

eigenvalues, eigenvectors = LA.eig(adj_matrix)

print(eigenvalues)

# #Sage Code

# # Number of vertices
# n = 22

# # Create an empty n×n zero matrix over the integers
# adj_matrix = matrix(ZZ, n, n, 0)

# # Adjacency list (1-indexed)
# adj_list = {
#     1:  [2, 11],
#     2:  [1, 3, 11, 12],
#     3:  [2, 4, 8, 12],
#     4:  [3, 5, 8],
#     5:  [4, 6],
#     6:  [5, 9, 7],
#     7:  [6, 9, 10, 16],
#     8:  [3, 4, 9, 12, 14],
#     9:  [6, 7, 8, 10, 14],
#     10: [9, 7, 14, 15, 16],
#     11: [1, 2, 12, 17],
#     12: [2, 3, 8, 11, 13, 17, 18],
#     13: [12, 14, 19],
#     14: [8, 9, 10, 13, 15, 19, 20],
#     15: [10, 14, 16, 20, 21, 22],
#     16: [7, 10, 15, 22],
#     17: [11, 12, 18],
#     18: [12, 17, 19],
#     19: [13, 14, 18, 20, 21],
#     20: [14, 15, 19, 21],
#     21: [15, 19, 20, 22],
#     22: [15, 16, 21]
# }

# # Fill the adjacency matrix for an undirected graph
# for i, neighbors in adj_list.items():
#     for j in neighbors:
#         adj_matrix[i-1, j-1] = 1
#         adj_matrix[j-1, i-1] = 1  # ensure symmetry

# G = Graph(adj_matrix)

# G.show(title="First Journey EU")
