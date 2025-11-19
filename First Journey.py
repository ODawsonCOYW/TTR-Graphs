import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

# Number of vertices
n = 19

# Initialize an n×n matrix of zeros
adj_matrix = np.zeros((n, n), dtype=int)

# Adjacency list (1-indexed)
adj_list = {
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

# Fill the adjacency matrix (undirected graph)
for i, neighbors in adj_list.items():
    for j in neighbors:
        adj_matrix[i-1, j-1] = 1
        adj_matrix[j-1, i-1] = 1  # symmetric

# Print the adjacency matrix
# print(adj_matrix)

# eigenvalues, eigenvectors = LA.eig(adj_matrix)

# print(eigenvalues)

degrees = [len(adj_list[i]) for i in sorted(adj_list)]

norm_degrees = [d / (len(degrees) - 1) for d in degrees]

plt.hist(degrees)
plt.xlabel("Degree")
plt.ylabel("Frequency")
plt.title("Histogram of Node Degrees (FJ USA)")
plt.show()

plt.hist(norm_degrees)
plt.xlabel("Degree")
plt.ylabel("Frequency")
plt.title("Histogram of Normalised Node Degrees (FJ USA)")
plt.show()


# # # Sage Math Code

# # Number of vertices
# n = 19

# # Initialize n×n zero matrix over integers
# A = matrix(ZZ, n, n, 0)

# # Adjacency list (1-indexed)
# adj = {
#     1: [2, 3, 9, 10],
#     2: [1, 3, 4],
#     3: [1, 2, 4, 10, 11],
#     4: [2, 3, 5],
#     5: [4, 6, 8, 11, 12],
#     6: [5, 7, 8],
#     7: [6, 8, 14],
#     8: [5, 6, 7, 12, 13, 14],
#     9: [1, 15, 10],
#     10: [1, 3, 9, 11, 15, 16],
#     11: [3, 5, 12, 16, 10],
#     12: [5, 8, 11, 17, 13],
#     13: [18, 19, 14, 12, 8, 17],
#     14: [7, 8, 13, 19],
#     15: [9, 10, 16],
#     16: [10, 11, 15, 17],
#     17: [16, 18, 13, 12],
#     18: [17, 13, 19],
#     19: [18, 13, 14]
# }
