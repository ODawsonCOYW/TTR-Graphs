import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt 

# Number of vertices
n = 15

# Initialize an n×n matrix of zeros
adj_matrix = np.zeros((n, n), dtype=int)

# Adjacency list (1-indexed)
adj_list = {
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

# Fill the adjacency matrix (undirected graph)
for i, neighbors in adj_list.items():
    for j in neighbors:
        adj_matrix[i-1, j-1] = 1
        adj_matrix[j-1, i-1] = 1  # symmetric

# Print the adjacency matrix
# print(adj_matrix)

eigenvalues, eigenvectors = LA.eig(adj_matrix)

# print(eigenvalues)
# print(eigenvectors)

degrees = [len(adj_list[i]) for i in sorted(adj_list)]

norm_degrees = [d / (len(degrees) - 1) for d in degrees]

plt.hist(degrees)
plt.xlabel("Degree")
plt.ylabel("Frequency")
plt.title("Histogram of Node Degrees (New York)")
plt.show()

plt.hist(norm_degrees)
plt.xlabel("Degree")
plt.ylabel("Frequency")
plt.title("Histogram of Normalised Node Degrees (New York)")
plt.show()

# #Sage 

# # Number of vertices
# n = 15

# # Initialize n×n zero matrix over integers
# A = matrix(ZZ, n, n, 0)

# # Adjacency list (1-indexed)
# adj = {
#     1: [2, 3, 4],
#     2: [1, 4, 5],
#     3: [1, 4, 6, 7],
#     4: [1, 2, 3, 5, 6],
#     5: [2, 4, 6, 8],
#     6: [3, 4, 5, 7, 8],
#     7: [3, 6, 8, 9, 11],
#     8: [5, 6, 7, 9, 10],
#     9: [7, 8, 10, 11, 12, 13],
#     10: [8, 9, 12],
#     11: [7, 9, 14],
#     12: [9, 10, 13, 15],
#     13: [15, 14, 12, 9],
#     14: [11, 13, 15],
#     15: [14, 13, 12]
# }

# # Fill the adjacency matrix (undirected)
# for i, neighbors in adj.items():
#     for j in neighbors:
#         A[i-1, j-1] = 1
#         A[j-1, i-1] = 1  # symmetric for undirected graph

# # Display the adjacency matrix
# A


