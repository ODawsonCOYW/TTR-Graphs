import networkx as nx
from Graph_Data import adj_list_USA, adj_list_USA_W
import math
import matplotlib.pyplot as plt

G = nx.Graph(adj_list_USA)
pos = nx.planar_layout(G)

def make_rotation_system(adj, pos):
    rot = {}
    for v, nbrs in adj.items():
        x0, y0 = pos[v]
        # Sort neighbors counter-clockwise around v
        rot[v] = sorted(nbrs, key=lambda u: math.atan2(pos[u][1]-y0, pos[u][0]-x0))
    return rot

def find_faces(rot):
    # Create set of directed edges
    half_edges = {(u, v) for u in rot for v in rot[u]}
    visited = set()
    faces = []

    def next_edge(u, v):
        nbrs = rot[v]
        i = nbrs.index(u)
        w = nbrs[(i - 1) % len(nbrs)]  # previous neighbor in CCW order
        return (v, w)

    for h in half_edges:
        if h in visited:
            continue
        face = []
        curr = h
        while curr not in visited:
            visited.add(curr)
            u, v = curr
            face.append(u)
            curr = next_edge(u, v)
        faces.append(face)
    return faces

def face_weight(face, edge_weights):
    total = 0
    n = len(face)
    for i in range(n):
        u = face[i]
        v = face[(i+1) % n]  # wrap around to form a cycle
        total += edge_weights[frozenset([u,v])]
    return total

def face_info(adj, adj_W):
    
    # Returns a dictionary of tuples (Total Face Weight, Face Degree)

    rot_adj = make_rotation_system(adj, pos)
    faces = find_faces(rot_adj)

    edge_weights = {}
    for u, nbrs in adj_W.items():
        for v, w in nbrs:
            edge_weights[frozenset([u,v])] = w

    # Compute weights for all faces
    face_weights = {}
    for i, f in enumerate(faces, 1):
        face_weights[i] = (face_weight(f, edge_weights), len(f))

    return face_weights

face_info_USA = face_info(adj_list_USA, adj_list_USA_W)

xs = []
ys = []

for u,v in face_info_USA.values():
    xs.append(v)
    ys.append(u)
    
xs.remove(max(xs))
ys.remove(max(ys))
    
# Create scatter plot
plt.scatter(xs, ys, color='blue', marker='o')  # optional: color and marker style

# Add labels and title
plt.xlabel("Face Degree")
plt.ylabel("Face Weight")
plt.title("How does Face Degree Affect Total Face Weight")

# Show the plot
plt.show()
    
