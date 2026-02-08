import networkx as nx
from Graph_Data import Graph, adj_list_EU, adj_list_Am, adj_list_FJ, adj_list_FJE, adj_list_London, routes_Germany
from Graph_Data import adj_list_NY, adj_list_NY_W, adj_list_USA, adj_NL, adj_list_GER, adj_list_OW, adj_list_HoA
from Graph_Data import Europe_Route_Freq, adj_list_PEN, routes_EU, routes_USA, routes_NY, routes_PEN, routes_London
from Graph_Data import routes_India, adj_list_India, adj_list_USA_W, adj_list_EU_W, adj_list_PEN_W, adj_list_India_W
from Graph_Data import adj_list_London_W, adj_list_GER_W
import math
import matplotlib.pyplot as plt
from collections import Counter

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

def face_degrees(faces, remove_outer = True):
    """
    faces: list of faces, where each face is a list of vertices
    returns: list of face degrees
    """
    
    degrees = [len(face) for face in faces]
    
    if remove_outer and degrees:
        degrees.remove(max(degrees))
    
    return degrees


def plot_face_degree_histogram(faces, title="Face Degree Distribution"):
    """
    Plots a histogram of face degrees using matplotlib.pyplot
    """
    degrees = face_degrees(faces)

    # Count frequencies (useful for discrete histogram)
    counts = Counter(degrees)
    xs = sorted(counts.keys())
    ys = [counts[k] for k in xs]

    plt.figure()
    plt.bar(xs, ys)
    plt.xlabel("Face degree (number of edges)")
    plt.ylabel("Number of faces")
    plt.title(title)
    plt.xticks(xs)
    plt.show()
    
    
G = nx.Graph(adj_NL)
pos = nx.planar_layout(G)
    
rot_London = make_rotation_system(adj_NL, pos)

face_list = find_faces(rot_London)
    
plot_face_degree_histogram(face_list, "Face Degree Distribution (Nederlands)")

    


def face_weight(face, edge_weights):
    total = 0
    n = len(face)
    for i in range(n):
        u = face[i]
        v = face[(i+1) % n]  # wrap around to form a cycle
        total += edge_weights[frozenset([u,v])]
    return total

def face_info(adj, adj_W, pos):
    
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

def face_plots(adj, adj_W, name):   
    G = nx.Graph(adj)
    pos = nx.planar_layout(G)
    
    face_stuff = face_info(adj, adj_W, pos)
    
    print(face_stuff)
    
    xs = []
    ys = []
    
    for u,v in face_stuff.values():
        xs.append(v)
        ys.append(u)
        
    xs.remove(max(xs))
    ys.remove(max(ys))
        
    # Create scatter plot
    plt.scatter(xs, ys, color='blue', marker='o')  # optional: color and marker style
    
    # Add labels and title
    plt.xlabel("Face Degree")
    plt.ylabel("Face Weight")
    plt.title(f"How does Face Degree Affect Total Face Weight ({name})")
    
    # Show the plot
    plt.show()
    
def face_degree_counts(faces):
    # Find the maximum face degree
    max_deg = max(len(face) for face in faces)
    
    counts = []
    for d in range(3, max_deg + 1):
        c = sum(1 for face in faces if len(face) == d)
        counts.append(c)
    return counts

