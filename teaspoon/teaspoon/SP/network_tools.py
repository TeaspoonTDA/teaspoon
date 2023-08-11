from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from numpy.linalg import inv
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# In[ ]: distance defining functions for graphs


def shortest_path(A, weighted_path_lengths=False):
    """This function calculates the shortest path distance matrix for a signal component graph represented as an adjacency matrix A.

    Args:
        A (2-D array): adjacency matrix of single component graph.

    Other Parameters:
        n (Optional[int]): option to calculate the shortest path between nodes using the inverse of the edge weights.

    Returns:
        [2-D array]: Distance matrix of shortest path lengths between all node pairs.
    """

    A_sp = np.copy(A)
    N = len(A_sp)
    D = np.zeros((N, N))

    if weighted_path_lengths == False:
        A_sp[A_sp > 0] = 1
        G = nx.from_numpy_array(A_sp)
        lengths = dict(nx.all_pairs_shortest_path_length(G))
        for i in range(N):
            for j in range(N):
                D[i][j] = lengths[i][j]
        return D

    else:
        # initiate graph
        np.seterr(divide='ignore')  # ignores divide by zero warning
        np.fill_diagonal(A_sp, 0)  # set diagonal to zero
        A_inv = 1/A_sp  # take element-wise inverse
        G = nx.from_numpy_array(A_inv)
        paths = dict(nx.all_pairs_dijkstra(G))
        for i in range(N):
            for j in range(N):
                D[i][j] = len(paths[i][1][j])-1
        return D


def weighted_path_dist(path, A):
    path_weight = 0
    for n in range(len(path)-1):
        edge_weight = A[path[n]][path[n+1]]
        path_weight += edge_weight
    return path_weight


def weighted_shortest_path(A):
    """This function calculates the weighted shortest path distance matrix for a signal component graph represented as an adjacency matrix A. The paths are found tusing the edge weights as the inverse of the adjacency matrix weight and the distance is the sum of the adjacency matrix weights along that path.

    Args:
        A (2-D array): adjacency matrix of single component graph.

    Returns:
        [2-D array]: Distance matrix of shortest path lengths between all node pairs.
    """

    # initiate graph
    np.seterr(divide='ignore')  # ignores divide by zero warning
    np.fill_diagonal(A, 0)  # set diagonal to zero
    A_inv = 1/A  # take element-wise inverse
    N = len(A_inv)
    D = np.zeros((N, N))
    G = nx.from_numpy_array(A_inv)
    paths = dict(nx.all_pairs_dijkstra(G))
    for i in range(N):
        for j in range(N):
            path = paths[i][1][j]
            D[i][j] = weighted_path_dist(path, A)
    return D


def degree_matrix(A):
    """
    This function calculates the degree matrix from theadjacency matrix. A degree matrix is an empty matrix with diagonal filled with degree vector.

    Args:
        A (2-D array): adjacency matrix of single component graph.

    Returns:
        [2-D array]: Degree matrix.
    """
    n = len(A)  # get degree of graph
    d = np.sum(A, axis=1)  # get degree vector from graph
    D = np.zeros((n, n))  # initialize degree matrix as zeros
    np.fill_diagonal(D, d)  # fill diagonal with degree vector
    return D


def degree_vector(A):
    A_unweighted = np.copy(A)
    A_unweighted = A_unweighted + A_unweighted.T
    A_unweighted[A_unweighted > 0] = 1
    # define unweighted adjacency matrix to calculate degree sequence
    np.fill_diagonal(A_unweighted, 0)
    deg_vec = np.sum(A_unweighted, axis=1)  # degree sequence
    return deg_vec


def random_walk(A, lazy=False):
    """This function calculates the 1 step random walk probability matrix using the adjacency matrix.

    Args:
        A (matrix): adjacency matrix of graph.

    Other Parameters:
        lazy (Optional[boolean]): optional to use self transition probability (50%).

    Returns:
        [2-D square array]: A (2-D weighted and directed square probability transition matrix)
    """
    N = len(A)

    W = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            W[i][j] = A[i][j]/np.sum(A[i])

    P = W.T  # notation for W and P might be switched?

    if lazy == True:
        P = 0.5*(np.eye(N) + P)

    return P


def diffusion_distance(A, d, t, lazy=False):
    """This function calculates the diffusion distance for t random walk steps.

    Args:
        A (matrix): adjacency matrix of graph.
        d (array): degree vector
        t (int): random walk steps. Should be approximately twice the diameter of graph.

    Other Parameters:
        lazy (Optional[boolean]): optional to use self transition probability (50%).

    Returns:
        [2-D square array]: A (2-D weighted and directed square probability transition matrix)
    """
    W = random_walk(A, lazy)
    P = W.T
    Pt = np.linalg.matrix_power(P, t)
    N = len(A)  # number of vertices
    D = np.zeros((N, N))  # initialize distance matrix
    for a in np.arange(N):
        for b in np.arange(N):
            # see equation on top of page 118 in lecture 21 notes.
            D[a][b] = np.sum((1/d)*(Pt[a] - Pt[b])**2)**0.5

    return D


# In[ ]: forming adjacency matrix functions

def remove_zeros(A):
    """This function removes unused vertices from adjacency matrix.

    Args:
        A (matrix): 2-D weighted and directed square probability transition matrix

    Returns: [2D matrix]: 2-D weighted and directed square probability transition matrix
    """
    # removes unused rows/columns from adjacency matrix for unused vertices.
    import pandas as pd

    degr_seq = degree_vector(A).astype(int)
    to_drop = np.argwhere(degr_seq == 0).T[0]
    A_mod = pd.DataFrame(A).drop(to_drop).drop(to_drop, axis=1).to_numpy()
    return A_mod


# In[ ]: visualization tools


def visualize_network(A, position_iterations=1000, remove_deg_zero_nodes=False):
    """This function creates a networkx graph and position of the nodes using the adjacency matrix.

    Args:
        A (2-D array): 2-D square adjacency matrix

    Other Parameters:
        position_iterations (Optional[int]): Number of spring layout position iterations. Default is 1000.

    Returns:
        [dictionaries, list]: G (networkx graph representation), pos (position of nodes for networkx drawing)
    """

    A = A + A.T  # make undirected adjacency matrix
    np.fill_diagonal(A, 0)  # get rid of diagonal
    A[A > 0] = 1  # make unweighted

    G = nx.Graph()
    G.add_nodes_from(range(len(A[0])))

    edges1 = []
    edges2 = []
    for h in range(0, len(A[0])):
        edges1 = np.append(edges1, np.nonzero(A[h]))
        L = len(np.nonzero(A[h])[0])
        edges2 = np.append(edges2, np.zeros(L) + h)
    edges1 = edges1.astype(int)+1
    edges2 = edges2.astype(int)+1
    edges = zip(edges1, edges2)
    G.add_edges_from(edges)

    fixed_nodes = []
    for i in range(0, len(A[0])):
        if G.degree[i] == 0:
            fixed_nodes.append(i)

    pos = nx.spring_layout(G, iterations=0)
    pos = nx.spring_layout(G, pos=pos, fixed=fixed_nodes,
                           iterations=position_iterations)

    if remove_deg_zero_nodes == False:
        if 0 in G.nodes():  # removes this node because it shouldn't be in graph
            G.remove_node(0)
    else:
        for i in range(0, len(A[0])):
            if G.degree[i] == 0:
                G.remove_node(i)

    return G, pos


def make_network(A, position_iterations=1000, remove_deg_zero_nodes=False):
    """This function creates a networkx graph and position of the nodes using the adjacency matrix.

    Args:
        A (2-D array): 2-D square adjacency matrix

    Other Parameters:
        position_iterations (Optional[int]): Number of spring layout position iterations. Default is 1000.

    Returns:
        [dictionaries, list]: G (networkx graph representation), pos (position of nodes for networkx drawing)
    """

    import numpy as np
    A = A + A.T  # make undirected adjacency matrix
    np.fill_diagonal(A, 0)  # get rid of diagonal
    A[A > 0] = 1  # make unweighted

    import networkx as nx
    G = nx.Graph()
    G.add_nodes_from(range(len(A[0])))

    edges1 = []
    edges2 = []
    for h in range(0, len(A[0])):
        edges1 = np.append(edges1, np.nonzero(A[h]))
        L = len(np.nonzero(A[h])[0])
        edges2 = np.append(edges2, np.zeros(L) + h)
    edges1 = edges1.astype(int)+1
    edges2 = edges2.astype(int)+1
    edges = zip(edges1, edges2)
    G.add_edges_from(edges)

    fixed_nodes = []
    for i in range(0, len(A[0])):
        if G.degree[i] == 0:
            fixed_nodes.append(i)

    pos = nx.spring_layout(G, iterations=0)
    pos = nx.spring_layout(G, pos=pos, fixed=fixed_nodes,
                           iterations=position_iterations)

    if remove_deg_zero_nodes == False:
        if 0 in G.nodes():  # removes this node because it shouldn't be in graph
            G.remove_node(0)
    else:
        for i in range(0, len(A[0])):
            if G.degree[i] == 0:
                G.remove_node(i)

    return G, pos


# In[ ]:

# Only runs if running from this file (This will show basic example)
if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

    import numpy as np
    t = np.linspace(0, 30, 200)
    ts = np.sin(t) + np.sin(2*t)  # generate a simple time series

    from teaspoon.SP.network import knn_graph
    A = knn_graph(ts)

    from teaspoon.SP.network_tools import make_network
    G, pos = make_network(A)

    import matplotlib.pyplot as plt
    import networkx as nx
    plt.figure(figsize=(8, 8))
    plt.title('Network', size=16)
    nx.draw(G, pos, with_labels=False, font_weight='bold', node_color='blue',
            width=1, font_size=10, node_size=30)
    plt.show()
