import numpy as np
import networkx as nx
import dionysus as dio
import gudhi
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def sliding_windows(window_width, overlap_ratio, time_range):
    """This function gets a list of sliding windows and their unions between.

    Args:
        window_width (float): width of sliding window.
        overlap_ratio (float): ratio of overlap between adjacent sliding windows.
        time_range (tuple): tuple of starting and ending time of sliding windows.

    Returns:
        [2xN array]: array of N intervals for each sliding window and union window interleaved.
    """
    w = window_width  # sliding window width
    # ranges from no overlap with 0 to 1 with complete overlap between adjacent windows.
    OR = overlap_ratio
    t_range = time_range  # range of time to create sliding windows from

    # normal windows without unions between
    normal_windows = np.array([np.arange(t_range[0], t_range[1], w*(1-OR)),
                               np.arange(t_range[0], t_range[1], w*(1-OR)) + w]).T
    # union windows as the union of time adjacent windows
    union_windows = np.array([[normal_windows[i][0], normal_windows[i+1][1]]
                              for i in range(len(normal_windows)-1)])

    # alternate between normal and union windows
    windows = [None]*(len(normal_windows)+len(union_windows))
    windows[::2] = normal_windows
    windows[1::2] = union_windows
    windows = np.array(windows)

    return windows


def graph_snapshots(edges, I, windows):
    """This function gets generates graph snapshots from the temporal graph information and windows.

    Args:
        edges (array of tuples): edges of graph.
        I (list of arrays): list of all intervals for each edge indexed according to edge list.
        windows (2xN array): array of N intervals for each sliding window and union window interleaved.

    Returns:
        [list of networkx Graphs]: list of networkx graphs for each graph snapshot corresponding to a window.
    """
    def getOverlapStatus(a, b):
        # function to check if two intervals a and b overlap based on allen's algebra
        a, b = np.sort(a), np.sort(b)
        overlap_status = False
        overlap = max(0, min(a[1], b[1]) - max(a[0], b[0]))
        if overlap > 0:  # check if there is an overlap
            overlap_status = True
        if a[0] == b[1] or b[0] == a[1]:  # get if end points overlap
            overlap_status = True
        return overlap_status

    G_snapshots = []  # initialize list of graph snapshots
    for window in windows:  # go through each window which is representative of a graph snapshot
        G_snapshot = nx.Graph()  # initialize snapshot graph
        # go through each edges intervals
        for i, intervals_edge in enumerate(I):
            for interval in intervals_edge:
                if getOverlapStatus(interval, window) == True:
                    G_snapshot.add_edge(edges[i][0], edges[i][1])
        G_snapshots.append(G_snapshot)
    return G_snapshots


def simplicial_complex_representation(TG, G_snapshots, windows, K=1):
    """This function represents the graph snapshots as vietoris rips simplicial complex based on a 
    filtration (short path) distance K. For the simplicial complex representaiton of the graph set K=1. 
    For higher order simplices choose $K$ correspondingly.

    Args:
        TG (graph): Networkx graph
        G_snapshots (list of networkx Graphs): list of networkx graphs for each graph snapshot corresponding to a window.
        windows (2xN array): array of N intervals for each sliding window and union window interleaved.
        K (int): shortest path distance filtration for simplicial complex formation.

    Returns:
        [list of lists]: S simplices of simplicial complex.
        [list of lists]: T times of simplices according to dionysus format for zigzag persistence.
    """
    t_Gs = [np.mean(window) for window in windows]
    G_uw = TG
    D = nx.floyd_warshall_numpy(G_uw)
    rips_complex = gudhi.RipsComplex(distance_matrix=D,
                                     max_edge_length=K)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=K-1)
    S_full = []
    for filtered_value in simplex_tree.get_filtration():
        S_full.append(tuple(filtered_value[0]))

    # initialize times when things turn on and off
    T = []
    for s in S_full:
        T.append([])
    # simplex status vector from previous graph
    status_vector_prev = [0]*len(S_full)
    ell = len(G_snapshots)
    for i in range(0, ell):  # go through each graph
        Gi = G_snapshots[i]
        # distance matrix using unweighted shortest path
        Di = nx.floyd_warshall_numpy(Gi)
        rips_complex = gudhi.RipsComplex(distance_matrix=Di, max_edge_length=K)
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=K-1)
        S = []
        for filtered_value in simplex_tree.get_filtration():
            S.append(tuple(filtered_value[0]))
        # simplex status vector for when simplices are on.
        status_vector = [0]*len(S_full)
        for s in S:  # go through each simplex in simplices of graph
            s_index = S_full.index(s)  # get index in S_full where s is.
            # set simplex as on if it is in simplicial complex
            status_vector[s_index] = 1

        for s_index in range(len(status_vector)):
            # if the simplex appears
            if status_vector[s_index] == 1 and status_vector_prev[s_index] == 0:
                T[s_index].append(t_Gs[i])
            # if the simplex dissapeared.
            if status_vector[s_index] == 0 and status_vector_prev[s_index] == 1:
                T[s_index].append(t_Gs[i])
        status_vector_prev = np.copy(status_vector)

    # make sure all persistence pairs die at end and dont go to infinity
    t_final = windows[-1][-1]
    for i, T_i in enumerate(T):
        if len(T_i) % 2 == 1:
            T[i].append(t_final)

    S = []
    for i, s in enumerate(S_full):
        ell_S, simp = len(s), []
        for j in range(ell_S):
            simp.append(s[j])
        S.append(simp)
    return S, T


def plot_persistence_diagram(dgms, windows, dimensions=[0, 1], FS=18):
    """This function plots the persistence diagram from the zigzag persistence filtration.

    Args:
        dgms (persistence diagrams): persistence diagrams from dionysus2 format output.
        dimensions (array of integers): dimensions to be included in persistence diagram.
        windows (2xN array): array of N intervals for each sliding window and union window interleaved.
        FS (int): font size.

    Returns:
        [NA]: Does not return persistence diagram.
    """
    plt.figure(figsize=(5, 5))
    fontsize = FS
    MS = 6
    ts = [np.mean(window) for window in windows]
    max_val = max(ts)
    for i, dgm in enumerate(dgms):
        plot_markers = ['gs', 'bo', 'rd']
        for j, p in enumerate(dgm):
            # this is really stupid but the dionysus point p is not a subsctiptable object.
            pair = (str(p)[1:-1]).split(",")
            pair = [float(pair[0]), float(pair[1])]
            if pair[1] == np.inf:
                pair[1] = max_val
            if i in dimensions:
                if j == 0:
                    plt.plot([pair[0]], [pair[1]], plot_markers[i],
                             markersize=MS, label="$H_"+str(i)+"$", alpha=0.5)
                else:
                    plt.plot([pair[0]], [pair[1]], plot_markers[i],
                             markersize=MS, alpha=0.5)
    ts_range = max(ts) - min(ts)
    plt.plot([-0.5, 10.5],
             [-0.5, 10.5], 'k--')
    plt.xlabel('Birth', fontsize=fontsize)
    plt.ylabel('Death', fontsize=fontsize)
    plt.grid()
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    plt.xlim(min(ts) - 0.05*ts_range, max(ts) + 0.05*ts_range)
    plt.ylim(-0.5, 10.5)
    plt.legend(loc='lower right', fontsize=fontsize)
    plt.tight_layout()
    plt.show()


# In[ ]:
if __name__ == "__main__":

    import numpy as np
    import networkx as nx
    import dionysus as dio
    import gudhi
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    # list of edges and nodes
    nodes = [0, 1, 2, 3, 4]
    edges = [(0, 1),  # edge 0
             (1, 2),  # edge 1
             (2, 3),  # edge 2
             (3, 4),  # edge 3
             (4, 0)]  # edge 4

    # I is a list of intervals for each edge.
    I = [[[0.1, 2.3], [3.6, 8.7]],  # edge 0
         [[4.4, 9.9]],             # edge 1
         [[2.1, 3.9], [6.1, 9.1]],  # edge 2
         [[1.3, 9.3]],             # edge 3
         [[3.2, 7.6]]]             # edge 4

    # positions of each node (for plotting purposes only)
    node_positions = [[0.0, 0.2],  # node 0
                      [0.0, 1.0],  # node 1
                      [1.0, 1.0],  # node 2
                      [1.0, 0.2],  # node 3
                      [0.5, -0.5]]  # node 4

    TG = nx.Graph()  # temporal graph as attributed networkx representation
    # add attribute information to each node and edge and construct as networkx graph
    for i, node in enumerate(nodes):  # add position information to nodes
        x, y = node_positions[i][0], node_positions[i][1]
        TG.add_node(node, position=(x, y))
    for i, edge in enumerate(edges):  # add time information to edges
        TG.add_edge(edge[0], edge[1], intervals=I[i])

    # define sliding window intervals
    window_width, overlap_ratio = 1.0, 0.0
    time_range = [0, 10]
    windows = sliding_windows(window_width, overlap_ratio, time_range)

    # get graph snapshots
    G_snapshots = graph_snapshots(edges, I, windows)

    # get simplices and associated times based on dionysus format needed.
    S, T = simplicial_complex_representation(TG, G_snapshots, windows, K=1)

    # run dionysus2 zigzag persistence
    f = dio.Filtration(S)
    zz, dgms, cells = dio.zigzag_homology_persistence(f, T)

    # print and plot persistence
    for i, dgm in enumerate(dgms):
        print("Dimension:", i)
        for j, p in enumerate(dgm):
            print(p)
    diag = plot_persistence_diagram(dgms, windows, dimensions=[0, 1], FS=22)
