#import packages
from teaspoon.SP import network
from teaspoon.SP import tsa_tools
from teaspoon.SP import network_tools
from ripser import ripser
import numpy as np
import networkx as nx

# import sub modules
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def DistanceMatrix(A, method='shortest_unweighted_path'):
    """This function calculates the distance matrix from a connected graph represented as an adjacency matrix using an available method."

    Args:
        A (2-D array): 2-D square adjacency matrix
        method (str): Method for calculating distances between nodes. default is shortest_unweighted_path. Options are shortest_unweighted_path, shortest_weighted_path, weighted_shortest_path, and diffusion_distance.

    Returns:
        [2-D array]: Distance matrix between all node pairs.
    """

    methods = ['shortest_unweighted_path', 'shortest_weighted_path',
               'weighted_shortest_path', 'diffusion_distance']

    if method not in methods:
        print('Error: method listed for distance matrix not available.')
        print('Defaulting to unweighted shortest path.')
        method = 'shortest_unweighted_path'

    A = network_tools.remove_zeros(A)
    np.fill_diagonal(A, 0)
    A = A + A.T

    if method == 'shortest_unweighted_path':
        D = network_tools.shortest_path(A, weighted_path_lengths=False)

    if method == 'shortest_weighted_path':
        D = network_tools.shortest_path(A, weighted_path_lengths=True)

    if method == 'weighted_shortest_path':
        D = network_tools.weighted_shortest_path(A)

    if method == 'diffusion_distance':
        G = nx.from_numpy_array(A)
        diam = nx.algorithms.distance_measures.diameter(G)
        walk_steps = int(2*diam)
        # degree vector used later.
        deg_vec = network_tools.degree_vector(np.copy(A))
        D = network_tools.diffusion_distance(
            A, d=deg_vec, t=walk_steps, lazy=True)

    return D


def point_summaries(diagram, A):
    """This function calculates the persistent homology statistics for a graph from the paper "Persistent Homology of Complex Networks for Dynamic State Detection."

    Args:
        A (2-D array): 2-D square adjacency matrix
        diagram (list): persistence diagram from ripser from a graph's distance matrix

    Returns:
        [array 1-D]: statistics (R, En M) as (maximum persistence ratio, persistent entropy normalized, homology class ratio). Returns NaNs if empty diagram.
    """

    # assertion errors to check data types

    assert (len(A[0]) == len(A.T[0])), "A is not square adjacency matrix."
    assert (len(diagram) > 1), "Diagram should include atleast 0D and 1D persistene diagrams as a list of numpy.ndarrays."
    assert (type(diagram) is list), "Diagram should include atleast 0D and 1D persistene diagrams as a list of numpy.ndarrays."
    assert (type(
        diagram[0]) is np.ndarray), "Diagram should include atleast 0D and 1D persistene diagrams as a list of numpy.ndarrays."

    def persistentEntropy(lt):

        if len(lt) > 1:
            L = sum(lt)
            p = lt/L
            E = sum(-p*np.log2(p))
            Emax = np.log2(sum(lt))
            E = E/Emax
        if len(lt) == 1:
            E = 0
        if len(lt) == 0:
            E = 1
        return E

    if len(diagram[1]) > 0:

        # ------------Entropy---------------
        lt = np.array(diagram[1].T[1]-diagram[1].T[0])
        D1 = np.array([diagram[1].T[0], diagram[1].T[1]])
        D1 = D1
        lt = lt[lt < 10**10]
        num_lifetimes1 = len(lt)
        num_unique = len(A[0])
        En = persistentEntropy(lt)

        # ------------maximum persistence ratio---------------
        H1 = diagram[1].T
        delta = 0.1
        R = (1-np.nanmax(H1)/np.floor((num_unique/3)-delta))

        # ------------homology class ratio---------------
        M = num_lifetimes1/num_unique
        statistics = [R, En, M]
    else:
        statistics = [np.nan, np.nan, np.nan]

    return statistics


def PH_network(D, max_homology_dimension=1):
    """This function calculates the persistent homology of the graph represented by the adjacency matrix A using a distance algorithm defined by user.

    Args:
        D (2-D array): Distance matrix between all node pairs.

    Other Parameters:
        max_homology_dimension (Optional[int]): maximum dimension of the homology.

    Returns:
        [list]: list of lists where ech list is a persistence diagram (standard ripser format).
    """
    from scipy import sparse
    D_sparse = sparse.coo_matrix(D).tocsr()
    result = ripser(D_sparse, distance_matrix=True,
                    maxdim=max_homology_dimension)
    diagram = result['dgms']

    return diagram


# In[ ]:

# Only runs if running from this file (This will show basic example)
if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

    # ---------------------------Complex Example---------------------------------------

    # import needed packages
    import numpy as np
    from teaspoon.SP.network import ordinal_partition_graph
    from teaspoon.TDA.PHN import PH_network
    from teaspoon.SP.network_tools import make_network
    from teaspoon.parameter_selection.MsPE import MsPE_tau

    # generate a siple sinusoidal time series
    t = np.linspace(0, 30, 300)
    ts = np.sin(t) + np.sin(2*t)

    # Get appropriate dimension and delay parameters for permutations
    tau = int(MsPE_tau(ts))
    n = 5

    # create adjacency matrix, this
    A = ordinal_partition_graph(ts, n, tau)

    # get distance matrix
    D = DistanceMatrix(A, method='shortest_unweighted_path')

    # get networkx representation of network for plotting
    G, pos = make_network(A, position_iterations=1000,
                          remove_deg_zero_nodes=True)

    # calculate persistence diagram
    diagram = PH_network(D)

    print('1-D Persistent Homology (loops): ', diagram[1])

    stats = point_summaries(diagram, A)
    print('Persistent homology of network statistics: ', stats)

    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import networkx as nx

    TextSize = 14
    plt.figure(2)
    plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(4, 2)

    ax = plt.subplot(gs[0:2, 0:2])  # plot time series
    plt.title('Time Series', size=TextSize)
    plt.plot(ts, 'k')
    plt.xticks(size=TextSize)
    plt.yticks(size=TextSize)
    plt.xlabel('$t$', size=TextSize)
    plt.ylabel('$x(t)$', size=TextSize)
    plt.xlim(0, len(ts))

    ax = plt.subplot(gs[2:4, 0])
    plt.title('Network', size=TextSize)
    nx.draw(G, pos, with_labels=False, font_weight='bold', node_color='blue',
            width=1, font_size=10, node_size=30)

    ax = plt.subplot(gs[2:4, 1])
    plt.title('Persistence Diagram', size=TextSize)
    MS = 3
    if len(diagram[1]) > 0:
        top = max(diagram[1].T[1])
    else:
        top = 1
    plt.plot([0, top*1.25], [0, top*1.25], 'k--')
    plt.yticks(size=TextSize)
    plt.xticks(size=TextSize)
    plt.xlabel('Birth', size=TextSize)
    plt.ylabel('Death', size=TextSize)
    plt.plot(diagram[1].T[0], diagram[1].T[1], 'go', markersize=MS+2)
    plt.xlim(0, top*1.25)
    plt.ylim(0, top*1.25)

    plt.subplots_adjust(hspace=0.8)
    plt.subplots_adjust(wspace=0.35)
    plt.show()
