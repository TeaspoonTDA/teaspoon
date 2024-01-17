import numpy as np
import pandas as pd
import networkx as nx
from teaspoon.SP.tsa_tools import takens
from teaspoon.parameter_selection import MI_delay, MsPE, FNN_n
from teaspoon.SP.tsa_tools import permutation_sequence, cgss_sequence, k_NN
import os
import sys
import math
# get the coarse grained state space network represented as an adjacency matrix.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


# Gets Adjacency Matrix (weighted and direction) using permutation sequence

def Adjaceny_OP(perm_seq, n):  # adjacency matrix for ordinal partitions
    """This function takes the permutation sequence and creates a weighted and directed adjacency matrix.

    Args:
        perm_seq (array of ints): array of the permutations (chronologically ordered) that were visited.
        n (int): dimension of permutations.

    Returns:
        [A]: Adjacency matrix
    """
    N = math.factorial(n)  # number of possible nodes
    A = np.zeros((N, N))  # prepares A
    # go through all permutation transitions (This could be faster wiuthout for loop)
    for i in range(len(perm_seq)-1):
        # for each transition between permutations increment A_ij
        A[perm_seq[i]-1][perm_seq[i+1]-1] += 1
    return A  # this A is directional and weighted


def Adjacency_KNN(indices):  # adjacency matrix from k nearest neighbors
    """This function takes the indices of nearest neighbors and creates an adjacency matrix.

    Args:
        indices (array of ints): array of arrays of indices of n nearest neighbors.

    Returns:
        [A]: Adjacency matrix
    """
    A = np.zeros((len(indices.T[0]), len(indices.T[0])))
    for h in range(len(indices.T[0])):
        KNN_i = indices[h][indices[h] != h]  # indices of k nearest neighbors
        A[h][KNN_i] += 1  # increment A_ij for kNN indices
        A.T[h][KNN_i] += 1
    A[A > 0] = 1
    return A


# Coarse Grained State Space adjacency matrix
def Adjacency_CGSS(state_seq, N=None, delay=1):
    """This function takes the CGSS state sequence and creates a weighted and directed adjacency matrix.

    Args:
        state_seq (array of ints): array of the states visited in SSR (chronologically ordered).
        N (int): dnumber of total possible states.

    Returns:
        [A]: Adjacency matrix
    """
    import pandas as pd
    state_seq, uniques = pd.factorize(state_seq)
    if N == None:
        N = len(uniques)
    A = np.zeros((N, N))  # prepares A
    # go through all permutation transitions (This could be faster wiuthout for loop)
    for i in range(len(state_seq)-delay):
        # for each transition between permutations increment A_ij
        A[state_seq[i]][state_seq[i+delay]] += 1
    return A  # this A is directional and weighted


def knn_graph(ts, n=None, tau=None, k=4):
    """This function creates an k-NN network represented as an adjacency matrix A using a 1-D time series

    Args:
        ts (1-D array): 1-D time series signal

    Other Parameters:
        n (Optional[int]): embedding dimension for state space reconstruction. Default is uses FNN algorithm from parameter_selection module.
        tau (Optional[int]): embedding delay fro state space reconstruction. Default uses MI algorithm from parameter_selection module.
        k (Optional[int]): number of nearest neighbors for graph formation.

    Returns:
        [2-D square array]: A (2-D weighted and directed square adjacency matrix)
    """

    if tau == None:
        tau = MI_delay.MI_for_delay(
            ts, method='basic', h_method='sturge', k=2, ranking=True)
    if n == None:
        perc_FNN, n = FNN_n.FNN_n(ts, tau)

    ETS = takens(ts, n, tau)  # get embedded time series

    distances, indices = k_NN(ETS, k=k)
    # gets distances between embedded vectors and the indices of the nearest neighbors for every vector

    A = Adjacency_KNN(indices)  # get adjacency matrix (weighted, directional)

    return A


def ordinal_partition_graph(ts, n=None, tau=None):
    """This function creates an ordinal partition network represented as an adjacency matrix A using a 1-D time series

    Args:
        ts (1-D array): 1-D time series signal

    Other Parameters:
        n (Optional[int]): embedding dimension for state space reconstruction. Default is uses MsPE algorithm from parameter_selection module.
        tau (Optional[int]): embedding delay fro state space reconstruction. Default uses MsPE algorithm from parameter_selection module.

    Returns:
        [2-D square array]: A (2-D weighted and directed square adjacency matrix)
    """

    if n == None:
        tau = int(MsPE.MsPE_tau(ts))
        n = MsPE.MsPE_n(ts, tau)

    if tau == None:
        tau = int(MsPE.MsPE_tau(ts))

    PS = permutation_sequence(ts, n, tau)

    # gets adjacency matrix from permutation sequence transtitions
    A = Adjaceny_OP(PS, n)

    return A


def cgss_graph(ts, B_array, n=None, tau=None):
    """This function creates a coarse grained state space network (CGSSN) represented as an adjacency matrix A using a 1-D time series and binning array.

    Args:
        ts (1-D array): 1-D time series signal
        B_array (1-D array): array of bin edges for binning SSR for each dimension.

    Other Parameters:
        n (Optional[int]): embedding dimension for state space reconstruction. Default is uses MsPE algorithm from parameter_selection module.
        tau (Optional[int]): embedding delay fro state space reconstruction. Default uses MsPE algorithm from parameter_selection module.

    Returns:
        [2-D square array]: A (2-D weighted and directed square adjacency matrix)
    """

    if tau == None:
        tau = MI_delay.MI_for_delay(
            ts, method='basic', h_method='sturge', k=2, ranking=True)
    if n == None:
        perc_FNN, n = FNN_n.FNN_n(ts, tau)

    SSR = takens(ts, n, tau)

    symbol_seq = cgss_sequence(SSR, B_array)

    B = len(B_array) - 1
    n = len(SSR[0])
    N = B**n

    # get adjacency matrix from sequence
    A = Adjacency_CGSS(symbol_seq, N)

    return A


# In[ ]:

# Only runs if running from this file (This will show basic example)
if __name__ == "__main__":

    # import needed packages
    import numpy as np
    t = np.linspace(0, 30, 200)
    ts = np.sin(t) + np.sin(2*t)  # generate a simple time series

    A_knn = knn_graph(ts)  # ordinal partition network from time series

    A_op = ordinal_partition_graph(ts)  # knn network from time series
