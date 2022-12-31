from sklearn.neighbors import NearestNeighbors
from teaspoon.parameter_selection import MsPE
import matplotlib.pyplot as plt
from teaspoon.parameter_selection import FNN_n
from teaspoon.parameter_selection import MI_delay
import numpy as np
import itertools
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# In[ ]:

def cgss_binning(ts, n=None, tau=None, b=12, binning_method='equal_size', plot_binning=False):
    """This function generates the binning array applied to each dimension when constructing CGSSN.

    Args:
        ts (1-D array): 1-D time series signal

    Other Parameters:
        n (Optional[int]): embedding dimension for state space reconstruction. Default is uses MsPE algorithm from parameter_selection module.
        tau (Optional[int]): embedding delay from state space reconstruction. Default uses FNN algorithm from parameter_selection module.
        b (Optional[int]): Number of bins per dimension. Default is 12.
        plot_binning (Optional[bool]): Option to display plotting of binning over space occupied by SSR. default is False.
        binning_method (Optional[str]): Either equal_size or equal_probability for creating binning array. Default is equal_size.

    Returns:
        [1-D array]: One-dimensional array of bin edges.
    """
    B = b
    # ----------------Get embedding parameters if not defined----------------
    if tau == None:
        tau = MI_delay.MI_for_delay(
            ts, method='basic', h_method='sturge', k=2, ranking=True)
    if n == None:
        perc_FNN, n = FNN_n.FNN_n(ts, tau)

    def equalObs(x, B):  # define function to calculate equal-frequency bins based on interpolation
        return np.interp(np.linspace(0, len(x), B), np.arange(len(x)), np.sort(x))

    # get state space reconstruction from signal (SSR)
    from teaspoon.SP.tsa_tools import takens
    SSR = takens(ts, n, tau)

    # ----------------Define how to use the embedding----------------
    embedding = np.array(SSR).T

    # ----------------Define how to partition the embedding----------------
    if binning_method == 'equal_frequency':
        # define bins with equal-frequency or probability (approximately)
        B_array = equalObs(embedding.flatten(), B+1)
        B_array[-1] = B_array[-1]*(1 + 10**-10)
    if binning_method == 'equal_size':  # define bins based on equal spacing
        B_array = np.linspace(np.amin(embedding),
                              np.amax(embedding)*(1+10**-10), B+1)

    if plot_binning == True:
        x_max, x_min = np.max(B_array), np.min(B_array)
        plt.figure(0)
        plt.figure(figsize=(6, 6))

        TextSize = 17
        plt.xticks(size=TextSize)
        plt.yticks(size=TextSize)
        plt.xlabel('$x(t)$', size=TextSize)
        plt.ylabel(r'$x(t + \tau)$', size=TextSize)
        plt.scatter(SSR.T[0], SSR.T[1], c='k', marker='o',
                    alpha=0.2, label='$v_i$', s=10.)
        for b in B_array:
            plt.plot([x_max, x_min], [b, b], 'k--', linewidth=1)
            plt.plot([b, b], [x_max, x_min], 'k--', linewidth=1)
        plt.show()

    return B_array


def cgss_sequence(SSR, B_array):
    """This function generates the sequence of coarse-grained state space states from the state space reconstruction.

    Args:
        SSR (n-D array): n-dimensional state state space reconstruction using takens function from teaspoon.
        B_array (1-D array): binning array from function cgss_binning or self defined. Array of bin edges applied to each dimension of SSR for CGSS state sequence.

    Returns:
        [1-D array of integers]: array of coarse grained state space states represented as ints.
    """

    embedding = SSR.T

    B = len(B_array) - 1
    n = len(SSR[0])
    basis = B**(np.arange(n))  # basis for assigning symbolic value

    # ----------------digitize the embedding to a sequence----------------
    digitized_embedding = []  # prime the digitized version of deltas
    for e_i in embedding:  # nloop through n-1 delta positions
        # digitalize column delta_i
        digitzed_vector = np.digitize(e_i, bins=B_array)
        # append to digitalized deltas data structure
        digitized_embedding.append(digitzed_vector)
    # digitalize and stacked delta vectors
    digitized_embedding = np.array(digitized_embedding).T - 1
    # symbolic sequence from basis and digitzed embedding
    symbol_seq = np.sum(np.array(basis)*digitized_embedding, axis=1)

    return symbol_seq


# finds permutation sequency from modified pyentropy package
def permutation_sequence(ts, n=None, tau=None):
    """This function generates the sequence of permutations from a 1-D time series.

    Args:
        ts (1-D array): 1-D time series signal

    Other Parameters:
        n (Optional[int]): embedding dimension for state space reconstruction. Default is uses MsPE algorithm from parameter_selection module.
        tau (Optional[int]): embedding delay fro state space reconstruction. Default uses MsPE algorithm from parameter_selection module.

    Returns:
        [1-D array of intsegers]: array of permutations represented as int from [0, n!-1] from the time series.
    """

    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    time_series = ts

    if n == None:
        from teaspoon.parameter_selection import MsPE
        tau = int(MsPE.MsPE_tau(ts))
        n = MsPE.MsPE_n(ts, tau)

    if tau == None:
        from teaspoon.parameter_selection import MsPE
        tau = int(MsPE.MsPE_tau(ts))

    m, delay = n, tau

    import itertools
    import numpy as np

    def util_hash_term(perm):  # finds permutation type
        deg = len(perm)
        return sum([perm[k]*deg**k for k in range(deg)])
    L = len(time_series)  # total length of time series
    perm_order = []  # prepares permutation sequence array
    # prepares all possible permutations for comparison
    permutations = np.array(list(itertools.permutations(range(m))))
    hashlist = [util_hash_term(perm)
                for perm in permutations]  # prepares hashlist
    for i in range(L - delay * (m - 1)):
        # For all possible permutations in time series
        sorted_index_array = np.array(np.argsort(
            time_series[i:i + delay * m:delay], kind='quicksort'))
        # sort array for catagorization
        hashvalue = util_hash_term(sorted_index_array)
        # permutation type
        perm_order = np.append(
            perm_order, np.argwhere(hashlist == hashvalue)[0][0])
        # appends new permutation to end of array
    # sets permutation type as integer where $p_i \in \mathbb{z}_{>0}$
    perm_seq = perm_order.astype(int)+1
    return perm_seq  # returns sequence of permutations


# In[ ]:


def takens(ts, n=None, tau=None):
    """This function generates an array of n-dimensional arrays from a time-delay state-space reconstruction.

    Args:
        ts (1-D array): 1-D time series signal

    Other Parameters:
        n (Optional[int]): embedding dimension for state space reconstruction. Default is uses FNN algorithm from parameter_selection module.
        tau (Optional[int]): embedding delay fro state space reconstruction. Default uses MI algorithm from parameter_selection module.

    Returns:
        [arraay of n-dimensional arrays]: array of delyed embedded vectors of dimension n for state space reconstruction.
    """

    import numpy as np
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

    if tau == None:
        from teaspoon.parameter_selection import MI_delay
        tau = MI_delay.MI_for_delay(
            ts, method='basic', h_method='sturge', k=2, ranking=True)
    if n == None:
        from teaspoon.parameter_selection import FNN_n
        perc_FNN, n = FNN_n.FNN_n(ts, tau)

    # takens embedding method. Not the fastest algoriothm, but it works. Need to improve
    L = len(ts)  # total length of time series
    SSR = []
    for i in range(n):
        SSR.append(ts[i*tau:L-(n-i-1)*tau])
    SSR = np.array(SSR).T

    return np.array(SSR)


# In[ ]:


def k_NN(embedded_time_series, k=4):
    """This function gets the k nearest neighbors from an array of the state space reconstruction vectors

    Args:
        embedded_time_series (array of n-dimensional arrays): state space reconstructions vectors of dimension n. Can use takens function.

    Other Parameters:
        k (Optional[int]): number of nearest neighbors for graph formation. Default is 4.

    Returns:
        [distances, indices]: distances and indices of the k nearest neighbors for each vector.
    """

    ETS = embedded_time_series
    from sklearn.neighbors import NearestNeighbors
    # get nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(ETS)
    # get incidices of nearest neighbors
    distances, indices = nbrs.kneighbors(ETS)
    return distances, indices


# In[ ]:

# Only runs if running from this file (This will show basic example)
if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

    # ------------------------------------TAKENS-----------------------------------------

    import numpy as np
    t = np.linspace(0, 30, 200)
    ts = np.sin(t)  # generate a simple time series

    from teaspoon.SP.tsa_tools import takens
    embedded_ts = takens(ts, n=2, tau=10)

    import matplotlib.pyplot as plt
    plt.plot(embedded_ts.T[0], embedded_ts.T[1], 'k.')
    plt. show()

    # ------------------------------------PS-----------------------------------------

    import numpy as np
    t = np.linspace(0, 30, 200)
    ts = np.sin(t)  # generate a simple time series

    from teaspoon.SP.tsa_tools import permutation_sequence
    PS = permutation_sequence(ts, n=3, tau=12)

    import matplotlib.pyplot as plt
    plt.plot(t[:len(PS)], PS, 'k')
    plt. show()

    # ------------------------------------kNN-----------------------------------------

    import numpy as np
    t = np.linspace(0, 15, 100)
    ts = np.sin(t)  # generate a simple time series

    from teaspoon.SP.tsa_tools import takens
    embedded_ts = takens(ts, n=2, tau=10)

    from teaspoon.SP.tsa_tools import k_NN
    distances, indices = k_NN(embedded_ts, k=4)

    import matplotlib.pyplot as plt
    plt.plot(embedded_ts.T[0], embedded_ts.T[1], 'k.')
    i = 20  # choose arbitrary index to get NN of.
    NN = indices[i][1:]  # get nearest neighbors of point with that index.
    plt.plot(embedded_ts.T[0][NN], embedded_ts.T[1][NN], 'rs')  # plot NN
    plt.plot(embedded_ts.T[0][i], embedded_ts.T[1]
             [i], 'bd')  # plot point of interest
    plt. show()
