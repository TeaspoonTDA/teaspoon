
def Adjaceny_OP(perm_seq, n): #Gets Adjacency Matrix (weighted and direction) using permutation sequence
    import numpy as np
    N = np.math.factorial(n) #number of possible nodes
    A = np.zeros((N,N)) #prepares A
    for i in range(len(perm_seq)-1): #go through all permutation transitions (This could be faster wiuthout for loop)
        A[perm_seq[i]-1][perm_seq[i+1]-1] += 1 #for each transition between permutations increment A_ij
    return A #this A is directional and weighted


def Adjacency_KNN(indices):
    import numpy as np
    A = np.zeros((len(indices.T[0]),len(indices.T[0])))
    for h in range(len(indices.T[0])):
        KNN_i = indices[h][indices[h]!=h] #indices of k nearest neighbors
        A[h][KNN_i]+=1 #increment A_ij for kNN indices
        A.T[h][KNN_i]+=1
    A[A>0] = 1
    return A


def knn_graph(ts, n = None, tau = None, k = 4):
    
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
    
    #import sub modules
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__),'..','..'))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__),'..'))
    from teaspoon.SP import tsa_tools
    
    if tau == None:
        from parameter_selection import MI_delay
        tau = MI_delay.MI_for_delay(ts, method = 'basic', h_method = 'sturge', k = 2, ranking = True)
    if n == None:
        from parameter_selection import FNN_n
        perc_FNN, n = FNN_n.FNN_n(ts, tau)
    
    ETS = tsa_tools.takens(ts, n, tau) #get embedded time series
    
    distances, indices = tsa_tools.k_NN(ETS, k= k) 
    #gets distances between embedded vectors and the indices of the nearest neighbors for every vector
        
    A = Adjacency_KNN(indices) #get adjacency matrix (weighted, directional)
    
    return A


def ordinal_partition_graph(ts, n = None, tau = None):
    
    """This function creates an ordinal partition network represented as an adjacency matrix A using a 1-D time series
    
    Args:
        ts (1-D array): 1-D time series signal
    
    Other Parameters:
        n (Optional[int]): embedding dimension for state space reconstruction. Default is uses MsPE algorithm from parameter_selection module.
        tau (Optional[int]): embedding delay fro state space reconstruction. Default uses MsPE algorithm from parameter_selection module.
        k (Optional[int]): number of nearest neighbors for graph formation.
        
    Returns:
        [2-D square array]: A (2-D weighted and directed square adjacency matrix)
    """
    
    #import sub modules
    import os
    import sys
    
    sys.path.insert(0, os.path.join(os.path.dirname(__file__),'..'))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__),'..','..'))
    from teaspoon.SP.tsa_tools import permutation_sequence
    
    if n == None:
        from parameter_selection import MsPE
        tau = int(MsPE.MsPE_tau(ts)) 
        n = MsPE.MsPE_n(ts, tau)
        
    if tau == None:
        from parameter_selection import MsPE
        tau = int(MsPE.MsPE_tau(ts)) 
    
    PS = permutation_sequence(ts, n, tau)
        
    A = Adjaceny_OP(PS, n) #gets adjacency matrix from permutation sequence transtitions
    
    return A
    

    
# In[ ]:
    
if __name__ == "__main__": #Only runs if running from this file (This will show basic example)

    #import needed packages
    import numpy as np
    t = np.linspace(0,30,200)
    ts = np.sin(t) + np.sin(2*t) #generate a simple time series
    
    A_knn = knn_graph(ts) #ordinal partition network from time series
    
    A_op = ordinal_partition_graph(ts) #knn network from time series
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    