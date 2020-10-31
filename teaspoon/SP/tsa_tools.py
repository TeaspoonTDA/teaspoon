

# In[ ]:
    


def permutation_sequence(ts, n = None, tau = None): #finds permutation sequency from modified pyentropy package

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
    sys.path.insert(0, os.path.join(os.path.dirname(__file__),'..','..'))
    time_series = ts
    
    if n == None:
        from teaaspoon.parameter_selection import MsPE
        tau = int(MsPE.MsPE_tau(ts)) 
        n = MsPE.MsPE_n(ts, tau)
        
    if tau == None:
        from teaaspoon.parameter_selection import MsPE
        tau = int(MsPE.MsPE_tau(ts))
    
    m, delay = n, tau
    
    import itertools
    import numpy as np
    def util_hash_term(perm): #finds permutation type
        deg = len(perm)
        return sum([perm[k]*deg**k for k in range(deg)])
    L = len(time_series) #total length of time series
    perm_order = [] #prepares permutation sequence array
    permutations = np.array(list(itertools.permutations(range(m)))) #prepares all possible permutations for comparison
    hashlist = [util_hash_term(perm) for perm in permutations] #prepares hashlist
    for i in range(L - delay * (m - 1)): 
    #For all possible permutations in time series
        sorted_index_array = np.array(np.argsort(time_series[i:i + delay * m:delay], kind='quicksort')) 
        #sort array for catagorization
        hashvalue = util_hash_term(sorted_index_array);
        #permutation type
        perm_order = np.append(perm_order, np.argwhere(hashlist == hashvalue)[0][0])
        #appends new permutation to end of array
    perm_seq = perm_order.astype(int)+1 #sets permutation type as integer where $p_i \in \mathbb{z}_{>0}$
    return perm_seq #returns sequence of permutations



# In[ ]:
    

def takens(ts, n= None, tau= None):
    
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
    sys.path.insert(0, os.path.join(os.path.dirname(__file__),'..','..'))
    
    if tau == None:
        from teaspoon.parameter_selection import MI_delay
        tau = MI_delay.MI_for_delay(ts, method = 'basic', h_method = 'sturge', k = 2, ranking = True)
    if n == None:
        from teaspoon.parameter_selection import FNN_n
        perc_FNN, n = FNN_n.FNN_n(ts, tau)
    
    #takens embedding method. Not the fastest algoriothm, but it works. Need to improve
    L = len(ts) #total length of time series
    SSR = [] 
    for i in range(L - tau * (n - 1)): 
        v_i = ts[i:i + tau * n:tau]
        SSR.append(v_i)
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
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(ETS) #get nearest neighbors
    distances, indices = nbrs.kneighbors(ETS) #get incidices of nearest neighbors
    return distances, indices




# In[ ]:
    
if __name__ == "__main__": #Only runs if running from this file (This will show basic example)
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__),'..', '..'))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__),'..'))

    #------------------------------------TAKENS-----------------------------------------
    
    import numpy as np
    t = np.linspace(0,30,200)
    ts = np.sin(t)  #generate a simple time series
    
    from teaspoon.SP.tsa_tools import takens
    embedded_ts = takens(ts, n = 2, tau = 10)
    
    import matplotlib.pyplot as plt
    plt.plot(embedded_ts.T[0], embedded_ts.T[1], 'k.')
    plt. show()
    
    
    
    #------------------------------------PS-----------------------------------------
    
    import numpy as np
    t = np.linspace(0,30,200)
    ts = np.sin(t)  #generate a simple time series
    
    from teaspoon.SP.tsa_tools import permutation_sequence
    PS = permutation_sequence(ts, n = 3, tau = 12)
    
    import matplotlib.pyplot as plt
    plt.plot(t[:len(PS)], PS, 'k')
    plt. show()
    
    
    
    
    #------------------------------------kNN-----------------------------------------
    
    import numpy as np
    t = np.linspace(0,15,100)
    ts = np.sin(t)  #generate a simple time series
    
    from teaspoon.SP.tsa_tools import takens
    embedded_ts = takens(ts, n = 2, tau = 10)
    
    from teaspoon.SP.tsa_tools import k_NN
    distances, indices = k_NN(embedded_ts, k=4)
    
    
    import matplotlib.pyplot as plt
    plt.plot(embedded_ts.T[0], embedded_ts.T[1], 'k.')
    i = 20 #choose arbitrary index to get NN of.
    NN = indices[i][1:] #get nearest neighbors of point with that index.
    plt.plot(embedded_ts.T[0][NN], embedded_ts.T[1][NN], 'rs') #plot NN
    plt.plot(embedded_ts.T[0][i], embedded_ts.T[1][i], 'bd') #plot point of interest
    plt. show()
    
    
    