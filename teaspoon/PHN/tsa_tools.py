def permutation_sequence(time_series, m, delay): #finds permutation sequency from modified pyentropy package
    #inputs: time_series = time series (1-d), m = embedding or motif dimension, and delay = embedding delay from function delay_op
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

def embed_time_series(ts, n, tau):
    import numpy as np
    #takens embedding method. Not the fastest algoriothm, but it works. Need to improve
    ets = np.tile(ts,(len(ts),1))
    li = np.tril_indices(len(ts), k = -1)
    ui = np.triu_indices(len(ts), k = tau*(n-1)+1)
    ets[li] = 0
    ets[ui] = 0
    ets = ets[:-(tau*(n-1))]
    a = []
    for i in range(len(ets)):
        b = np.trim_zeros(ets[i])
        a = np.append(a, b[::tau])
    if len(a)%n != 0:
        a = a[len(a)%n:]
        ets = a.reshape(len(ets)-1,n)
    
    else:
        ets = a.reshape(len(ets),n)
    return ets

def k_NN(embedded_time_series, k):
    ETS = embedded_time_series
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(ETS) #get nearest neighbors
    neighbordistances, indices = nbrs.kneighbors(ETS) #get incidices of nearest neighbors
    from scipy.spatial import distance
    distances = distance.cdist(ETS, ETS, 'euclidean') #get euclidean distance between embedded vectors
    return distances, indices