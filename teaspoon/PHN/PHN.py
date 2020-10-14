
def PH_network(ts, tau, n, method = 'unweighted', distance = 'shortest_path', 
               network = 'ordinal_partition', plotting = False, k= 4):
    
    """This function provides a library of dynamical system models to simulate with the time series as the output. 
    
    Args:
        ts (array): 1-D time series.
        tau (int): permutation or embedding delay.
        n (int): permutation or embedding dimension.
        
    Other Parameters:
        method (Optional[string]): either 'unweighted', 'simple', 'inverse', or 'difference'. Default is 'unweighted'.
        distance (Optional[string]): either 'shortest_path', 'longest_path' (if using 'simple' distance), or 'resistance'. Default is 'shortest_path'.
        network (Optional[string]): either 'ordinal_partition' or 'knn' for an ordinal partition network or k nearest neighbors network, respectively. Default is 'ordinal_partition'.
        k (Optional[array]): number of nearest neighbors. Default is k=4.
        plotting (Optional[boolean]): either True or False. Default is False.
    
    Returns:
        [1-D array, 2-D array, 2-D array, list, 1-D array]: PS (permutation sequence if an ordinal partition network is used), A (adjanency matrix), D (distance matrix), diagram (persistence diagram statistics), (maximum persistence ratio, persistent entropy normalized, homology class ratio)
    """
    
    #import packages
    import numpy as np
    import scipy
    from ripser import ripser
    
    #import sub modules
    import tsa_tools
    import network_tools
    import adjacency_matrix
    import distance_matrix
    
    ETS = tsa_tools.embed_time_series(ts, n, tau) #get embedded time series
    PS = [None]
    #takens embedding of time series in n dimenional space with delay tau
        
    if network == 'knn':
        distances, indices = tsa_tools.k_NN(ETS, k= k) 
        #gets distances between embedded vectors and the indices of the nearest neighbors for every vector
        
        A = adjacency_matrix.Adjacency_KNN(indices)
        #get adjacency matrix (weighted, directional)
        if plotting == True:
            G, pos = network_tools.MakeNetwork(A)
            #get network graph based on adjacency matrix (unweighted, non-directional)
              
        D = distance_matrix.DistanceMatrix(A, method = method, distance = distance)  
        #get distance matrix. Specify if weighting is desired or shortest path 
        
        from scipy import sparse
        D_sparse = sparse.coo_matrix(D).tocsr()
        result = ripser(D_sparse, distance_matrix=True, maxdim=1)
        diagram = result['dgms']
        
        statistics = network_tools.PHN_statistics(diagram, A) # statistics = [R, En, M]
        
    if network == 'ordinal_partition':
        PS = tsa_tools.permutation_sequence(ts, n, tau)
        
        A = adjacency_matrix.Adjaceny_OP(PS, n)
        A = A[~(A==0).all(1)] #removes all columns with all elements == 0 (node desn't exist)
        A = A[:,~(A==0).all(0)] #removes all rows with all elements == 0 (node desn't exist)
        #gets adjacency matrix from permutation sequence transtitions
        
        D = distance_matrix.DistanceMatrix(A, method = method, distance = distance)   
        #gets distance matrix from adjacency matrix with weighting as option and shortest path as option.
        if plotting == True:
            G, pos = network_tools.MakeNetwork(A)
        #makes graph from adjacency (unweighted, non-directional) matrix
        
        D_sparse = scipy.sparse.coo_matrix(D).tocsr()
        result = ripser(D_sparse, distance_matrix=True, maxdim=1)
        diagram = np.array(result['dgms'])
        
        statistics = network_tools.PHN_statistics(diagram, A) # statistics = [R, En, M]
    
    if plotting == True:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        import networkx as nx
        TextSize = 14
        plt.figure(2) 
        plt.figure(figsize=(13,11))
        gs = gridspec.GridSpec(5, 3) 
        
        ht = 2
        if network == 'ordinal_partition': ht = 1
        ax = plt.subplot(gs[0:ht, 0:2]) #plot time series
        plt.title('Time Series', size = TextSize)
        plt.plot(ts, 'k')
        plt.xticks(size = TextSize)
        plt.yticks(size = TextSize)
        plt.xlabel('$t$', size = TextSize)
        plt.ylabel('$x(t)$', size = TextSize)
        plt.xlim(0,len(ts))
        
        if network == 'ordinal_partition':
            ax = plt.subplot(gs[1, 0:2]) #plot 
            plt.title('Permutation Sequence', size = TextSize)
            plt.plot(PS, 'k')
            plt.scatter(np.arange(len(PS)), PS, 
                            c = PS, cmap = plt.get_cmap("viridis"), 
                            s = 5, zorder = 10)
            plt.xticks(size = TextSize)
            plt.yticks(size = TextSize)
            plt.xlabel('$i$', size = TextSize)
            plt.ylabel('$\pi_i$  $(n='+str(n)+')$', size = TextSize)
            plt.xlim(0,len(PS))
        
        ax = plt.subplot(gs[0:2, 2]) 
        plt.title('Takens Embedded (2D)', size = TextSize)
        plt.plot(ETS.T[0],ETS.T[1])
        plt.xlabel(r'$x(t)$', size = TextSize)
        plt.ylabel(r'$x(t+\tau)$', size = TextSize)
        plt.xticks(size = TextSize)
        plt.yticks(size = TextSize)
        
        
        ax = plt.subplot(gs[2:4, 0]) 
        plt.title('Network', size = TextSize)
        nx.draw(G, pos, with_labels=False, font_weight='bold', node_color='blue',
                width=1, font_size = 10, node_size = 30)
        
        
        ax = plt.subplot(gs[2:4, 1]) 
        plt.title('Distance Matrix', size = TextSize)
        plt.imshow(D)
        plt.colorbar()
        plt.xticks(size = TextSize)
        plt.yticks(size = TextSize)
        
        ax = plt.subplot(gs[2:4, 2]) 
        plt.title('Persistence Diagram', size = TextSize)
        MS = 3
        top = max(diagram[1].T[1])
        plt.plot([0,top*1.25],[0,top*1.25],'k--')
        plt.yticks( size = TextSize)
        plt.xticks(size = TextSize)
        plt.xlabel('Birth', size = TextSize)
        plt.ylabel('Death', size = TextSize)
        plt.plot(diagram[1].T[0],diagram[1].T[1] ,'go', markersize = MS+2)
        plt.xlim(0,top*1.25)
        plt.ylim(0,top*1.25)
        
        plt.subplots_adjust(hspace= 0.8)
        plt.subplots_adjust(wspace= 0.35)
        plt.show()
        
    return PS, A, D, diagram, statistics


# In[ ]:
    
if __name__ == "__main__": #Only runs if running from this file (This will show basic example)
    
    #import needed packages
    import numpy as np
    t = np.linspace(0,30,200)
    ts = np.sin(t) + np.sin(2*t) #generate a simple time series
    
    n, tau = 5, 6
    PS, A, D, diagram, statistics = PH_network(ts, tau, n, network = 'knn',
                                               method = 'unweighted', distance = 'shortest_path',
                                               plotting = True)
    # PS = permutation sequence, A = adjacency matrix, D = distance matrix, 
    # diagram = persistence diagram, and 
    # statistics =  [maximum persistence ratio, persistent entropy normalized, homology class ratio]
    
    
            
          
    