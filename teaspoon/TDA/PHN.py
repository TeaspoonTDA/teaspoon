def point_summaries(diagram, A):
    
    """This function calculates the persistent homology statistics for a graph from the paper "Persistent Homology of Complex Networks for Dynamic State Detection."
    
    Args:
        A (2-D array): 2-D square adjacency matrix
        diagram (list): persistence diagram from ripser from a graph's distance matrix
        
    Returns:
        [array 1-D]: statistics (R, En M) as (maximum persistence ratio, persistent entropy normalized, homology class ratio)
    """
    
    
    import numpy as np
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
        
    lt = np.array(diagram[1].T[1]-diagram[1].T[0])
    D1 = np.array([diagram[1].T[0], diagram[1].T[1]])
    D1 = D1
    lt = lt[lt<10**10]
    num_lifetimes1 = len(lt)
    num_unique = len(A[0])
    En = persistentEntropy(lt)
    
    H1 = diagram[1].T
    delta = 0.1
    if len(H1[H1<10**10])>0:
        R = (1-np.nanmax(H1)/np.floor((num_unique/3)-delta))
    else:
        R = np.nan
    M = num_lifetimes1/num_unique
    statistics = [R, En, M]
    return statistics



def PH_network(A, method = 'unweighted', distance = 'shortest_path'):
    
    """This function calculates the persistent homology of the graph represented by the adjacency matrix A using a distance algorithm defined by user.
    
    Args:
        A (2-D array): 2-D square adjacency matrix
        
    Other Parameters:
        method (Optional[string]): either 'unweighted', 'simple', 'inverse', or 'difference'. Default is 'unweighted'.
        distance (Optional[string]): either 'shortest_path', 'longest_path' (if using 'simple' distance), or 'resistance'. Default is 'shortest_path'.
    
    Returns:
        [square matrix (2-D array), list]: D (distance matrix), diagram (persistence diagram -- 0 and 1 dimension)
    """
    
    #import packages
    from ripser import ripser
    import numpy as np
    
    #import sub modules
    import os,sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__),'..','..'))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__),'..'))
    from teaspoon.TDA import distance_matrix
    
    D = distance_matrix.DistanceMatrix(np.array(A), method = method, distance = distance)  
    #get distance matrix. Specify if weighting is desired or shortest path 
    
    from scipy import sparse
    D_sparse = sparse.coo_matrix(D).tocsr()
    result = ripser(D_sparse, distance_matrix=True, maxdim=1)
    diagram = result['dgms']
        
    return D, diagram


# In[ ]:
    
if __name__ == "__main__": #Only runs if running from this file (This will show basic example)
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__),'..', '..'))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__),'..'))


    #----------------------------Example Simple--------------------------------------
    
    #import needed packages
    import numpy as np
    from teaspoon.SP.network import knn_graph
    from teaspoon.TDA.PHN import PH_network
    
    #generate a siple sinusoidal time series
    t = np.linspace(0,30,300)
    ts = np.sin(t) + np.sin(2*t)
    
    #create adjacency matrix, this 
    A = knn_graph(ts)
    
    #create distance matrix and calculate persistence diagram
    D, diagram = PH_network(A, method = 'unweighted', distance = 'shortest_path') 
    
    print('1-D Persistent Homology (loops): ', diagram[1])
    
    stats = point_summaries(diagram, A)
    print('Persistent homology of network statistics: ', stats)
    
    
    #---------------------------Complex Example---------------------------------------
    
    #import needed packages
    import numpy as np
    from teaspoon.SP.network import ordinal_partition_graph
    from teaspoon.TDA.PHN import PH_network
    from teaspoon.SP.network_tools import make_network
    from teaspoon.parameter_selection.MsPE import MsPE_tau
    
    #generate a siple sinusoidal time series
    t = np.linspace(0,30,300)
    ts = np.sin(t) + np.sin(2*t)
    
    #Get appropriate dimension and delay parameters for permutations
    tau = int(MsPE_tau(ts)) 
    n = 5
    
    #create adjacency matrix, this 
    A = ordinal_partition_graph(ts, n, tau)
    
    #get networkx representation of network for plotting
    G, pos = make_network(A, position_iterations = 1000, remove_deg_zero_nodes = True)
    
    #create distance matrix and calculate persistence diagram
    D, diagram = PH_network(A, method = 'unweighted', distance = 'shortest_path') 
    
    
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import networkx as nx

    TextSize = 14
    plt.figure(2) 
    plt.figure(figsize=(8,8))
    gs = gridspec.GridSpec(4, 2) 
    
    ax = plt.subplot(gs[0:2, 0:2]) #plot time series
    plt.title('Time Series', size = TextSize)
    plt.plot(ts, 'k')
    plt.xticks(size = TextSize)
    plt.yticks(size = TextSize)
    plt.xlabel('$t$', size = TextSize)
    plt.ylabel('$x(t)$', size = TextSize)
    plt.xlim(0,len(ts))
    
    ax = plt.subplot(gs[2:4, 0]) 
    plt.title('Network', size = TextSize)
    nx.draw(G, pos, with_labels=False, font_weight='bold', node_color='blue',
            width=1, font_size = 10, node_size = 30)
    
    ax = plt.subplot(gs[2:4, 1]) 
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
    
    
    
            
          
    