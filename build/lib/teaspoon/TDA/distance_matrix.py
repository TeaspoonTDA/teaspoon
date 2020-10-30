def shortest_path_distance(A, Dmat, weighted = True):
    import numpy as np
    import networkx as nx
    G = nx.Graph()
    s_arr, e_arr, w_arr = [], [], [] #array of starts, ends, and weights
    for s in range(len(Dmat[0])): #finds all connections between nodes
        for e in range(len(Dmat[0])):
            if np.isinf(Dmat[s][e]) == False and Dmat[s][e] != 0: #if a path exists
                s_arr, e_arr, w_arr = np.append(s_arr, s), np.append(e_arr, e), np.append(w_arr, Dmat[s,e]) 
                    
    s_arr, e_arr, w_arr =  s_arr.astype(int), e_arr.astype(int), w_arr.astype(float)  
        
    if weighted == False:
        edges = zip(s_arr, e_arr) # zips edges together
        G.add_edges_from(edges) #adds all edges into graph G
        length = dict(nx.all_pairs_shortest_path_length(G)) #finds shortest distance between edges
            
    if weighted == True:
        edges_w = zip(s_arr, e_arr, w_arr) # zips edges and weights together
        G.add_weighted_edges_from(edges_w) #adds all weighted edges into graph G
        length = dict(nx.all_pairs_dijkstra_path_length(G)) #finds shortest distance between nodes
    Dmat_new = A[:]
    infin = np.nan
    for s in range(len(Dmat[0])):
        for e in range(s+1,len(Dmat[0])):
            if s in length:
                if e in length[s]:
                    ell = length[s][e]
                    Dmat_new[s][e], Dmat_new[e][s] = ell, ell
                else:
                    Dmat_new[s][e], Dmat_new[e][s] = infin, infin
            else:
                Dmat_new[s][e], Dmat_new[e][s] = infin, infin
            
    return Dmat_new
    
def longest_path_distance(A, weighted = True):
    import numpy as np
    import networkx as nx
    A = A + A.T #make undirected adjacency matrix
    np.fill_diagonal(A, 0) #get rid of diagonal for proper weighting (diagonal can have very high values from self transitions)
    Dmat = 1/A[:]
            
    G = nx.Graph()
    s_arr, e_arr, w_arr = [], [], [] #array of starts, ends, and weights
    for s in range(len(Dmat[0])): #finds all connections between nodes
        for e in range(len(Dmat[0])):
            if np.isinf(Dmat[s][e]) == False and Dmat[s][e] != 0: #if a path exists
                s_arr, e_arr, w_arr = np.append(s_arr, s), np.append(e_arr, e), np.append(w_arr, Dmat[s,e]) 
                    
    s_arr, e_arr, w_arr =  s_arr.astype(int), e_arr.astype(int), w_arr.astype(float) 
        
    if weighted == False:
        edges = zip(s_arr, e_arr) # zips edges together
        G.add_edges_from(edges) #adds all edges into graph G
        paths = dict(nx.all_pairs_shortest_path(G)) #finds shortest distance between edges
            
    if weighted == True:
        edges_w = zip(s_arr, e_arr, w_arr) # zips edges and weights together
        G.add_weighted_edges_from(edges_w) #adds all weighted edges into graph G
        paths = dict(nx.all_pairs_dijkstra_path(G)) #finds shortest path between nodes
        
    Dmat_new = A[:]
    for s in range(len(Dmat[0])):
        for e in range(len(Dmat[0])):
            if Dmat[s][e] ==  0 or np.isinf(Dmat[s][e]) == True: #if a direct connection doesn't exist
                path_ij = paths[s][e]
                path_steps = []
                path_steps.append(path_ij[:-1])
                path_steps.append(path_ij[1:])
                path_steps = np.array(path_steps)
                indices = (np.array(path_steps[0]).astype(int), np.array(path_steps[1]).astype(int))
                path_dist = np.sum(A[indices])
                Dmat_new[s][e] = path_dist
    return Dmat_new
    
def resistence_distance(Dmat, weighted = True):
    import numpy as np
    import networkx as nx
    G = nx.Graph()
    s_arr, e_arr, w_arr = [], [], [] #array of starts, ends, and weights
    for s in range(len(Dmat[0])): #finds all connections between nodes
        for e in range(len(Dmat[0])):
            if np.isinf(Dmat[s][e]) == False and Dmat[s][e] != 0: #if a path exists
                s_arr, e_arr, w_arr = np.append(s_arr, s), np.append(e_arr, e), np.append(w_arr, Dmat[s,e]) 
                    
    s_arr, e_arr, w_arr =  s_arr.astype(int), e_arr.astype(int), w_arr.astype(float) 
        
    edges = zip(s_arr, e_arr, w_arr) # zips edges together
    G.add_weighted_edges_from(edges) #adds all weighted edges into graph G
    Dmat_res = Dmat[:]
    for s in range(len(Dmat[0])):
        for e in range(s+1,len(Dmat[0])):
            if weighted == True:
                D = nx.resistance_distance(G, nodeA = s, nodeB = e, weight = 'weight', invert_weight = False)
            else:
                D = nx.resistance_distance(G, nodeA = s, nodeB = e)
            Dmat_res[s][e], Dmat_res[e][s] = D, D
    np.fill_diagonal(Dmat_res, 0)
    return Dmat_res
    
    
def DistanceMatrix(A, method = 'inverse', distance = 'shortest_path'):
    #inputs: A = weighted, directional adjacency matrix, weighted = weighting on A for calculating distance matrix 
    import numpy as np
    A = A + A.T #make undirected adjacency matrix
    np.fill_diagonal(A, 0)
    
    methods = ['unweighted', 'simple', 'difference', 'inverse']
    
    if method not in methods:
        print('Error: method listed for distance matrix not available.')
        print('Defaulting to unweighted')
        method = 'unweighted'
            
    if method == 'simple':
        Dmat = A
        Dmat[Dmat==0] = np.inf
        np.fill_diagonal(Dmat, np.inf) #gets rid of diagonal byu setting distance to infinity
            
    if method == 'difference':
        np.fill_diagonal(A, 0) #get rid of diagonal for proper weighting (diagonal can have very high values from self transitions)
        Amax = np.nanmax(A) #finds max of A for transformation into distance matrix
        Dmat = (Amax - A + 1) # flips distance matrix so that high edge weights = short distance
        Dmat[Dmat==np.nanmax(Dmat)] = np.inf  #replaces max distance with infinity because this would represent no connection
        np.fill_diagonal(Dmat, np.inf) #gets rid of diagonal byu setting distance to infinity
            
    if method == 'inverse':
        np.seterr(divide='ignore') #ignores divide by zero warning
        np.fill_diagonal(A, 0) #set diagonal to zero
        Dmat = 1/A #take element-wise inverse
            
    if method == 'unweighted':
        A[A>0] = 1 #sets all edge weights = 1
        Dmat = A 
        np.fill_diagonal(Dmat, 0) #get rid of diagonal as we don't care about self transitions
        
    if distance == 'shortest_path': Dmat = shortest_path_distance(A, Dmat)   
    if distance == 'longest_path': Dmat = longest_path_distance(Dmat)  
    if distance == 'resistance': Dmat = resistence_distance(Dmat)   
    
    return Dmat